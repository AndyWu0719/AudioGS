"""
Flow Matching for Audio Gaussian Splatting.

Implements Optimal Transport Conditional Flow Matching (OT-CFM) with:
- Mini-batch OT coupling for straightened trajectories
- High-order ODE solvers (RK4, Midpoint) for fast inference

Reference: Lipman et al. "Flow Matching for Generative Modeling" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable, Literal

# For OT coupling
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Warning] scipy not available, OT coupling disabled")


def compute_ot_matching(
    x0: torch.Tensor, 
    x1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Optimal Transport matching within mini-batch.
    
    Reorders x0 to minimize total transport distance to x1.
    This straightens the flow trajectories and improves training.
    
    Args:
        x0: [B, N, D] Source samples (noise)
        x1: [B, N, D] Target samples (data)
        
    Returns:
        x0_matched: [B, N, D] Reordered noise
        x1: [B, N, D] Original data (unchanged)
    """
    if not SCIPY_AVAILABLE:
        return x0, x1
    
    B = x0.shape[0]
    
    # Flatten for distance computation: [B, N*D]
    x0_flat = x0.view(B, -1)
    x1_flat = x1.view(B, -1)
    
    # Compute pairwise distance matrix: [B, B]
    # dist[i, j] = ||x0[i] - x1[j]||^2
    cost_matrix = torch.cdist(x0_flat, x1_flat, p=2).pow(2)
    
    # Solve assignment problem (on CPU)
    cost_np = cost_matrix.detach().cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost_np)
    
    # Reorder x0 to match x1
    # After matching: x0_matched[i] is paired with x1[i]
    device = x0.device
    perm = torch.tensor(col_idx, device=device, dtype=torch.long)
    
    # x0_matched[i] = x0[perm[i]]
    x0_matched = x0[perm]
    
    return x0_matched, x1


class ConditionalFlowMatching(nn.Module):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).
    
    Learns a vector field v(x, t) that transports samples from
    a simple prior (Gaussian) to the data distribution.
    
    Features:
    - Mini-batch OT coupling for straightened flows
    - Standard CFM loss with optional masking
    
    OT Path: x_t = (1 - (1-σ_min)t) * x_0 + t * x_1
    Target:  u_t = x_1 - (1-σ_min) * x_0
    """
    
    def __init__(
        self,
        sigma_min: float = 1e-4,
        use_ot: bool = True,
    ):
        """
        Initialize CFM.
        
        Args:
            sigma_min: Minimum noise level (for numerical stability)
            use_ot: Whether to use OT coupling (recommended)
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.use_ot = use_ot and SCIPY_AVAILABLE
        
        if use_ot and not SCIPY_AVAILABLE:
            print("[Warning] OT coupling requested but scipy not available")
    
    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time uniformly from [0, 1]."""
        return torch.rand(batch_size, device=device)
    
    def sample_noise(self, x1: torch.Tensor) -> torch.Tensor:
        """Sample x_0 from prior (standard Gaussian)."""
        return torch.randn_like(x1)
    
    def get_xt(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute interpolated sample x_t on the OT path.
        
        Args:
            x0: [B, N, D] Source samples (noise)
            x1: [B, N, D] Target samples (real atoms)
            t: [B] Time values in [0, 1]
            
        Returns:
            x_t: [B, N, D] Interpolated samples
        """
        # Reshape t for broadcasting: [B, 1, 1]
        t = t.view(-1, 1, 1)
        
        # OT path: x_t = (1 - (1-σ_min)t) * x_0 + t * x_1
        xt = (1 - (1 - self.sigma_min) * t) * x0 + t * x1
        
        return xt
    
    def get_target_velocity(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target velocity u_t for OT-CFM.
        
        Args:
            x0: [B, N, D] Source samples (noise)
            x1: [B, N, D] Target samples (real atoms)
            
        Returns:
            u_t: [B, N, D] Target velocity
        """
        # u_t = x_1 - (1 - σ_min) * x_0
        return x1 - (1 - self.sigma_min) * x0
    
    def compute_loss(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute CFM training loss with optional OT coupling.
        
        Args:
            model: Vector field model v(x, t, condition)
            x1: [B, N, D] Target samples (real atoms)
            mask: [B, N] Mask for valid atoms (True = valid)
            **model_kwargs: Additional conditioning for model
            
        Returns:
            loss: Scalar loss value
            info: Dictionary with loss components
        """
        device = x1.device
        batch_size = x1.shape[0]
        
        # Sample time t ~ U[0, 1]
        t = self.sample_t(batch_size, device)
        
        # Sample noise x_0 ~ N(0, 1)
        x0 = self.sample_noise(x1)
        
        # Apply OT coupling if enabled
        if self.use_ot:
            x0, x1 = compute_ot_matching(x0, x1)
        
        # Get interpolated sample x_t
        xt = self.get_xt(x0, x1, t)
        
        # Get target velocity
        ut = self.get_target_velocity(x0, x1)
        
        # Predict velocity with model
        vt = model(xt, t, **model_kwargs)
        
        # Compute MSE loss
        loss = F.mse_loss(vt, ut, reduction='none')  # [B, N, D]
        
        # Apply mask if provided (only loss on valid atoms)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # [B, N, 1]
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss, {'cfm_loss': loss.item(), 'use_ot': self.use_ot}


class FlowODESolver:
    """
    ODE solver for Flow Matching inference.
    
    Integrates the learned vector field from t=0 (noise) to t=1 (data).
    
    Supports multiple integration methods:
    - euler: Simple first-order (baseline)
    - midpoint: Second-order (Heun's method)
    - rk4: Fourth-order Runge-Kutta (recommended)
    """
    
    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 1e-4,
    ):
        """
        Initialize solver.
        
        Args:
            model: Trained vector field model v(x, t, condition)
            sigma_min: Same sigma_min used during training
        """
        self.model = model
        self.sigma_min = sigma_min
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 25,
        method: Literal['euler', 'midpoint', 'rk4'] = 'rk4',
        device: torch.device = None,
        guidance_scale: float = 1.0,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Generate samples using specified ODE solver.
        
        Args:
            shape: Output shape [B, N, D]
            num_steps: Number of integration steps (25 recommended for RK4)
            method: Integration method ('euler', 'midpoint', 'rk4')
            device: Target device
            guidance_scale: CFG scale (1.0 = no guidance)
            **model_kwargs: Conditioning for model
            
        Returns:
            x1: [B, N, D] Generated samples
        """
        if method == 'euler':
            return self.sample_euler(shape, num_steps, device, **model_kwargs)
        elif method == 'midpoint':
            return self.sample_midpoint(shape, num_steps, device, **model_kwargs)
        elif method == 'rk4':
            return self.sample_rk4(shape, num_steps, device, **model_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @torch.no_grad()
    def sample_euler(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: torch.device = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Generate samples using Euler integration.
        
        First-order method. Requires more steps for accuracy.
        Recommended steps: 50-100
        """
        device = device or next(self.model.parameters()).device
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Time steps
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((shape[0],), i / num_steps, device=device)
            
            # Predict velocity
            v = self.model(x, t, **model_kwargs)
            
            # Euler step
            x = x + v * dt
        
        return x
    
    @torch.no_grad()
    def sample_midpoint(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 25,
        device: torch.device = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Generate samples using Midpoint (Heun/RK2) integration.
        
        Second-order method. More accurate than Euler.
        Recommended steps: 25-50
        
        Algorithm:
            x_mid = x_t + v(x_t, t) * dt/2
            x_{t+1} = x_t + v(x_mid, t + dt/2) * dt
        """
        device = device or next(self.model.parameters()).device
        
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = i / num_steps
            t_tensor = torch.full((shape[0],), t, device=device)
            t_mid_tensor = torch.full((shape[0],), t + dt/2, device=device)
            
            # k1: velocity at current point
            k1 = self.model(x, t_tensor, **model_kwargs)
            
            # Midpoint
            x_mid = x + k1 * (dt / 2)
            
            # k2: velocity at midpoint
            k2 = self.model(x_mid, t_mid_tensor, **model_kwargs)
            
            # Full step using midpoint velocity
            x = x + k2 * dt
        
        return x
    
    @torch.no_grad()
    def sample_rk4(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 25,
        device: torch.device = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Generate samples using RK4 (Runge-Kutta 4) integration.
        
        Fourth-order method. Most accurate per step.
        Recommended steps: 10-25 (with OT coupling)
        
        Algorithm:
            k1 = v(x, t)
            k2 = v(x + k1*dt/2, t + dt/2)
            k3 = v(x + k2*dt/2, t + dt/2)
            k4 = v(x + k3*dt, t + dt)
            x_{t+1} = x + (k1 + 2*k2 + 2*k3 + k4) * dt/6
        """
        device = device or next(self.model.parameters()).device
        
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = i / num_steps
            t_tensor = torch.full((shape[0],), t, device=device)
            t_mid_tensor = torch.full((shape[0],), t + dt/2, device=device)
            t_next_tensor = torch.full((shape[0],), t + dt, device=device)
            
            # k1
            k1 = self.model(x, t_tensor, **model_kwargs)
            
            # k2
            k2 = self.model(x + k1 * (dt/2), t_mid_tensor, **model_kwargs)
            
            # k3
            k3 = self.model(x + k2 * (dt/2), t_mid_tensor, **model_kwargs)
            
            # k4
            k4 = self.model(x + k3 * dt, t_next_tensor, **model_kwargs)
            
            # RK4 update
            x = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)
        
        return x
    
    @torch.no_grad()
    def sample_with_cfg(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 25,
        method: Literal['euler', 'midpoint', 'rk4'] = 'rk4',
        device: torch.device = None,
        guidance_scale: float = 2.0,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Generate samples with Classifier-Free Guidance.
        
        CFG enhances conditioning influence for more faithful generation.
        
        Args:
            guidance_scale: CFG scale (1.0 = no guidance, 2.0-4.0 typical)
        """
        if guidance_scale == 1.0:
            return self.sample(shape, num_steps, method, device, **model_kwargs)
        
        device = device or next(self.model.parameters()).device
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        
        # Null conditioning
        null_kwargs = model_kwargs.copy()
        if 'text_embeddings' in model_kwargs:
            null_kwargs['text_embeddings'] = torch.zeros_like(model_kwargs['text_embeddings'])
        
        for i in range(num_steps):
            t = i / num_steps
            t_tensor = torch.full((shape[0],), t, device=device)
            
            # Unconditional and conditional predictions
            v_uncond = self.model(x, t_tensor, **null_kwargs)
            v_cond = self.model(x, t_tensor, **model_kwargs)
            
            # CFG combination
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            
            # Simple Euler step (could extend to RK4)
            x = x + v * dt
        
        return x


# Convenience function
def get_cfm_loss_fn(sigma_min: float = 1e-4, use_ot: bool = True) -> ConditionalFlowMatching:
    """Create CFM loss function with optional OT coupling."""
    return ConditionalFlowMatching(sigma_min=sigma_min, use_ot=use_ot)
