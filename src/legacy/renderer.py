"""
Differentiable Gabor Atom Renderer.

Implements the Gabor atom equation:
g_i(t) = A * exp(-(t-τ)²/(2σ²)) * cos(2π(ω(t-τ) + ½γ(t-τ)²) + φ)

Uses vectorized scatter_add for efficient overlap-add rendering,
computing only within [τ-3σ, τ+3σ] for each atom to save memory.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class GaborRenderer(nn.Module):
    """
    Differentiable renderer for Gabor atoms.
    
    Uses chunked/windowed rasterization with vectorized scatter_add
    to efficiently render atom contributions onto the waveform.
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        sigma_multiplier: float = 3.0,
        max_chunk_size: int = 4096,
    ):
        """
        Initialize the Gabor renderer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            sigma_multiplier: Multiple of sigma to use for window bounds
            max_chunk_size: Maximum samples per atom window (memory limit)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.sigma_multiplier = sigma_multiplier
        self.max_chunk_size = max_chunk_size
        
    def forward(
        self,
        amplitude: torch.Tensor,
        tau: torch.Tensor,
        omega: torch.Tensor,
        sigma: torch.Tensor,
        phi: torch.Tensor,
        gamma: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Render atoms to waveform using vectorized scatter_add.
        
        Args:
            amplitude: [N] Atom amplitudes
            tau: [N] Atom temporal centers (seconds)
            omega: [N] Atom frequencies (Hz)
            sigma: [N] Atom envelope widths (seconds)
            phi: [N] Atom phases (radians)
            gamma: [N] Atom chirp rates (Hz/s)
            num_samples: Output waveform length
            
        Returns:
            Rendered waveform of shape [num_samples]
        """
        device = amplitude.device
        num_atoms = amplitude.shape[0]
        
        if num_atoms == 0:
            return torch.zeros(num_samples, device=device)
        
        # Convert time parameters to samples
        tau_samples = tau * self.sample_rate  # [N]
        sigma_samples = sigma * self.sample_rate  # [N]
        
        # Compute window bounds for each atom [tau - 3*sigma, tau + 3*sigma]
        half_window = (sigma_samples * self.sigma_multiplier).clamp(min=1, max=self.max_chunk_size / 2)
        start_samples = (tau_samples - half_window).long().clamp(min=0)
        end_samples = (tau_samples + half_window).long().clamp(max=num_samples - 1)
        
        # Calculate maximum window size needed
        max_window_size = int((half_window.max() * 2).item()) + 1
        max_window_size = min(max_window_size, self.max_chunk_size)
        
        # Create local time indices for each atom
        # Shape: [N, max_window_size]
        local_indices = torch.arange(max_window_size, device=device).unsqueeze(0)  # [1, W]
        local_indices = local_indices.expand(num_atoms, -1)  # [N, W]
        
        # Compute global sample indices
        global_indices = start_samples.unsqueeze(1) + local_indices  # [N, W]
        
        # Create validity mask (within bounds and within atom's window)
        valid_mask = (global_indices >= 0) & (global_indices < num_samples)
        window_lengths = end_samples - start_samples + 1  # [N]
        valid_mask = valid_mask & (local_indices < window_lengths.unsqueeze(1))
        
        # Convert global indices to time in seconds
        t_global = global_indices.float() / self.sample_rate  # [N, W]
        
        # Compute (t - tau) for each sample
        t_centered = t_global - tau.unsqueeze(1)  # [N, W]
        
        # Compute Gaussian envelope: exp(-(t-τ)²/(2σ²))
        sigma_expanded = sigma.unsqueeze(1)  # [N, 1]
        envelope = torch.exp(-t_centered.pow(2) / (2 * sigma_expanded.pow(2) + 1e-8))
        
        # Compute carrier: cos(2π(ω(t-τ) + ½γ(t-τ)²) + φ)
        omega_expanded = omega.unsqueeze(1)  # [N, 1]
        phi_expanded = phi.unsqueeze(1)  # [N, 1]
        gamma_expanded = gamma.unsqueeze(1)  # [N, 1]
        
        phase = (
            2 * math.pi * (
                omega_expanded * t_centered +
                0.5 * gamma_expanded * t_centered.pow(2)
            ) + phi_expanded
        )
        carrier = torch.cos(phase)
        
        # Compute atom contributions
        amplitude_expanded = amplitude.unsqueeze(1)  # [N, 1]
        contributions = amplitude_expanded * envelope * carrier  # [N, W]
        
        # Apply validity mask
        contributions = contributions * valid_mask.float()
        
        # Flatten for scatter_add
        flat_contributions = contributions.reshape(-1)  # [N * W]
        flat_indices = global_indices.reshape(-1).clamp(0, num_samples - 1)  # [N * W]
        flat_mask = valid_mask.reshape(-1)  # [N * W]
        
        # Only scatter valid contributions
        flat_contributions = flat_contributions * flat_mask.float()
        
        # Use scatter_add for vectorized accumulation
        waveform = torch.zeros(num_samples, device=device)
        waveform = waveform.scatter_add(0, flat_indices, flat_contributions)
        
        return waveform
    
    def render_single_atom(
        self,
        amplitude: float,
        tau: float,
        omega: float,
        sigma: float,
        phi: float,
        gamma: float,
        num_samples: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Render a single atom (for visualization).
        
        Args:
            amplitude, tau, omega, sigma, phi, gamma: Atom parameters
            num_samples: Output length
            device: Target device
            
        Returns:
            Waveform tensor
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return self.forward(
            amplitude=torch.tensor([amplitude], device=device),
            tau=torch.tensor([tau], device=device),
            omega=torch.tensor([omega], device=device),
            sigma=torch.tensor([sigma], device=device),
            phi=torch.tensor([phi], device=device),
            gamma=torch.tensor([gamma], device=device),
            num_samples=num_samples,
        )
    
    def render_batch(
        self,
        amplitude: torch.Tensor,
        tau: torch.Tensor,
        omega: torch.Tensor,
        sigma: torch.Tensor,
        phi: torch.Tensor,
        gamma: torch.Tensor,
        num_samples: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Render atoms for a batch of audio samples.
        
        This handles the case where we have batch_size audio samples,
        and atoms are shared or per-sample.
        
        Args:
            All parameters: [B, N] or [N] tensors
            num_samples: Output length per sample
            batch_size: Number of samples in batch
            
        Returns:
            Waveform tensor of shape [B, num_samples]
        """
        # If parameters are 1D, expand for batch
        if amplitude.dim() == 1:
            amplitude = amplitude.unsqueeze(0).expand(batch_size, -1)
            tau = tau.unsqueeze(0).expand(batch_size, -1)
            omega = omega.unsqueeze(0).expand(batch_size, -1)
            sigma = sigma.unsqueeze(0).expand(batch_size, -1)
            phi = phi.unsqueeze(0).expand(batch_size, -1)
            gamma = gamma.unsqueeze(0).expand(batch_size, -1)
        
        # Render each sample in the batch
        waveforms = []
        for b in range(batch_size):
            waveform = self.forward(
                amplitude[b], tau[b], omega[b], sigma[b], phi[b], gamma[b],
                num_samples
            )
            waveforms.append(waveform)
        
        return torch.stack(waveforms, dim=0)


class GaborRendererFast(nn.Module):
    """
    Faster Gabor renderer using fully vectorized operations.
    
    Trades memory for speed by computing all atoms at all time points
    then masking. Best for smaller number of atoms or shorter audio.
    """
    
    def __init__(self, sample_rate: int = 24000, sigma_multiplier: float = 4.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.sigma_multiplier = sigma_multiplier
        
    def forward(
        self,
        amplitude: torch.Tensor,
        tau: torch.Tensor,
        omega: torch.Tensor,
        sigma: torch.Tensor,
        phi: torch.Tensor,
        gamma: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Render using dense computation with masking.
        
        More memory-intensive but potentially faster for small atom counts.
        """
        device = amplitude.device
        num_atoms = amplitude.shape[0]
        
        if num_atoms == 0:
            return torch.zeros(num_samples, device=device)
        
        # Create time grid: [T]
        t = torch.arange(num_samples, device=device).float() / self.sample_rate
        
        # Expand for broadcasting: [N, T]
        t = t.unsqueeze(0)  # [1, T]
        tau_exp = tau.unsqueeze(1)  # [N, 1]
        sigma_exp = sigma.unsqueeze(1)  # [N, 1]
        omega_exp = omega.unsqueeze(1)  # [N, 1]
        phi_exp = phi.unsqueeze(1)  # [N, 1]
        gamma_exp = gamma.unsqueeze(1)  # [N, 1]
        amplitude_exp = amplitude.unsqueeze(1)  # [N, 1]
        
        # Compute centered time
        t_centered = t - tau_exp  # [N, T]
        
        # Create sparse mask: only compute within [tau - k*sigma, tau + k*sigma]
        mask = (t_centered.abs() <= self.sigma_multiplier * sigma_exp).float()
        
        # Gaussian envelope
        envelope = torch.exp(-t_centered.pow(2) / (2 * sigma_exp.pow(2) + 1e-8))
        
        # Carrier with chirp
        phase = 2 * math.pi * (
            omega_exp * t_centered +
            0.5 * gamma_exp * t_centered.pow(2)
        ) + phi_exp
        carrier = torch.cos(phase)
        
        # Compute atom contributions with mask
        atoms = amplitude_exp * envelope * carrier * mask  # [N, T]
        
        # Sum across atoms
        waveform = atoms.sum(dim=0)  # [T]
        
        return waveform
