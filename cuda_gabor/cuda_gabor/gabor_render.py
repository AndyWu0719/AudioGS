"""
Python wrapper for CUDA Gabor renderer.

Provides GaborRendererCUDA class with torch.autograd.Function for seamless
integration with PyTorch training loops.
"""

import torch
import torch.nn as nn
from typing import Tuple

# Import compiled CUDA extension (built into this package)
try:
    from . import _C
    CUDA_EXT_AVAILABLE = True
except ImportError as e:
    CUDA_EXT_AVAILABLE = False
    print(f"[cuda_gabor] Warning: CUDA extension not compiled. Run: pip install -e cuda_gabor/ (Error: {e})")


class GaborRenderFunction(torch.autograd.Function):
    """
    Custom autograd function for CUDA Gabor rendering.
    """
    
    @staticmethod
    def forward(
        ctx,
        amplitude: torch.Tensor,
        tau: torch.Tensor,
        omega: torch.Tensor,
        sigma: torch.Tensor,
        phi: torch.Tensor,
        gamma: torch.Tensor,
        num_samples: int,
        sample_rate: float,
        sigma_multiplier: float,
    ) -> torch.Tensor:
        """Forward pass using CUDA kernel."""
        
        # Save for backward
        ctx.save_for_backward(
            amplitude.contiguous(),
            tau.contiguous(),
            omega.contiguous(),
            sigma.contiguous(),
            phi.contiguous(),
            gamma.contiguous(),
        )
        ctx.sample_rate = sample_rate
        ctx.sigma_multiplier = sigma_multiplier
        
        # Call CUDA kernel
        output = _C.forward(
            amplitude.contiguous(),
            tau.contiguous(),
            omega.contiguous(),
            sigma.contiguous(),
            phi.contiguous(),
            gamma.contiguous(),
            num_samples,
            sample_rate,
            sigma_multiplier,
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using CUDA kernel."""
        
        amplitude, tau, omega, sigma, phi, gamma = ctx.saved_tensors
        sample_rate = ctx.sample_rate
        sigma_multiplier = ctx.sigma_multiplier
        
        # Call CUDA backward kernel
        grads = _C.backward(
            amplitude,
            tau,
            omega,
            sigma,
            phi,
            gamma,
            grad_output.contiguous(),
            sample_rate,
            sigma_multiplier,
        )
        
        # Return gradients (None for non-tensor args: num_samples, sample_rate, sigma_mult)
        return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], None, None, None


class GaborRendererCUDA(nn.Module):
    """
    CUDA-accelerated Gabor atom renderer.
    
    Uses custom CUDA kernels for maximum performance.
    Expected speedup: 2-3x compared to PyTorch scatter_add.
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        sigma_multiplier: float = 5.0,  # 5σ ≈ -108dB truncation (avoids hissing noise from 3σ ≈ -39dB)
    ):

        super().__init__()
        self.sample_rate = float(sample_rate)
        self.sigma_multiplier = float(sigma_multiplier)
        
        if not CUDA_EXT_AVAILABLE:
            raise RuntimeError(
                "CUDA extension not available. Build with:\n"
                "  cd cuda_gabor && pip install -e ."
            )
    
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
        Render Gabor atoms to waveform.
        
        Args:
            amplitude: [N] Atom amplitudes
            tau: [N] Temporal centers (seconds)
            omega: [N] Frequencies (Hz)
            sigma: [N] Envelope widths (seconds)
            phi: [N] Phases (radians)
            gamma: [N] Chirp rates (Hz/s)
            num_samples: Output waveform length
            
        Returns:
            Waveform tensor [num_samples]
        """
        return GaborRenderFunction.apply(
            amplitude, tau, omega, sigma, phi, gamma,
            num_samples, self.sample_rate, self.sigma_multiplier,
        )


def get_cuda_gabor_renderer(sample_rate: int = 24000) -> GaborRendererCUDA:
    """Factory function to get CUDA renderer."""
    if not CUDA_EXT_AVAILABLE:
        raise RuntimeError("CUDA extension not available")
    print("[Renderer] Using C++/CUDA extension")
    return GaborRendererCUDA(sample_rate=sample_rate)
