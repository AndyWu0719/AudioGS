"""
CPU Renderer for Gabor Atoms.

This module provides the PyTorch CPU fallback renderer, matching CUDA behavior.
Moved to src/renderers/ for proper import across all entry points.

Key features matching CUDA:
- Windowed computation (only render within sigma_multiplier * sigma)
- Hann taper in outer 20% of window
- 1ms sigma clamp
"""
import torch
from typing import Optional


def render_pytorch(
    amplitude: torch.Tensor,
    tau: torch.Tensor,
    omega: torch.Tensor,
    sigma: torch.Tensor,
    phi: torch.Tensor,
    gamma: torch.Tensor,
    num_samples: int,
    sample_rate: int,
    device: torch.device,
    sigma_multiplier: float = 5.0
) -> torch.Tensor:
    """
    PyTorch fallback renderer matching CUDA behavior.
    
    Args:
        amplitude: [N] amplitude values
        tau: [N] time center positions (seconds)
        omega: [N] frequencies (Hz)
        sigma: [N] envelope widths (seconds)
        phi: [N] phase values (radians)
        gamma: [N] chirp rates
        num_samples: output waveform length
        sample_rate: audio sample rate
        device: computation device
        sigma_multiplier: window truncation (default 5.0 = 5σ)
    
    Returns:
        [num_samples] rendered waveform
    """
    PI = 3.14159265358979
    TWO_PI = 2 * PI
    
    t = torch.arange(num_samples, device=device, dtype=torch.float32) / sample_rate
    output = torch.zeros(num_samples, device=device)
    
    for i in range(len(amplitude)):
        A = amplitude[i]
        
        # Skip negligible amplitude (match CUDA)
        if abs(A.item()) < 1e-8:
            continue
        
        tau_i = tau[i]
        omega_i = omega[i]
        # Match CUDA 1ms sigma clamp
        sigma_i = torch.clamp(sigma[i], min=0.001)
        phi_i = phi[i]
        gamma_i = gamma[i]
        
        sigma_sq = sigma_i * sigma_i
        window_bound = sigma_i * sigma_multiplier
        
        # Compute window bounds in samples (match CUDA)
        tau_i_scalar = tau_i.item()
        window_bound_scalar = window_bound.item()
        window_start = max(0, int((tau_i_scalar - window_bound_scalar) * sample_rate))
        window_end = min(num_samples - 1, int((tau_i_scalar + window_bound_scalar) * sample_rate))
        
        if window_start > window_end:
            continue
        
        # Only compute within window (optimization + match CUDA)
        t_window = t[window_start:window_end+1]
        t_centered = t_window - tau_i
        t_sq = t_centered ** 2
        
        # Envelope
        envelope = torch.exp(-t_sq / (2.0 * sigma_sq))
        
        # Hann taper in outer 20% of window (match CUDA)
        normalized_dist = torch.abs(t_centered) / window_bound
        window_factor = torch.ones_like(normalized_dist)
        outer_mask = normalized_dist > 0.8
        if outer_mask.any():
            edge_t = (normalized_dist[outer_mask] - 0.8) / 0.2
            window_factor[outer_mask] = 0.5 * (1.0 + torch.cos(PI * edge_t))
        
        # Phase with chirp (match CUDA conventions)
        phase = TWO_PI * (omega_i * t_centered + 0.5 * gamma_i * t_sq) + phi_i
        carrier = torch.cos(phase)
        
        # Accumulate (match CUDA atomicAdd)
        output[window_start:window_end+1] += A * envelope * window_factor * carrier
    
    return output
