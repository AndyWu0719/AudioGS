"""
Gabor frame utilities (STFT/ISTFT).

This module is the shared backbone for Stage00 (exact reconstruction) and the
Stage01+ codec (AE/VAE) that learns a compact latent representation on top of a
deterministic Gabor frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch

WindowType = Literal["hann", "gaussian"]


@dataclass(frozen=True)
class GaborFrameConfig:
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int = 1024
    window: WindowType = "gaussian"
    gaussian_std_frac: float = 0.125
    center: bool = True
    pad_mode: str = "reflect"
    periodic: bool = True

    @property
    def freq_bins(self) -> int:
        return self.n_fft // 2 + 1


def make_window(cfg: GaborFrameConfig, device: torch.device) -> torch.Tensor:
    win_length = int(cfg.win_length)
    if cfg.window == "hann":
        return torch.hann_window(win_length, periodic=cfg.periodic, device=device)
    if cfg.window == "gaussian":
        std_frac = float(cfg.gaussian_std_frac)
        if std_frac <= 0:
            raise ValueError("gaussian_std_frac must be > 0")
        std = std_frac * win_length
        n = torch.arange(win_length, device=device, dtype=torch.float32)
        center = (win_length - 1) / 2.0
        return torch.exp(-0.5 * ((n - center) / std) ** 2)
    raise ValueError(f"Unsupported window type: {cfg.window}")


def num_frames(num_samples: int, cfg: GaborFrameConfig) -> int:
    """Compute STFT frame count for a 1D signal of length `num_samples`."""
    if num_samples <= 0:
        return 0
    if cfg.center:
        pad = cfg.n_fft // 2
    else:
        pad = 0
    effective = num_samples + 2 * pad
    if effective < cfg.n_fft:
        return 1
    return (effective - cfg.n_fft) // cfg.hop_length + 1


def stft(waveform: torch.Tensor, cfg: GaborFrameConfig) -> torch.Tensor:
    """
    Compute complex STFT.

    Args:
        waveform: [T] or [B, T]
    Returns:
        stft: [B, F, TT] complex64/complex32
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    window = make_window(cfg, device=waveform.device)
    x = torch.stft(
        waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        return_complex=True,
        center=cfg.center,
        pad_mode=cfg.pad_mode,
    )
    return x


def istft(stft_tensor: torch.Tensor, cfg: GaborFrameConfig, length: Optional[int] = None) -> torch.Tensor:
    """
    Invert complex STFT to waveform.

    Args:
        stft_tensor: [B, F, TT] complex
        length: Optional output length in samples
    Returns:
        waveform: [B, T]
    """
    if stft_tensor.dim() == 2:
        stft_tensor = stft_tensor.unsqueeze(0)
    window = make_window(cfg, device=stft_tensor.device)
    y = torch.istft(
        stft_tensor,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=cfg.center,
        length=length,
    )
    return y


def stft_to_frame_features(stft_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert complex STFT to per-frame real features.

    Args:
        stft_tensor: [B, F, TT] complex
    Returns:
        features: [B, TT, 2*F] float
    """
    if stft_tensor.dim() == 2:
        stft_tensor = stft_tensor.unsqueeze(0)
    ri = torch.view_as_real(stft_tensor)  # [B, F, TT, 2]
    ri = ri.permute(0, 2, 1, 3).contiguous()  # [B, TT, F, 2]
    return ri.view(ri.shape[0], ri.shape[1], -1)


def frame_features_to_stft(features: torch.Tensor, freq_bins: int) -> torch.Tensor:
    """
    Convert per-frame real features back to complex STFT.

    Args:
        features: [B, TT, 2*F]
        freq_bins: F
    Returns:
        stft: [B, F, TT] complex
    """
    if features.dim() != 3:
        raise ValueError(f"Expected features [B, TT, 2*F], got {tuple(features.shape)}")
    B, TT, C = features.shape
    if C != 2 * freq_bins:
        raise ValueError(f"Expected last dim {2*freq_bins}, got {C}")
    ri = features.view(B, TT, freq_bins, 2).permute(0, 2, 1, 3).contiguous()  # [B, F, TT, 2]
    return torch.complex(ri[..., 0], ri[..., 1])

