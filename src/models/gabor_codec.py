"""
Gabor-frame codec (AE / VAE) built on top of a deterministic STFT/ISTFT.

Design goals:
  - Keep the "Gabor atoms" foundation explicit (STFT coefficients).
  - Learn a compact latent sequence z that is easy for Flow to model.
  - Avoid any continuous-atom renderer; decoding uses ISTFT only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.gabor_frame import (
    GaborFrameConfig,
    frame_features_to_stft,
    istft,
    num_frames,
    stft,
    stft_to_frame_features,
)


class ResidualConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.gelu(self.norm2(x)))
        return x + residual


@dataclass
class CodecLossWeights:
    time_l1: float = 1.0
    stft_mss: float = 0.5
    stft_feat_l1: float = 0.0
    complex_stft: float = 0.0
    gabor_stft_complex: float = 0.0
    frame_consistency: float = 0.0
    kl: float = 1e-4
    latent_l1: float = 0.0


class GaborFrameCodec(nn.Module):
    """
    Latent codec: waveform -> STFT -> latent z -> STFT_hat -> ISTFT -> waveform_hat.

    z is a sequence [B, Tz, Dz] with a lower frame-rate than STFT frames (controlled by
    `time_downsample`), making it suitable for Flow modeling.
    """

    def __init__(
        self,
        gabor_cfg: GaborFrameConfig,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 6,
        time_downsample: int = 4,
        use_vae: bool = False,
        dropout: float = 0.0,
        dilation_schedule: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        if time_downsample <= 0:
            raise ValueError("time_downsample must be > 0")

        self.gabor_cfg = gabor_cfg
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_downsample = time_downsample
        self.use_vae = use_vae

        in_dim = 2 * gabor_cfg.freq_bins  # real/imag flattened per frame

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        if dilation_schedule is None:
            dilations = [1] * num_layers
        else:
            dilations = list(dilation_schedule)
            if len(dilations) != num_layers:
                raise ValueError(f"dilation_schedule length {len(dilations)} != num_layers {num_layers}")

        self.enc_blocks = nn.ModuleList(
            [
                ResidualConv1d(hidden_dim, kernel_size=5, dilation=dilations[i], dropout=dropout)
                for i in range(num_layers)
            ]
        )

        self.to_mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1) if use_vae else None

        self.from_latent = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.dec_blocks = nn.ModuleList(
            [
                ResidualConv1d(hidden_dim, kernel_size=5, dilation=dilations[i], dropout=dropout)
                for i in range(num_layers)
            ]
        )
        self.out_proj = nn.Conv1d(hidden_dim, in_dim, kernel_size=1)

    @torch.no_grad()
    def stft_frames_for_num_samples(self, num_samples: int) -> int:
        return num_frames(num_samples, self.gabor_cfg)

    def _encode_frames(self, frame_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            frame_features: [B, T, 2*F]
        Returns:
            mu: [B, Tz, Dz]
            logvar: [B, Tz, Dz] or None
        """
        B, T, C = frame_features.shape
        x = self.in_proj(frame_features)  # [B, T, H]
        x = x.transpose(1, 2).contiguous()  # [B, H, T]

        for block in self.enc_blocks:
            x = block(x)

        # Downsample time to latent frame-rate.
        x = F.avg_pool1d(x, kernel_size=self.time_downsample, stride=self.time_downsample, ceil_mode=True)

        mu = self.to_mu(x).transpose(1, 2).contiguous()  # [B, Tz, Dz]
        logvar = None
        if self.use_vae and self.to_logvar is not None:
            logvar = self.to_logvar(x).transpose(1, 2).contiguous()
        return mu, logvar

    def encode(
        self, waveform: torch.Tensor, sample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveform: [B, T] or [T]
            sample: If True and use_vae, sample z ~ N(mu, exp(logvar))
        Returns:
            z: [B, Tz, Dz]
            logvar: [B, Tz, Dz] or None
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        X = stft(waveform, self.gabor_cfg)  # [B, F, TT]
        feats = stft_to_frame_features(X)  # [B, TT, 2F]
        mu, logvar = self._encode_frames(feats)
        z = mu
        if self.use_vae and sample and logvar is not None:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        return z, mu, logvar

    def _decode_to_frames(self, z: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        Args:
            z: [B, Tz, Dz]
            target_frames: TT
        Returns:
            frame_features_hat: [B, TT, 2F]
        """
        x = z.transpose(1, 2).contiguous()  # [B, Dz, Tz]
        x = self.from_latent(x)  # [B, H, Tz]

        # Upsample to STFT frame-rate.
        x = F.interpolate(x, size=target_frames, mode="linear", align_corners=False)

        for block in self.dec_blocks:
            x = block(x)

        out = self.out_proj(x)  # [B, 2F, TT]
        return out.transpose(1, 2).contiguous()  # [B, TT, 2F]

    def decode_to_stft(self, z: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Decode latent to complex STFT with the correct frame count for `num_samples`.
        Returns: [B, F, TT] complex
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)
        target_frames = num_frames(num_samples, self.gabor_cfg)
        feats_hat = self._decode_to_frames(z, target_frames)
        return frame_features_to_stft(feats_hat, freq_bins=self.gabor_cfg.freq_bins)

    def decode(self, z: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Decode latent to waveform [B, T]."""
        X_hat = self.decode_to_stft(z, num_samples=num_samples)
        y = istft(X_hat, self.gabor_cfg, length=num_samples)
        return y

    def forward(self, waveform: torch.Tensor, sample_latent: bool = False, return_features: bool = False) -> dict:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        _, T = waveform.shape

        # Compute analysis STFT once.
        X = stft(waveform, self.gabor_cfg)  # [B, F, TT]
        feats = stft_to_frame_features(X)  # [B, TT, 2F]
        mu, logvar = self._encode_frames(feats)

        z = mu
        if self.use_vae and sample_latent and logvar is not None:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)

        X_hat = self.decode_to_stft(z, num_samples=T)
        recon = istft(X_hat, self.gabor_cfg, length=T)

        out = {"recon": recon, "z": z, "mu": mu, "logvar": logvar}
        if return_features:
            out["X"] = X
            out["X_hat"] = X_hat
            feats_hat = stft_to_frame_features(X_hat)
            out["stft_feats"] = feats
            out["stft_feats_hat"] = feats_hat
        return out

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(q(z|x) || N(0,1)) for diagonal Gaussian per time step.
        return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)
