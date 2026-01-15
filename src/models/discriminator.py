"""
Multi-Scale and Multi-Period Discriminators for AudioGS.

Following HiFi-GAN architecture for adversarial audio generation.
Reference: https://arxiv.org/abs/2010.05646

Exports:
- MultiPeriodDiscriminator (MPD): Captures periodic patterns
- MultiScaleDiscriminator (MSD): Captures multi-resolution structure
- CombinedDiscriminator: Combines MPD + MSD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from typing import List, Tuple


# ============================================================
# Period Discriminator (for MPD)
# ============================================================

class PeriodDiscriminator(nn.Module):
    """
    Single period discriminator.
    Reshapes 1D waveform into 2D based on period, then applies 2D convolutions.
    """
    
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        
        # Progressively increasing channels
        channels = [1, 32, 128, 512, 1024, 1024]
        
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.convs.append(
                weight_norm(nn.Conv2d(
                    channels[i], channels[i+1],
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1) if i < len(channels) - 2 else (1, 1),
                    padding=((kernel_size - 1) // 2, 0),
                ))
            )
        
        # Final conv for prediction
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Waveform [B, 1, T]
            
        Returns:
            output: Discrimination score [B, 1, T', 1]
            features: List of intermediate features for feature matching
        """
        features = []
        
        # Pad to make length divisible by period
        b, c, t = x.shape
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), mode='reflect')
            t = x.shape[2]
        
        # Reshape to 2D: [B, 1, T] -> [B, 1, T/period, period]
        x = x.view(b, c, t // self.period, self.period)
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        
        return x.flatten(1, -1), features


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD).
    Uses multiple periods [2, 3, 5, 7, 11] to capture different periodic patterns.
    """
    
    def __init__(self, periods: List[int] = None):
        super().__init__()
        self.periods = periods or [2, 3, 5, 7, 11]
        
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in self.periods
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: Waveform [B, T] or [B, 1, T]
            
        Returns:
            outputs: List of discrimination scores
            features: List of feature lists for feature matching
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]
        
        outputs = []
        all_features = []
        
        for disc in self.discriminators:
            out, feats = disc(x)
            outputs.append(out)
            all_features.append(feats)
        
        return outputs, all_features


# ============================================================
# Scale Discriminator (for MSD)
# ============================================================

class ScaleDiscriminator(nn.Module):
    """
    Single scale discriminator.
    1D convolutions on waveform at a single scale.
    """
    
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, stride=1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, stride=2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, stride=2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, stride=4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, stride=4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, stride=1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, stride=1, padding=2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, stride=1, padding=1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Waveform [B, 1, T]
            
        Returns:
            output: Discrimination score [B, 1, T']
            features: List of intermediate features
        """
        features = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        
        return x.flatten(1, -1), features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD).
    Operates on original, 2x downsampled, and 4x downsampled waveforms.
    """
    
    def __init__(self):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),  # Original scale
            ScaleDiscriminator(),  # 2x downsampled
            ScaleDiscriminator(),  # 4x downsampled
        ])
        
        # Average pooling for downsampling
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, stride=2, padding=2),
            nn.AvgPool1d(4, stride=2, padding=2),
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: Waveform [B, T] or [B, 1, T]
            
        Returns:
            outputs: List of discrimination scores
            features: List of feature lists
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        outputs = []
        all_features = []
        
        for pool, disc in zip(self.pools, self.discriminators):
            x_pooled = pool(x)
            out, feats = disc(x_pooled)
            outputs.append(out)
            all_features.append(feats)
        
        return outputs, all_features


# ============================================================
# Combined Discriminator
# ============================================================

class CombinedDiscriminator(nn.Module):
    """
    Combines MPD and MSD for complete discrimination.
    """
    
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Returns combined outputs and features from both MPD and MSD.
        """
        mpd_outs, mpd_feats = self.mpd(x)
        msd_outs, msd_feats = self.msd(x)
        
        return mpd_outs + msd_outs, mpd_feats + msd_feats


# ============================================================
# Loss Functions
# ============================================================

def discriminator_loss(
    real_outputs: List[torch.Tensor],
    fake_outputs: List[torch.Tensor],
) -> torch.Tensor:
    """
    Hinge loss for discriminator.
    D wants: real -> 1, fake -> -1
    """
    loss = torch.tensor(0.0, device=real_outputs[0].device)
    for real, fake in zip(real_outputs, fake_outputs):
        # Clamp outputs to prevent extreme values
        real_clamped = real.clamp(-100, 100)
        fake_clamped = fake.clamp(-100, 100)
        loss = loss + torch.mean(F.relu(1 - real_clamped)) + torch.mean(F.relu(1 + fake_clamped))
    return loss


def generator_loss(fake_outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Hinge loss for generator.
    G wants: fake -> 1
    """
    loss = torch.tensor(0.0, device=fake_outputs[0].device)
    for fake in fake_outputs:
        # Clamp outputs to prevent extreme values
        fake_clamped = fake.clamp(-100, 100)
        loss = loss + torch.mean(F.relu(1 - fake_clamped))
    return loss


def feature_matching_loss(
    real_features: List[List[torch.Tensor]],
    fake_features: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    L1 feature matching loss across all discriminator layers.
    
    This is CRITICAL for stable GAN training - it provides dense gradients
    to the generator instead of just a scalar adversarial signal.
    """
    loss = 0
    for real_feats, fake_feats in zip(real_features, fake_features):
        for real_f, fake_f in zip(real_feats, fake_feats):
            loss += F.l1_loss(fake_f, real_f.detach())
    return loss


# ============================================================
# Convenience Builder
# ============================================================

def build_discriminator() -> CombinedDiscriminator:
    """Build the combined MPD + MSD discriminator."""
    return CombinedDiscriminator()