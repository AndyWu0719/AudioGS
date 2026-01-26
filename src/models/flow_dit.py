"""
Flow DiT for codec latent sequences (Stage04+).

This replaces the old continuous-atom/anchor-based generator. The model predicts the
velocity field v(x, t | text, speaker) for Flow Matching on latent sequences:

  x: [B, Tz, Dz]  (codec latents)

Tz is variable; padding is handled via `latent_mask`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for scalar t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """[1, length, dim] sinusoidal positional encoding."""
    if length <= 0:
        return torch.zeros(1, 0, dim, device=device)
    pos = torch.arange(length, device=device).float()
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=device).float() / max(1, half - 1))
    angles = pos[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(length, 1, device=device)], dim=-1)
    return emb.unsqueeze(0)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        num_mod = 9 if use_cross_attention else 6
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, num_mod * hidden_dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mod = self.adaLN(c)
        if self.use_cross_attention:
            (shift1, scale1, gate1, shift_cross, scale_cross, gate_cross, shift2, scale2, gate2) = mod.chunk(
                9, dim=-1
            )
        else:
            shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # Self-attn
        x_norm = modulate(self.norm1(x), shift1, scale1)
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = ~attn_mask  # True = ignore
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + gate1.unsqueeze(1) * attn_out

        # Cross-attn
        if self.use_cross_attention and text_emb is not None:
            x_norm = modulate(self.norm_cross(x), shift_cross, scale_cross)
            text_key_padding_mask = None
            if text_mask is not None:
                text_key_padding_mask = ~text_mask
            cross_out, _ = self.cross_attn(x_norm, text_emb, text_emb, key_padding_mask=text_key_padding_mask)
            x = x + gate_cross.unsqueeze(1) * cross_out

        # FFN
        x_norm = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.ffn(x_norm)
        return x


class FlowDiT(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        cond_dim: int = 512,
        text_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_speakers: int = 500,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.in_proj = nn.Linear(latent_dim, hidden_dim)

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.speaker_embed = nn.Embedding(num_speakers, cond_dim // 2)
        self.cond_proj = nn.Linear(cond_dim + cond_dim // 2, cond_dim)

        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_cross_attention=use_cross_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        latent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, Tz, Dz]
            t: [B]
            latent_mask: [B, Tz] (True = valid)
        Returns:
            v: [B, Tz, Dz]
        """
        B, Tz, Dz = x.shape
        x = self.in_proj(x)
        x = x + sinusoidal_positions(Tz, self.hidden_dim, device=x.device)

        t_emb = self.time_embed(t)  # [B, cond_dim]
        if speaker_ids is None:
            spk = torch.zeros(B, self.speaker_embed.embedding_dim, device=x.device)
        else:
            spk = self.speaker_embed(speaker_ids)
        c = self.cond_proj(torch.cat([t_emb, spk], dim=-1))

        text_emb = self.text_proj(text_embeddings) if text_embeddings is not None else None

        for block in self.blocks:
            x = block(x, c, text_emb=text_emb, text_mask=text_mask, attn_mask=latent_mask)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.out_proj(x)


def get_flow_model(size: str = "base", **kwargs) -> FlowDiT:
    presets = {
        "small": dict(hidden_dim=256, cond_dim=256, num_layers=4, num_heads=4),
        "base": dict(hidden_dim=512, cond_dim=512, num_layers=8, num_heads=8),
        "large": dict(hidden_dim=768, cond_dim=768, num_layers=12, num_heads=12),
    }
    if size not in presets:
        raise ValueError(f"Unknown size={size}, choose from {list(presets)}")
    merged = {**presets[size], **kwargs}
    return FlowDiT(**merged)

