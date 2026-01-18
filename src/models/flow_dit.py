"""
Flow DiT: Diffusion Transformer for Flow Matching Audio Generation.

Implements a DiT-style Transformer for generating Gabor atom parameters
from text and speaker conditioning using Flow Matching.

Reference: Peebles & Xie "Scalable Diffusion Models with Transformers" (2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] Time values in [0, 1]
        Returns:
            emb: [B, dim] Sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm Zero (AdaLN-Zero) block.
    
    Modulates layer norm with learned scale/shift from conditioning,
    with zero-initialization for residual connections.
    """
    
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # 6 outputs: shift, scale, gate for attn and ffn
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim),
        )
        # Zero init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            c: [B, cond_dim] Conditioning
        Returns:
            6 modulation vectors: shift1, scale1, gate1, shift2, scale2, gate2
        """
        return self.adaLN_modulation(c).chunk(6, dim=-1)


class DiTBlock(nn.Module):
    """
    DiT Transformer Block with AdaLN-Zero modulation.
    
    Architecture:
        x -> AdaLN -> Self-Attn -> Gate -> + -> AdaLN -> Cross-Attn (opt) -> Gate -> + -> AdaLN -> FFN -> Gate -> +
    """
    
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
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention (for text conditioning)
        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        # FFN
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # AdaLN modulation
        # 6 outputs for self-attn + FFN (or 9 if cross-attn)
        num_modulation = 9 if use_cross_attention else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_modulation * hidden_dim),
        )
        # Zero init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim] Input tokens
            c: [B, cond_dim] Conditioning (time + speaker)
            text_emb: [B, L, hidden_dim] Text embeddings for cross-attention
            text_mask: [B, L] Text attention mask
            attn_mask: [B, N] Atom attention mask
        """
        # Get modulation parameters
        modulation = self.adaLN_modulation(c)
        if self.use_cross_attention:
            (shift1, scale1, gate1, 
             shift_cross, scale_cross, gate_cross,
             shift2, scale2, gate2) = modulation.chunk(9, dim=-1)
        else:
            shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        x_norm = modulate(self.norm1(x), shift1, scale1)
        
        # Create attention mask if needed
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = ~attn_mask  # True = ignore
        
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + gate1.unsqueeze(1) * attn_out
        
        # Cross-attention (if text conditioning)
        if self.use_cross_attention and text_emb is not None:
            x_norm = modulate(self.norm_cross(x), shift_cross, scale_cross)
            
            text_key_padding_mask = None
            if text_mask is not None:
                text_key_padding_mask = ~text_mask  # True = ignore
            
            cross_out, _ = self.cross_attn(
                x_norm, text_emb, text_emb, 
                key_padding_mask=text_key_padding_mask
            )
            x = x + gate_cross.unsqueeze(1) * cross_out
        
        # FFN with AdaLN
        x_norm = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.ffn(x_norm)
        
        return x


class FlowDiT(nn.Module):
    """
    Flow Matching DiT for generating Gabor atom parameters.
    
    ANCHOR-BASED ARCHITECTURE (solves permutation variance in set generation):
    -------------------------------------------------------------------------
    Instead of processing all atoms directly, we use temporal anchors:
    
    1. Input atoms are pooled into `num_anchors` anchor tokens via Conv1d
    2. Transformer operates on anchors (not full atom count)
    3. Splitting head expands each anchor back to `split_factor` atoms
    
    Example configuration (for 3s audio at 24kHz with 20480 atoms):
    - num_anchors = 2560 (temporal anchor grid)
    - split_factor = 8 (atoms per anchor)
    - total_atoms = 2560 × 8 = 20480
    - Anchor spacing: 3.0s / 2560 ≈ 1.17ms between anchors
    
    BENEFITS:
    - Reduces attention complexity from O(N²) to O((N/K)²)
    - Atoms are generated RELATIVE to anchors (resolves permutation ambiguity)
    - Hierarchical structure: anchors for coarse timing, splits for fine detail
    
    This is inspired by DiffGS (Hierarchical 3D Gaussian Diffusion).
    """
    
    def __init__(
        self,
        atom_dim: int = 6,              # [tau, omega, sigma, amp, phi, gamma]
        hidden_dim: int = 512,
        cond_dim: int = 512,            # Time + speaker embedding dim
        text_dim: int = 512,            # Text encoder hidden dim
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_anchors: int = 2560,        # Transformer sequence length
        split_factor: int = 8,          # Each anchor splits into K atoms
        num_speakers: int = 500,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.num_anchors = num_anchors
        self.split_factor = split_factor
        self.total_atoms = num_anchors * split_factor  # e.g., 20480
        
        # Input projection: atoms -> hidden
        # We receive total_atoms, need to pool to num_anchors
        self.input_pool = nn.Conv1d(
            atom_dim, hidden_dim, 
            kernel_size=split_factor, 
            stride=split_factor
        )  # Pools K atoms into 1 anchor
        
        # Positional embedding for anchors (not full atoms)
        self.pos_embed = nn.Parameter(torch.randn(1, num_anchors, hidden_dim) * 0.02)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Speaker embedding
        self.speaker_embed = nn.Embedding(num_speakers, cond_dim // 2)
        
        # Text projection (project text encoder output to hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Conditioning fusion
        self.cond_proj = nn.Linear(cond_dim + cond_dim // 2, cond_dim)
        
        # DiT blocks - operate on num_anchors sequence length
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_cross_attention=use_cross_attention,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        
        # Splitting head: each anchor -> split_factor atoms
        # Output: hidden_dim -> split_factor * atom_dim
        self.output_proj = nn.Linear(hidden_dim, split_factor * atom_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        atom_mask: Optional[torch.Tensor] = None,  # Ignored in anchor-based
    ) -> torch.Tensor:
        """
        Predict velocity v_t for flow matching.
        
        Args:
            x: [B, total_atoms, 6] Noisy atom parameters (e.g., 20480 atoms)
            t: [B] Time values in [0, 1]
            speaker_ids: [B] Speaker IDs
            text_embeddings: [B, L, text_dim] Text encoder outputs
            text_mask: [B, L] Text attention mask
            atom_mask: Ignored (fixed size, no masking needed)
            
        Returns:
            v: [B, total_atoms, 6] Predicted velocity
        """
        B, N, D = x.shape
        assert N == self.total_atoms, f"Expected {self.total_atoms} atoms, got {N}"
        
        # Pool atoms to anchors
        # x: [B, total_atoms, atom_dim] -> [B, num_anchors, hidden_dim]
        x_t = x.permute(0, 2, 1)  # [B, 6, total_atoms]
        x = self.input_pool(x_t)  # [B, hidden_dim, num_anchors]
        x = x.permute(0, 2, 1)    # [B, num_anchors, hidden_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Time conditioning
        t_emb = self.time_embed(t)  # [B, cond_dim]
        
        # Speaker conditioning
        if speaker_ids is not None:
            spk_emb = self.speaker_embed(speaker_ids)  # [B, cond_dim//2]
        else:
            spk_emb = torch.zeros(B, self.speaker_embed.embedding_dim, device=x.device)
        
        # Fuse conditioning
        c = self.cond_proj(torch.cat([t_emb, spk_emb], dim=-1))  # [B, cond_dim]
        
        # Project text embeddings
        if text_embeddings is not None:
            text_emb = self.text_proj(text_embeddings)  # [B, L, hidden_dim]
        else:
            text_emb = None
        
        # Transformer blocks (no atom_mask needed for fixed-size anchors)
        for block in self.blocks:
            x = block(x, c, text_emb, text_mask, attn_mask=None)
        
        # Final layer
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        
        # Splitting head: each anchor -> split_factor atoms
        # x: [B, num_anchors, hidden_dim] -> [B, num_anchors, split_factor * atom_dim]
        x = self.output_proj(x)  # [B, 2560, 48]
        
        # Reshape to full atom count
        # [B, num_anchors, split_factor * atom_dim] -> [B, total_atoms, atom_dim]
        v = x.view(B, self.total_atoms, self.atom_dim)  # [B, 20480, 6]
        
        return v


class FlowDiTSmall(FlowDiT):
    """Small DiT for testing."""
    def __init__(self, **kwargs):
        # Remove conflicting keys that are set by this class
        kwargs.pop('hidden_dim', None)
        kwargs.pop('cond_dim', None)
        kwargs.pop('num_layers', None)
        kwargs.pop('num_heads', None)
        text_dim = kwargs.pop('text_dim', 256)
        super().__init__(
            hidden_dim=256,
            cond_dim=256,
            text_dim=text_dim,
            num_layers=4,
            num_heads=4,
            **kwargs
        )


class FlowDiTBase(FlowDiT):
    """Base DiT configuration."""
    def __init__(self, **kwargs):
        # Remove conflicting keys that are set by this class
        kwargs.pop('hidden_dim', None)
        kwargs.pop('cond_dim', None)
        kwargs.pop('num_layers', None)
        kwargs.pop('num_heads', None)
        text_dim = kwargs.pop('text_dim', 512)
        super().__init__(
            hidden_dim=512,
            cond_dim=512,
            text_dim=text_dim,
            num_layers=8,
            num_heads=8,
            **kwargs
        )


class FlowDiTLarge(FlowDiT):
    """Large DiT configuration."""
    def __init__(self, **kwargs):
        # Remove conflicting keys that are set by this class
        kwargs.pop('hidden_dim', None)
        kwargs.pop('cond_dim', None)
        kwargs.pop('num_layers', None)
        kwargs.pop('num_heads', None)
        text_dim = kwargs.pop('text_dim', 512)
        super().__init__(
            hidden_dim=768,
            cond_dim=768,
            text_dim=text_dim,
            num_layers=12,
            num_heads=12,
            **kwargs
        )


def get_flow_model(size: str = 'base', **kwargs) -> FlowDiT:
    """Factory function for flow models."""
    models = {
        'small': FlowDiTSmall,
        'base': FlowDiTBase,
        'large': FlowDiTLarge,
    }
    return models[size](**kwargs)
