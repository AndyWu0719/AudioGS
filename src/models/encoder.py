"""
GaborVAEEncoder: Variational Autoencoder for Gabor Atom Prediction

Architecture (VITS-inspired):
- Posterior Encoder: Mel → Backbone → Downsample → μ, log_σ → z
- Projection Decoder: z → Upsample → Grid Head → Gabor Params

Key features:
- Temporal downsampling (stride=2) for efficient latent space
- Reparameterization trick for VAE sampling
- cos/sin phase prediction for stability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple, Optional, Dict


class ResidualBlock1D(nn.Module):
    """1D Residual block with pre-activation."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(x))
        x = self.conv1(x)
        x = F.gelu(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual


class ConformerBlock(nn.Module):
    """Simplified Conformer block for audio encoding."""
    
    def __init__(self, channels: int, num_heads: int = 4, kernel_size: int = 31):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.ff1 = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 4, channels),
            nn.Dropout(0.1),
        )
        
        self.norm2 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, num_heads, 
                                                dropout=0.1, batch_first=True)
        
        self.norm3 = nn.LayerNorm(channels)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
            nn.Dropout(0.1),
        )
        
        self.norm4 = nn.LayerNorm(channels)
        self.ff2 = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 4, channels),
            nn.Dropout(0.1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x + 0.5 * self.ff1(self.norm1(x))
        _x = self.norm2(x)
        _x, _ = self.self_attn(_x, _x, _x)
        x = x + _x
        _x = self.norm3(x).transpose(1, 2)
        _x = self.conv(_x).transpose(1, 2)
        x = x + _x
        x = x + 0.5 * self.ff2(self.norm4(x))
        return x


class GaborVAEEncoder(nn.Module):
    """
    Variational Autoencoder for Gabor Atom Prediction.
    
    Architecture:
    - Posterior Encoder: Mel → Backbone → Downsample → μ, log_σ
    - Decoder: z → Upsample → Grid Head → Gabor Params
    
    Latent dimension: [B, hidden_channels, T_latent] where T_latent = T_mel / downsample_ratio
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        input_mel_bins: int = 80,
        hidden_channels: int = 256,
        grid_freq_bins: int = 128,
        atoms_per_cell: int = 2,
        time_downsample_factor: int = 240,  # Mel hop length
        latent_downsample_ratio: int = 2,   # Latent temporal downsampling
        num_layers: int = 6,
        use_conformer: bool = True,
        sigma_min: float = 0.002,
        sigma_max: float = 0.05,
        existence_threshold: float = 0.5,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.input_mel_bins = input_mel_bins
        self.hidden_channels = hidden_channels
        self.grid_freq_bins = grid_freq_bins
        self.atoms_per_cell = atoms_per_cell
        self.time_downsample_factor = time_downsample_factor
        self.latent_downsample_ratio = latent_downsample_ratio
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.existence_threshold = existence_threshold
        self.use_conformer = use_conformer
        
        # Derived constants
        self.time_hop_sec = time_downsample_factor / sample_rate
        self.nyquist = sample_rate / 2
        self.freq_bin_bandwidth = self.nyquist / grid_freq_bins
        self.num_params = 8  # existence, amp, cos_phi, sin_phi, d_tau, d_omega, sigma, gamma
        self.gamma_scale = 1000.0
        
        # ========== Mel Transform ==========
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=time_downsample_factor,
            n_mels=input_mel_bins,
            f_min=0,
            f_max=sample_rate // 2,
        )
        
        # ========== Encoder (Posterior) ==========
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_mel_bins, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        
        if use_conformer:
            self.backbone = nn.ModuleList([
                ConformerBlock(hidden_channels, num_heads=4) 
                for _ in range(num_layers)
            ])
        else:
            self.backbone = nn.ModuleList([
                ResidualBlock1D(hidden_channels, kernel_size=3, dilation=2**i)
                for i in range(num_layers)
            ])
        
        # Temporal Downsampling for latent space
        self.downsample = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=4, 
                      stride=latent_downsample_ratio, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        
        # VAE projections (mu, log_sigma)
        self.mu_proj = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.logs_proj = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        
        # ========== Decoder (Projection) ==========
        # Temporal Upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4,
                               stride=latent_downsample_ratio, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        
        # Grid projection head
        self.grid_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, grid_freq_bins * atoms_per_cell * self.num_params),
        )
        
        # Initialize biases
        self._init_custom_bias()
        self._init_grid_centers()
    
    def _init_custom_bias(self):
        """Initialize biases for stable training."""
        final_layer = self.grid_head[-1]
        
        if final_layer.bias is not None:
            with torch.no_grad():
                bias = final_layer.bias.view(
                    self.grid_freq_bins, self.atoms_per_cell, self.num_params
                )
                # Existence: +2.0 (~88% active)
                bias[:, :, 0].fill_(2.0)
                # Amplitude: -12.0 (very low initial energy)
                bias[:, :, 1].fill_(-12.0)
                # cos/sin phase: random unit vectors
                rand_phase = torch.rand_like(bias[:, :, 2]) * 2 * math.pi - math.pi
                bias[:, :, 2] = torch.cos(rand_phase)
                bias[:, :, 3] = torch.sin(rand_phase)
                
                final_layer.bias.data = bias.view(-1)
    
    def _init_grid_centers(self):
        """Pre-compute fixed grid centers for tau and omega."""
        freq_centers = torch.linspace(
            self.freq_bin_bandwidth / 2,
            self.nyquist - self.freq_bin_bandwidth / 2,
            self.grid_freq_bins,
        )
        self.register_buffer('freq_centers', freq_centers)
    
    def _compute_time_centers(self, num_frames: int, device) -> torch.Tensor:
        """Compute time centers for given number of frames."""
        return torch.arange(num_frames, device=device).float() * self.time_hop_sec + self.time_hop_sec / 2
    
    def reparameterize(self, mu: torch.Tensor, logs: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        Args:
            mu: Mean [B, C, T]
            logs: Log standard deviation [B, C, T]
        Returns:
            z: Sampled latent [B, C, T]
        """
        if self.training:
            std = torch.exp(logs)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu  # Deterministic at inference
    
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Encode audio to latent distribution.
        
        Returns:
            z: Sampled latent [B, C, T_latent]
            mu: Mean [B, C, T_latent]
            logs: Log std [B, C, T_latent]
            num_mel_frames: Original mel frame count (for upsampling alignment)
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        # Mel spectrogram
        mel = self.mel_transform(audio)  # [B, n_mels, T]
        mel = torch.log(mel + 1e-6)
        num_mel_frames = mel.shape[-1]
        
        # Backbone
        x = self.input_proj(mel)  # [B, C, T]
        
        if self.use_conformer:
            x = x.transpose(1, 2)  # [B, T, C]
            for block in self.backbone:
                x = block(x)
            x = x.transpose(1, 2)  # [B, C, T]
        else:
            for block in self.backbone:
                x = block(x)
        
        # Downsample
        x = self.downsample(x)  # [B, C, T_latent]
        
        # VAE projections with clamping for stability
        mu = self.mu_proj(x)
        logs = self.logs_proj(x)
        
        # Clamp logs to prevent exp explosion (critical for stability)
        logs = torch.clamp(logs, min=-10.0, max=2.0)
        
        # Sample
        z = self.reparameterize(mu, logs)
        
        return z, mu, logs, num_mel_frames
    
    def decode(self, z: torch.Tensor, target_frames: int) -> Dict[str, torch.Tensor]:
        """
        Decode latent to Gabor atom parameters.
        
        Args:
            z: Latent [B, C, T_latent]
            target_frames: Number of output mel frames (T_mel)
        
        Returns:
            Dict with all atom parameters
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Upsample
        x = self.upsample(z)  # [B, C, T_upsampled]
        
        # Align to target frames (handle padding mismatches)
        if x.shape[-1] != target_frames:
            x = F.interpolate(x, size=target_frames, mode='linear', align_corners=False)
        
        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        
        # Grid projection
        grid_flat = self.grid_head(x)  # [B, T, F*A*P]
        
        # Reshape to grid
        grid = grid_flat.view(
            batch_size, target_frames,
            self.grid_freq_bins, self.atoms_per_cell, self.num_params
        )
        
        # Parse parameters
        existence_logit = grid[..., 0]
        amplitude_raw = grid[..., 1]
        cos_phi_raw = grid[..., 2]
        sin_phi_raw = grid[..., 3]
        delta_tau_raw = grid[..., 4]
        delta_omega_raw = grid[..., 5]
        sigma_raw = grid[..., 6]
        gamma_raw = grid[..., 7]
        
        # Activations with clamping for numerical stability
        existence_prob = torch.sigmoid(existence_logit)
        
        # Clamp amplitude_raw before softplus to prevent overflow
        amplitude_raw = torch.clamp(amplitude_raw, min=-20.0, max=10.0)
        amplitude = F.softplus(amplitude_raw)
        amplitude = torch.clamp(amplitude, max=100.0)  # Hard limit on amplitude
        
        # Phase with stability
        phase = torch.atan2(sin_phi_raw, cos_phi_raw + 1e-7)
        
        delta_tau = torch.tanh(delta_tau_raw) * (self.time_hop_sec / 2)
        delta_omega = torch.tanh(delta_omega_raw) * (self.freq_bin_bandwidth / 2)
        sigma = torch.sigmoid(sigma_raw) * (self.sigma_max - self.sigma_min) + self.sigma_min
        gamma = torch.tanh(gamma_raw) * self.gamma_scale
        
        # Grid-to-list conversion
        time_centers = self._compute_time_centers(target_frames, device)
        
        tau_grid = time_centers.view(1, -1, 1, 1).expand(batch_size, -1, self.grid_freq_bins, self.atoms_per_cell)
        omega_grid = self.freq_centers.view(1, 1, -1, 1).expand(batch_size, target_frames, -1, self.atoms_per_cell)
        
        tau_final = tau_grid + delta_tau
        omega_final = omega_grid + delta_omega
        
        # Mask by existence
        existence_mask = existence_prob > self.existence_threshold
        amplitude_masked = amplitude * existence_prob
        
        # Flatten
        amplitude_flat = amplitude_masked.reshape(batch_size, -1)
        tau_flat = tau_final.reshape(batch_size, -1)
        omega_flat = omega_final.reshape(batch_size, -1)
        sigma_flat = sigma.reshape(batch_size, -1)
        phi_flat = phase.reshape(batch_size, -1)
        gamma_flat = gamma.reshape(batch_size, -1)
        existence_prob_flat = existence_prob.reshape(batch_size, -1)
        
        return {
            'amplitude': amplitude_flat,
            'tau': tau_flat,
            'omega': omega_flat,
            'sigma': sigma_flat,
            'phi': phi_flat,
            'gamma': gamma_flat,
            'existence_prob': existence_prob_flat,
            'existence_mask': existence_mask.reshape(batch_size, -1),
        }
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full VAE forward pass.
        
        Returns:
            Dict containing:
            - All Gabor atom parameters
            - 'z': Sampled latent [B, C, T_latent]
            - 'mu': Latent mean [B, C, T_latent]
            - 'logs': Latent log_std [B, C, T_latent]
        """
        z, mu, logs, num_mel_frames = self.encode(audio)
        
        atom_params = self.decode(z, num_mel_frames)
        
        # Add VAE outputs
        atom_params['z'] = z
        atom_params['mu'] = mu
        atom_params['logs'] = logs
        
        return atom_params
    
    @property
    def atoms_per_second(self) -> int:
        """Theoretical atom capacity per second."""
        frames_per_sec = self.sample_rate / self.time_downsample_factor
        return int(frames_per_sec * self.grid_freq_bins * self.atoms_per_cell)
    
    @property
    def latent_frames_per_second(self) -> float:
        """Latent sequence length per second of audio."""
        return self.sample_rate / self.time_downsample_factor / self.latent_downsample_ratio


# ============================================================
# Legacy Alias (for backward compatibility)
# ============================================================
GaborGridEncoder = GaborVAEEncoder


# ============================================================
# Factory Functions
# ============================================================

def build_encoder(config: dict) -> GaborVAEEncoder:
    """Build VAE encoder from config dict."""
    enc_config = config.get('encoder_model', {})
    
    return GaborVAEEncoder(
        sample_rate=config.get('data', {}).get('sample_rate', 24000),
        input_mel_bins=enc_config.get('input_mel_bins', 80),
        hidden_channels=enc_config.get('hidden_channels', 256),
        grid_freq_bins=enc_config.get('grid_freq_bins', 128),
        atoms_per_cell=enc_config.get('atoms_per_cell', 2),
        time_downsample_factor=enc_config.get('time_downsample_factor', 240),
        latent_downsample_ratio=enc_config.get('latent_downsample_ratio', 2),
        num_layers=enc_config.get('num_layers', 6),
        use_conformer=enc_config.get('use_conformer', True),
    )
