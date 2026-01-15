"""
GaborGridEncoder: Amortized Inference for Gabor Atom Prediction

Instead of optimizing atom parameters from scratch for every audio file (minutes),
this encoder predicts Gabor atom parameters directly from audio (milliseconds).

Architecture:
- Input: Log Mel-Spectrogram (80 mel bins)
- Backbone: Lightweight 1D ResNet/Conformer
- Output: Time-Frequency Grid of atom parameters [B, T, F, A, Params]

The existing GaborRenderer acts as a fixed, differentiable Decoder.
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
        # FFN 1 (half-step)
        x = x + 0.5 * self.ff1(self.norm1(x))
        
        # Self-attention
        _x = self.norm2(x)
        _x, _ = self.self_attn(_x, _x, _x)
        x = x + _x
        
        # Convolution
        _x = self.norm3(x).transpose(1, 2)  # [B, C, T]
        _x = self.conv(_x).transpose(1, 2)  # [B, T, C]
        x = x + _x
        
        # FFN 2 (half-step)
        x = x + 0.5 * self.ff2(self.norm4(x))
        
        return x


class GaborGridEncoder(nn.Module):
    """
    Encoder that predicts Gabor atom parameters as a Time-Frequency Grid.
    
    Grid Structure: [B, T_frames, F_bins, Atoms_per_cell, Params]
    - T_frames: ~100 Hz (10ms hop)
    - F_bins: 128 (covering 0-Nyquist)
    - Atoms_per_cell: 2
    - Params: 7 (existence, amplitude, phase, delta_tau, delta_omega, sigma, gamma)
    
    Total capacity: 100 * 128 * 2 = 25,600 atoms/sec
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        input_mel_bins: int = 80,
        hidden_channels: int = 256,
        grid_freq_bins: int = 128,
        atoms_per_cell: int = 2,
        time_downsample_factor: int = 240,  # 24000/240 = 100 Hz
        num_layers: int = 6,
        use_conformer: bool = True,
        use_checkpointing: bool = False,  # Gradient checkpointing for memory savings
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
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.existence_threshold = existence_threshold
        self.use_checkpointing = use_checkpointing
        
        # Derived constants
        self.time_hop_sec = time_downsample_factor / sample_rate  # 10ms
        self.nyquist = sample_rate / 2
        self.freq_bin_bandwidth = self.nyquist / grid_freq_bins
        # 7 params: amp, cos_phi, sin_phi, d_tau, d_omega, sigma, gamma (no existence)
        self.num_params = 7
        self.gamma_scale = 100.0  # Chirp rate scaling (Hz/s) - reduced for stability
        
        # Mel-Spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=time_downsample_factor,
            n_mels=input_mel_bins,
            f_min=0,
            f_max=sample_rate // 2,
        )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_mel_bins, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        
        # Backbone
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
        
        self.use_conformer = use_conformer
        
        # Grid projection head
        # Output: [B, T, grid_freq_bins * atoms_per_cell * num_params]
        self.grid_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, grid_freq_bins * atoms_per_cell * self.num_params),
        )
        
        # Initialize bias (Existence +2.0, Phase uniform)
        self._init_custom_bias()
        
        # Register buffers for grid centers (computed once)
        self._init_grid_centers()
    
    def _init_custom_bias(self):
        """
        Initialize custom biases for the grid head.
        
        Pure Autoencoder: No existence. Let amplitude learn to be zero for silence.
        
        1. Amplitude (Index 0): -2.0 (Softplus(-2) ≈ 0.13, reasonable start)
        2. cos_phi (Index 1) and sin_phi (Index 2): Random unit vector
        3. d_tau (Index 3): 0 (centered)
        4. d_omega (Index 4): 0 (centered)
        5. Sigma (Index 5): log(0.01) ≈ -4.6 (10ms wide atoms to catch gradients)
        6. Gamma (Index 6): 0 (no chirp initially)
        """
        import math
        final_layer = self.grid_head[-1]
        
        if final_layer.bias is not None:
            with torch.no_grad():
                # Reshape bias to [F, A, P]
                # P=7: [amp, cos_phi, sin_phi, d_tau, d_omega, sigma, gamma]
                bias = final_layer.bias.view(
                    self.grid_freq_bins, self.atoms_per_cell, self.num_params
                )
                
                # 1. Amplitude bias -> -2.0 (Softplus(-2) ≈ 0.13)
                bias[:, :, 0].fill_(-2.0)
                
                # 2. cos_phi/sin_phi: Random unit vectors
                rand_phase = torch.rand_like(bias[:, :, 1]) * 2 * math.pi - math.pi
                bias[:, :, 1] = torch.cos(rand_phase)  # cos_phi
                bias[:, :, 2] = torch.sin(rand_phase)  # sin_phi
                
                # 3-4. d_tau, d_omega: centered at 0
                bias[:, :, 3].fill_(0.0)  # d_tau
                bias[:, :, 4].fill_(0.0)  # d_omega
                
                # 5. Sigma: log(0.01) ≈ -4.6 for ~10ms atoms
                bias[:, :, 5].fill_(math.log(0.01))
                
                # 6. Gamma: 0 (no chirp)
                bias[:, :, 6].fill_(0.0)
                
                # Flatten back
                final_layer.bias.data = bias.view(-1)
        
    def _init_grid_centers(self):
        """Pre-compute fixed grid centers for tau and omega."""
        # Frequency centers for each bin (will be expanded to match time frames)
        freq_centers = torch.linspace(
            self.freq_bin_bandwidth / 2,  # Center of first bin
            self.nyquist - self.freq_bin_bandwidth / 2,  # Center of last bin
            self.grid_freq_bins
        )
        self.register_buffer('freq_centers', freq_centers)
        
    def _compute_time_centers(self, num_frames: int, device: torch.device) -> torch.Tensor:
        """Compute time centers for given number of frames."""
        return torch.arange(num_frames, device=device) * self.time_hop_sec + self.time_hop_sec / 2
        
    def forward(
        self, 
        audio: torch.Tensor,  # [B, num_samples] or [B, 1, num_samples]
        return_grid: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Audio -> Mel -> Grid -> Flat Atom Params
        
        Returns:
            Dict with keys:
            - 'amplitude': [B, N_atoms]
            - 'tau': [B, N_atoms]
            - 'omega': [B, N_atoms]
            - 'sigma': [B, N_atoms]
            - 'phi': [B, N_atoms]
            - 'gamma': [B, N_atoms] (chirp, fixed to 0 for now)
            - 'existence_mask': [B, N_atoms] (bool)
            - 'existence_prob': [B, N_atoms]
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # [B, num_samples]
        
        batch_size = audio.shape[0]
        device = audio.device
        
        # ============================
        # 1. Mel-Spectrogram extraction
        # ============================
        mel = self.mel_transform(audio)  # [B, n_mels, T_frames]
        mel = torch.log(mel + 1e-6)  # Log compression
        
        num_frames = mel.shape[-1]
        
        # ============================
        # 2. Backbone encoding
        # ============================
        x = self.input_proj(mel)  # [B, hidden, T]
        
        if self.use_conformer:
            x = x.transpose(1, 2)  # [B, T, hidden]
            for block in self.backbone:
                if self.use_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            # x: [B, T, hidden]
        else:
            for block in self.backbone:
                if self.use_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            x = x.transpose(1, 2)  # [B, T, hidden]
        
        # ============================
        # 3. Grid projection
        # ============================
        grid_flat = self.grid_head(x)  # [B, T, F*A*P]
        
        # Reshape to grid: [B, T, F, A, P]
        grid = grid_flat.view(
            batch_size, num_frames, 
            self.grid_freq_bins, self.atoms_per_cell, self.num_params
        )
        
        # ============================
        # 4. Parse parameters (NO EXISTENCE - pure autoencoder)
        # ============================
        # Indices: 0=amplitude, 1=cos_phi, 2=sin_phi, 3=d_tau, 4=d_omega, 5=sigma, 6=gamma
        amplitude_raw = grid[..., 0]
        cos_phi_raw = grid[..., 1]
        sin_phi_raw = grid[..., 2]
        delta_tau_raw = grid[..., 3]
        delta_omega_raw = grid[..., 4]
        sigma_raw = grid[..., 5]
        gamma_raw = grid[..., 6]
        
        # Apply activations
        # Amplitude: softplus with clamp for numerical stability
        amplitude = F.softplus(amplitude_raw).clamp(max=10.0)  # Prevent extreme values
        
        # Phase recovery from cos/sin with numerical stability
        # Note: We output phi (scalar) for renderer, but Flow should predict cos/sin
        phase = torch.atan2(sin_phi_raw, cos_phi_raw + 1e-7)
        
        # Local offsets: Allow movement within ±1 cell
        delta_tau = torch.tanh(delta_tau_raw) * (self.time_hop_sec * 1.0)
        delta_omega = torch.tanh(delta_omega_raw) * (self.freq_bin_bandwidth * 1.0)
        
        # Sigma: Use softplus for bounded, stable output (NOT exp which can explode)
        # softplus(x - 2) + sigma_min gives range [sigma_min, ~inf) but bounded growth
        sigma = F.softplus(sigma_raw - 2.0) * 0.02 + self.sigma_min
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)
        
        # Gamma (chirp): Reduced scale for stability (±100 Hz/s)
        gamma = torch.tanh(gamma_raw) * self.gamma_scale
        
        # ============================
        # 5. Grid-to-List conversion (differentiable)
        # ============================
        time_centers = self._compute_time_centers(num_frames, device)  # [T]
        freq_centers = self.freq_centers  # [F]
        
        # Expand to full grid shape [B, T, F, A]
        tau_grid = time_centers.view(1, -1, 1, 1).expand(batch_size, -1, self.grid_freq_bins, self.atoms_per_cell)
        omega_grid = freq_centers.view(1, 1, -1, 1).expand(batch_size, num_frames, -1, self.atoms_per_cell)
        
        # Add offsets (differentiable)
        tau_final = tau_grid + delta_tau
        omega_final = omega_grid + delta_omega
        
        # ============================
        # 6. Flatten to [B, N_atoms]
        # ============================
        n_atoms = num_frames * self.grid_freq_bins * self.atoms_per_cell
        
        amplitude_flat = amplitude.view(batch_size, n_atoms)
        tau_flat = tau_final.view(batch_size, n_atoms)
        omega_flat = omega_final.view(batch_size, n_atoms)
        sigma_flat = sigma.view(batch_size, n_atoms)
        phi_flat = phase.view(batch_size, n_atoms)
        gamma_flat = gamma.view(batch_size, n_atoms)
        
        # Normalize cos/sin to unit vectors (for Flow Matching - continuous manifold)
        cos_phi_norm = cos_phi_raw / (torch.sqrt(cos_phi_raw**2 + sin_phi_raw**2) + 1e-7)
        sin_phi_norm = sin_phi_raw / (torch.sqrt(cos_phi_raw**2 + sin_phi_raw**2) + 1e-7)
        cos_phi_flat = cos_phi_norm.view(batch_size, n_atoms)
        sin_phi_flat = sin_phi_norm.view(batch_size, n_atoms)
        
        # Output: 6 params for Renderer, 8 params for Flow (with cos/sin instead of phi)
        result = {
            # For Renderer (6 params)
            'amplitude': amplitude_flat,
            'tau': tau_flat,
            'omega': omega_flat,
            'sigma': sigma_flat,
            'phi': phi_flat,         # Scalar phase for renderer
            'gamma': gamma_flat,
            # For Flow Matching (use these instead of phi - continuous manifold)
            'cos_phi': cos_phi_flat,
            'sin_phi': sin_phi_flat,
            # Metadata
            'num_frames': num_frames,
            'atoms_per_frame': self.grid_freq_bins * self.atoms_per_cell,
        }
        
        if return_grid:
            result['grid'] = grid
            
        return result
    
    def get_flat_params(
        self, 
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method returning flat params in GaborRenderer order.
        
        Returns:
            amplitude, tau, omega, sigma, phi, gamma - all [B, N] or [N] if B=1
        """
        result = self.forward(audio)
        return (
            result['amplitude'],
            result['tau'],
            result['omega'],
            result['sigma'],
            result['phi'],
            result['gamma'],
        )
    
    @property
    def atoms_per_second(self) -> int:
        """Theoretical atom capacity per second of audio."""
        frames_per_sec = self.sample_rate / self.time_downsample_factor
        return int(frames_per_sec * self.grid_freq_bins * self.atoms_per_cell)
    
    def count_active_atoms(self, existence_prob: torch.Tensor) -> torch.Tensor:
        """Count atoms with existence probability above threshold."""
        return (existence_prob > self.existence_threshold).sum(dim=-1).float()


class GaborAutoEncoder(nn.Module):
    """
    Full autoencoder: Encoder + GaborRenderer (Decoder).
    
    This combines the GaborGridEncoder with the differentiable GaborRenderer
    for end-to-end training.
    """
    
    def __init__(
        self,
        encoder: GaborGridEncoder,
        renderer,  # GaborRenderer from cuda_gabor
    ):
        super().__init__()
        self.encoder = encoder
        self.renderer = renderer
        
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass: Audio -> Atoms -> Reconstructed Audio
        
        Args:
            audio: [B, num_samples] or [B, 1, num_samples]
            
        Returns:
            reconstructed: [B, num_samples]
            encoder_output: Dict with atom parameters
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        num_samples = audio.shape[-1]
        
        # Encode
        enc_out = self.encoder(audio)
        
        # Render (per-sample in batch for now)
        batch_size = audio.shape[0]
        reconstructed_list = []
        
        for b in range(batch_size):
            recon = self.renderer(
                enc_out['amplitude'][b],
                enc_out['tau'][b],
                enc_out['omega'][b],
                enc_out['sigma'][b],
                enc_out['phi'][b],
                enc_out['gamma'][b],
                num_samples,
            )
            reconstructed_list.append(recon)
        
        reconstructed = torch.stack(reconstructed_list, dim=0)
        
        return reconstructed, enc_out


# Factory function
def build_encoder(config: dict) -> GaborGridEncoder:
    """Build encoder from config dict."""
    enc_config = config.get('encoder_model', {})
    
    return GaborGridEncoder(
        sample_rate=config.get('data', {}).get('sample_rate', 24000),
        input_mel_bins=enc_config.get('input_mel_bins', 80),
        hidden_channels=enc_config.get('hidden_channels', 256),
        grid_freq_bins=enc_config.get('grid_freq_bins', 128),
        atoms_per_cell=enc_config.get('atoms_per_cell', 2),
        time_downsample_factor=enc_config.get('time_downsample_factor', 240),
        num_layers=enc_config.get('num_layers', 6),
        use_conformer=enc_config.get('use_conformer', True),
        use_checkpointing=enc_config.get('use_checkpointing', False),
    )