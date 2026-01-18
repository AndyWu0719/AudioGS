"""
Multi-Resolution STFT Loss for Audio Reconstruction.

Uses multiple FFT sizes to capture both coarse and fine spectral details.
Combines spectral convergence loss and log-magnitude loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class STFTLoss(nn.Module):
    """Single-scale STFT loss."""
    
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        eps: float = 1e-8,
    ):
        """
        Initialize STFT loss.
        
        Args:
            fft_size: FFT size
            hop_size: Hop size between frames
            win_length: Window length
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.eps = eps
        
        # Register window as buffer
        self.register_buffer(
            "window",
            torch.hann_window(win_length)
        )
        
    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT magnitude.
        
        Args:
            x: Input waveform [B, T] or [T]
            
        Returns:
            STFT magnitude [B, F, T'] or [F, T']
        """
        # Ensure 2D input
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        
        # Move window to same device as input
        window = self.window.to(x.device)
        
        # Compute STFT
        stft_out = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        
        # Get magnitude
        mag = stft_out.abs()
        
        if squeeze_output:
            mag = mag.squeeze(0)
        
        return mag
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute STFT loss.
        
        Args:
            pred: Predicted waveform
            target: Target waveform
            
        Returns:
            Tuple of (spectral_convergence_loss, log_magnitude_loss)
        """
        # Compute STFT magnitudes
        pred_mag = self.stft(pred)
        target_mag = self.stft(target)
        
        # Spectral convergence loss (Frobenius norm)
        sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + self.eps
        )
        
        # Log magnitude loss
        log_pred = torch.log(pred_mag + self.eps)
        log_target = torch.log(target_mag + self.eps)
        log_mag_loss = F.l1_loss(log_pred, log_target)
        
        return sc_loss, log_mag_loss


class ComplexSTFTLoss(nn.Module):
    """
    Complex STFT Loss for phase-aware audio reconstruction.
    
    Unlike magnitude-only STFT loss, this loss penalizes both
    magnitude AND phase errors by treating STFT as complex tensors.
    
    Formula: L = ||STFT(pred) - STFT(target)||_2
    
    This is crucial for achieving PESQ > 3.0 as phase misalignment
    causes perceptual artifacts that magnitude loss cannot capture.
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512],
        hop_sizes: List[int] = [512, 256, 128],
        win_lengths: List[int] = [2048, 1024, 512],
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        # Register windows
        for i, win_len in enumerate(win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(win_len))
    
    def _get_window(self, idx: int, device: torch.device) -> torch.Tensor:
        return getattr(self, f"window_{idx}").to(device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale complex STFT loss.
        
        Args:
            pred: Predicted waveform [B, T] or [T]
            target: Target waveform [B, T] or [T]
            
        Returns:
            Complex STFT loss (scalar tensor)
        """
        # Ensure 2D
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        total_loss = 0.0
        
        for i, (fft_size, hop_size, win_len) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)
        ):
            window = self._get_window(i, pred.device)
            
            # Compute complex STFT
            pred_stft = torch.stft(
                pred, n_fft=fft_size, hop_length=hop_size, win_length=win_len,
                window=window, return_complex=True, center=True, pad_mode="reflect"
            )
            target_stft = torch.stft(
                target, n_fft=fft_size, hop_length=hop_size, win_length=win_len,
                window=window, return_complex=True, center=True, pad_mode="reflect"
            )
            
            # Complex L2 loss: ||pred - target||_2
            # This is equivalent to sqrt((real_diff)^2 + (imag_diff)^2)
            complex_diff = pred_stft - target_stft
            loss = complex_diff.abs().mean()
            
            total_loss = total_loss + loss
        
        return total_loss / len(self.fft_sizes)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss.
    
    Combines losses from multiple FFT sizes to capture both
    coarse (large FFT) and fine (small FFT) spectral details.
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512, 128],
        hop_sizes: List[int] = [512, 256, 128, 32],
        win_lengths: List[int] = [2048, 1024, 512, 128],
        spectral_weight: float = 1.0,
        log_mag_weight: float = 1.0,
        time_domain_weight: float = 0.1,
    ):
        """
        Initialize multi-resolution STFT loss.
        
        Args:
            fft_sizes: List of FFT sizes
            hop_sizes: List of hop sizes (must match fft_sizes length)
            win_lengths: List of window lengths (must match fft_sizes length)
            spectral_weight: Weight for spectral convergence loss
            log_mag_weight: Weight for log magnitude loss
            time_domain_weight: Weight for time-domain L1 loss
        """
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), \
            "fft_sizes, hop_sizes, and win_lengths must have same length"
        
        self.spectral_weight = spectral_weight
        self.log_mag_weight = log_mag_weight
        self.time_domain_weight = time_domain_weight
        
        # Create loss modules for each scale
        self.stft_losses = nn.ModuleList([
            STFTLoss(fft, hop, win)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-resolution STFT loss.
        
        Args:
            pred: Predicted waveform [B, T] or [T]
            target: Target waveform [B, T] or [T]
            
        Returns:
            Tuple of (total_loss, loss_dict with components)
        """
        sc_loss_total = 0.0
        log_mag_loss_total = 0.0
        
        # Accumulate losses from each scale
        for stft_loss in self.stft_losses:
            sc_loss, log_mag_loss = stft_loss(pred, target)
            sc_loss_total += sc_loss
            log_mag_loss_total += log_mag_loss
        
        # Average across scales
        num_scales = len(self.stft_losses)
        sc_loss_avg = sc_loss_total / num_scales
        log_mag_loss_avg = log_mag_loss_total / num_scales
        
        # Time-domain L1 loss
        time_loss = F.l1_loss(pred, target)
        
        # Combine losses
        total_loss = (
            self.spectral_weight * sc_loss_avg +
            self.log_mag_weight * log_mag_loss_avg +
            self.time_domain_weight * time_loss
        )
        
        # Return total loss and components for logging
        loss_dict = {
            "spectral_convergence": sc_loss_avg.item(),
            "log_magnitude": log_mag_loss_avg.item(),
            "time_domain": time_loss.item(),
            "total": total_loss.item(),
        }
        
        return total_loss, loss_dict


class MelSpectrogramLoss(nn.Module):
    """
    Mel-spectrogram loss for perceptually-weighted comparison.
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = None,
    ):
        """
        Initialize mel-spectrogram loss.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop size
            n_mels: Number of mel bands
            f_min: Minimum frequency
            f_max: Maximum frequency (defaults to sample_rate/2)
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        
        # Create mel filterbank
        self.register_buffer(
            "mel_basis",
            self._create_mel_filterbank()
        )
        self.register_buffer(
            "window",
            torch.hann_window(n_fft)
        )
        
    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix."""
        import numpy as np
        
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Create mel frequency points
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bin indices
        fft_freqs = np.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filterbank
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        
        for i in range(self.n_mels):
            left = bin_indices[i]
            center = bin_indices[i + 1]
            right = bin_indices[i + 2]
            
            # Rising slope
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            
            # Falling slope
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)
        
        return torch.FloatTensor(filterbank)
    
    def mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram.
        
        Args:
            x: Input waveform [B, T] or [T]
            
        Returns:
            Mel-spectrogram [B, n_mels, T'] or [n_mels, T']
        """
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        
        window = self.window.to(x.device)
        mel_basis = self.mel_basis.to(x.device)
        
        # STFT
        stft_out = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        
        # Magnitude spectrogram
        mag = stft_out.abs()  # [B, F, T']
        
        # Apply mel filterbank
        mel = torch.matmul(mel_basis, mag)  # [B, n_mels, T']
        
        if squeeze_output:
            mel = mel.squeeze(0)
        
        return mel
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mel-spectrogram loss.
        
        Args:
            pred: Predicted waveform
            target: Target waveform
            
        Returns:
            L1 loss on log mel-spectrograms
        """
        pred_mel = self.mel_spectrogram(pred)
        target_mel = self.mel_spectrogram(target)
        
        # Log compression
        pred_mel = torch.log(pred_mel + 1e-8)
        target_mel = torch.log(target_mel + 1e-8)
        
        return F.l1_loss(pred_mel, target_mel)


class CombinedAudioLoss(nn.Module):
    """
    Combined loss for AudioGS training.
    
    Includes phase-aware Complex STFT Loss for better PESQ.
    
    AUDIO PHYSICS - CRITICAL WEIGHT REQUIREMENT:
    Time-domain MSE is non-convex (Phase Retrieval problem).
    STFT-domain loss must DOMINATE (>90% weight) over time-domain L1 to ensure convergence.
    
    Recommended weights:
    - stft_weight + mel_weight + phase_weight: Should be ~90%+ of total
    - time_weight: Should be ≤10% of spectral weights
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        fft_sizes: List[int] = [2048, 1024, 512, 128],
        hop_sizes: List[int] = [512, 256, 128, 32],
        win_lengths: List[int] = [2048, 1024, 512, 128],
        n_mels: int = 80,
        stft_weight: float = 1.0,
        mel_weight: float = 1.0,
        time_weight: float = 0.5,
        phase_weight: float = 0.5,     # Complex STFT for phase alignment
        amp_reg_weight: float = 0.01,
        pre_emp_weight: float = 20.0,
    ):
        super().__init__()
        
        # AUDIO PHYSICS: Warn if time-domain weight is too high
        spectral_total = stft_weight + mel_weight + phase_weight
        if time_weight > 0.1 * spectral_total:
            import warnings
            warnings.warn(
                f"[CombinedAudioLoss] time_weight={time_weight} is >10% of spectral weights "
                f"({spectral_total}). This may cause non-convex optimization issues. "
                f"Recommended: time_weight ≤ {0.1 * spectral_total:.2f}",
                UserWarning
            )
        
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.time_weight = time_weight
        self.phase_weight = phase_weight
        self.amp_reg_weight = amp_reg_weight
        self.pre_emp_weight = pre_emp_weight
        
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            spectral_weight=1.0,
            log_mag_weight=1.0,
            time_domain_weight=0.0,
        )
        
        # Phase-aware loss (NEW)
        self.complex_stft_loss = ComplexSTFTLoss(
            fft_sizes=fft_sizes[:3],  # Use top 3 scales
            hop_sizes=hop_sizes[:3],
            win_lengths=win_lengths[:3],
        )
        
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

    def pre_emphasis(self, x: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
        """
        Apply pre-emphasis filter to boost high frequencies.
        y[t] = x[t] - coeff * x[t-1]
        """
        # x shape: [B, T] or [T]
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Pad left with one zero to maintain length
        x_pad = F.pad(x.unsqueeze(1), (1, 0), "constant", 0).squeeze(1)
        return x_pad[:, 1:] - coeff * x_pad[:, :-1]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model_amplitude: Optional[torch.Tensor] = None,
        model_sigma: Optional[torch.Tensor] = None,
        model_phase_raw: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sigma_diversity_weight: float = 0.001,
        phase_reg_weight: float = 0.1,  # Strong enough to keep vectors healthy
    ) -> Tuple[torch.Tensor, dict]:
        
        # 1. Standard Losses (Dominantly Low/Mid Freq)
        stft_total, stft_dict = self.stft_loss(pred, target)
        mel_loss = self.mel_loss(pred, target)
        time_loss = F.l1_loss(pred, target)
        
        # 2. Phase-aware Complex STFT Loss (crucial for PESQ > 3.0)
        phase_loss = self.complex_stft_loss(pred, target)
        
        # 3. Pre-emphasis Loss (Dominantly High Freq)
        pred_emp = self.pre_emphasis(pred)
        target_emp = self.pre_emphasis(target)
        pre_emp_loss = F.l1_loss(pred_emp, target_emp)
        
        # 4. Regularization
        amp_reg = 0.0
        if model_amplitude is not None:
            amp_reg = F.relu(model_amplitude - 0.5).mean()
        
        sigma_div = 0.0
        if model_sigma is not None:
            log_sigma = torch.log(model_sigma + 1e-8)
            sigma_variance = log_sigma.var()
            sigma_div = F.relu(1.0 - sigma_variance)
        
        # 5. Phase Vector Circular Regularization (NEW)
        # Penalize deviation from unit circle to prevent vector collapse (→ atan2 gradient explosion)
        phase_reg = 0.0
        if model_phase_raw is not None:
            cos_raw, sin_raw = model_phase_raw
            radius_sq = cos_raw**2 + sin_raw**2
            phase_reg = ((radius_sq - 1.0)**2).mean()
        
        # Combine all losses
        total_loss = (
            self.stft_weight * stft_total +
            self.mel_weight * mel_loss +
            self.time_weight * time_loss +
            self.phase_weight * phase_loss +
            self.amp_reg_weight * amp_reg +
            sigma_diversity_weight * sigma_div +
            self.pre_emp_weight * pre_emp_loss +
            phase_reg_weight * phase_reg  # NEW: circular regularization
        )
        
        loss_dict = {
            "stft": stft_total.item(),
            "mel": mel_loss.item(),
            "time": time_loss.item(),
            "phase": phase_loss.item(),
            "pre_emp": pre_emp_loss.item(),
            "phase_reg": phase_reg if isinstance(phase_reg, float) else phase_reg.item(),  # NEW
            "total": total_loss.item(),
        }
        
        return total_loss, loss_dict
