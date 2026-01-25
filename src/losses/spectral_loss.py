"""
Multi-Resolution STFT Loss for Audio Reconstruction.

Major Refactor: Compute STFTs ONCE, eliminate 3x redundancy.
Adds multiscale_stft helper and pre-computed spectrogram API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


def multiscale_stft(
    x: torch.Tensor,
    fft_sizes: List[int],
    hop_sizes: List[int],
    win_lengths: List[int],
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute STFT once at multiple resolutions.
    
    OPTIMIZATION: This eliminates redundant STFT computation across
    STFTLoss, ComplexSTFTLoss, and MelSpectrogramLoss.
    
    Args:
        x: Input waveform [B, T] or [T]
        fft_sizes: List of FFT sizes
        hop_sizes: List of hop sizes
        win_lengths: List of window lengths
        
    Returns:
        Dict mapping fft_size -> (magnitude, complex_stft)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    results = {}
    for fft_size, hop_size, win_len in zip(fft_sizes, hop_sizes, win_lengths):
        window = torch.hann_window(win_len, device=x.device)
        stft_out = torch.stft(
            x, n_fft=fft_size, hop_length=hop_size, win_length=win_len,
            window=window, return_complex=True, center=True, pad_mode="reflect"
        )
        mag = stft_out.abs()
        results[fft_size] = (mag, stft_out)
    
    return results


class STFTLoss(nn.Module):
    """Single-scale STFT loss with pre-computed spectrogram support."""
    
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.eps = eps
        
        self.register_buffer(
            "window",
            torch.hann_window(win_length)
        )
        
    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude."""
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        
        window = self.window.to(x.device)
        
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
        
        mag = stft_out.abs()
        
        if squeeze_output:
            mag = mag.squeeze(0)
        
        return mag
    
    def forward(
        self,
        pred: torch.Tensor = None,
        target: torch.Tensor = None,
        pred_mag: torch.Tensor = None,
        target_mag: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute STFT loss from raw audio OR pre-computed magnitudes.
        
        REFACTOR: Supports pre-computed spectrograms to eliminate redundancy.
        """
        # Use pre-computed if available
        if pred_mag is None:
            pred_mag = self.stft(pred)
        if target_mag is None:
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
    
    REFACTOR: Supports pre-computed complex spectrograms.
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
        
        for i, win_len in enumerate(win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(win_len))
    
    def _get_window(self, idx: int, device: torch.device) -> torch.Tensor:
        return getattr(self, f"window_{idx}").to(device)
    
    def forward(
        self,
        pred: torch.Tensor = None,
        target: torch.Tensor = None,
        pred_stft_dict: Dict[int, Tuple] = None,
        target_stft_dict: Dict[int, Tuple] = None,
    ) -> torch.Tensor:
        """
        Compute multi-scale complex STFT loss.
        
        REFACTOR: Supports pre-computed complex spectrograms to eliminate redundancy.
        """
        # Ensure 2D if computing from raw
        if pred is not None and pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target is not None and target.dim() == 1:
            target = target.unsqueeze(0)
        
        total_loss = 0.0
        
        for i, (fft_size, hop_size, win_len) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)
        ):
            # Use pre-computed if available
            if pred_stft_dict is not None and fft_size in pred_stft_dict:
                _, pred_stft = pred_stft_dict[fft_size]
                _, target_stft = target_stft_dict[fft_size]
            else:
                # Compute from raw audio
                window = self._get_window(i, pred.device)
                pred_stft = torch.stft(
                    pred, n_fft=fft_size, hop_length=hop_size, win_length=win_len,
                    window=window, return_complex=True, center=True, pad_mode="reflect"
                )
                target_stft = torch.stft(
                    target, n_fft=fft_size, hop_length=hop_size, win_length=win_len,
                    window=window, return_complex=True, center=True, pad_mode="reflect"
                )
            
            # Complex L2 loss
            complex_diff = pred_stft - target_stft
            loss = complex_diff.abs().mean()
            
            total_loss = total_loss + loss
        
        return total_loss / len(self.fft_sizes)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss with pre-computed support.
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
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.spectral_weight = spectral_weight
        self.log_mag_weight = log_mag_weight
        self.time_domain_weight = time_domain_weight
        
        self.stft_losses = nn.ModuleList([
            STFTLoss(fft, hop, win)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        
    def forward(
        self,
        pred: torch.Tensor = None,
        target: torch.Tensor = None,
        pred_stft_dict: Dict[int, Tuple] = None,
        target_stft_dict: Dict[int, Tuple] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-resolution STFT loss.
        
        REFACTOR: Supports pre-computed spectrograms.
        """
        sc_loss_total = 0.0
        log_mag_loss_total = 0.0
        
        for i, stft_loss in enumerate(self.stft_losses):
            fft_size = self.fft_sizes[i]
            
            # Use pre-computed if available
            if pred_stft_dict is not None and fft_size in pred_stft_dict:
                pred_mag, _ = pred_stft_dict[fft_size]
                target_mag, _ = target_stft_dict[fft_size]
                sc_loss, log_mag_loss = stft_loss(pred_mag=pred_mag, target_mag=target_mag)
            else:
                sc_loss, log_mag_loss = stft_loss(pred, target)
            
            sc_loss_total += sc_loss
            log_mag_loss_total += log_mag_loss
        
        num_scales = len(self.stft_losses)
        sc_loss_avg = sc_loss_total / num_scales
        log_mag_loss_avg = log_mag_loss_total / num_scales
        
        # Time-domain L1 loss
        time_loss = F.l1_loss(pred, target) if pred is not None else torch.tensor(0.0)
        
        total_loss = (
            self.spectral_weight * sc_loss_avg +
            self.log_mag_weight * log_mag_loss_avg +
            self.time_domain_weight * time_loss
        )
        
        loss_dict = {
            "spectral_convergence": sc_loss_avg.item(),
            "log_magnitude": log_mag_loss_avg.item(),
            "time_domain": time_loss.item() if torch.is_tensor(time_loss) else time_loss,
            "total": total_loss.item(),
        }
        
        return total_loss, loss_dict


class MelSpectrogramLoss(nn.Module):
    """
    Mel-spectrogram loss with pre-computed magnitude support.
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
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        
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
        
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        fft_freqs = np.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        
        for i in range(self.n_mels):
            left = bin_indices[i]
            center = bin_indices[i + 1]
            right = bin_indices[i + 2]
            
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)
        
        filterbank = filterbank + 1e-10
        
        return torch.FloatTensor(filterbank)
    
    def mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute power mel-spectrogram from raw audio."""
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        
        window = self.window.to(x.device)
        mel_basis = self.mel_basis.to(x.device)
        
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
        
        power = stft_out.abs().pow(2)
        mel = torch.matmul(mel_basis, power)
        
        if squeeze_output:
            mel = mel.squeeze(0)
        
        return mel
    
    def mel_from_mag(self, mag: torch.Tensor) -> torch.Tensor:
        """Compute mel-spectrogram from pre-computed magnitude."""
        mel_basis = self.mel_basis.to(mag.device)
        power = mag.pow(2)
        return torch.matmul(mel_basis, power)
    
    def forward(
        self,
        pred: torch.Tensor = None,
        target: torch.Tensor = None,
        pred_mag: torch.Tensor = None,
        target_mag: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute mel-spectrogram loss.
        
        REFACTOR: Supports pre-computed magnitude spectrograms.
        """
        if pred_mag is not None:
            pred_mel = self.mel_from_mag(pred_mag)
            target_mel = self.mel_from_mag(target_mag)
        else:
            pred_mel = self.mel_spectrogram(pred)
            target_mel = self.mel_spectrogram(target)
        
        # Log compression
        pred_mel = torch.log(pred_mel + 1e-8)
        target_mel = torch.log(target_mel + 1e-8)
        
        return F.l1_loss(pred_mel, target_mel)


class CombinedAudioLoss(nn.Module):
    """
    Combined loss for AudioGS training.
    
    REFACTOR: Computes STFTs ONCE and reuses across sub-losses.
    Adds phase vector regularization for unit-circle constraint.
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
        phase_weight: float = 0.5,
        amp_reg_weight: float = 0.01,
        pre_emp_weight: float = 20.0,
        hf_weight: float = 0.0,
        hf_min_freq_ratio: float = 0.6,
        energy_weight: float = 0.0,
        gamma_reg_weight: float = 0.0,
        phase_reg_weight: float = 0.1,
        spec_tv_weight: float = 0.0,
        spec_tv_freq_ratio: float = 0.5,
    ):
        super().__init__()
        
        # Store config for STFT computation
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        spectral_total = stft_weight + mel_weight + phase_weight
        if time_weight > 0.1 * spectral_total:
            import warnings
            warnings.warn(
                f"[CombinedAudioLoss] time_weight={time_weight} is >10% of spectral weights "
                f"({spectral_total}). This may cause non-convex optimization issues.",
                UserWarning
            )
        
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.time_weight = time_weight
        self.phase_weight = phase_weight
        self.amp_reg_weight = amp_reg_weight
        self.pre_emp_weight = pre_emp_weight
        self.hf_weight = hf_weight
        self.hf_min_freq_ratio = hf_min_freq_ratio
        self.energy_weight = energy_weight
        self.gamma_reg_weight = gamma_reg_weight
        self.phase_reg_weight = phase_reg_weight
        self.spec_tv_weight = spec_tv_weight
        self.spec_tv_freq_ratio = spec_tv_freq_ratio
        
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            spectral_weight=1.0,
            log_mag_weight=1.0,
            time_domain_weight=0.0,
        )
        
        self.complex_stft_loss = ComplexSTFTLoss(
            fft_sizes=fft_sizes[:3],
            hop_sizes=hop_sizes[:3],
            win_lengths=win_lengths[:3],
        )
        
        # Use 1024 FFT for mel (standard for speech)
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=sample_rate,
            n_fft=1024,
            n_mels=n_mels,
        )

    def pre_emphasis(self, x: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
        """Apply pre-emphasis filter to boost high frequencies."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
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
        phase_reg_weight: Optional[float] = None,
        model_gamma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined audio loss.
        
        REFACTOR: 
        - Computes STFTs ONCE via multiscale_stft helper
        - Uses model_phase_raw for unit-circle regularization
        
        Args:
            pred: Predicted waveform
            target: Target waveform
            model_amplitude: Atom amplitudes for regularization
            model_sigma: Atom sigmas for diversity regularization
            model_phase_raw: Tuple of (cos, sin) from model.phase_vector for unit-circle regularization
            sigma_diversity_weight: Weight for sigma diversity loss
            phase_reg_weight: Weight for phase vector unit-circle regularization
        """
        # Clamp prediction to avoid NaN in log-losses
        pred = torch.clamp(pred, min=-10.0, max=10.0)
        
        # OPTIMIZATION: Compute all STFTs ONCE
        pred_stft_dict = multiscale_stft(pred, self.fft_sizes, self.hop_sizes, self.win_lengths)
        target_stft_dict = multiscale_stft(target, self.fft_sizes, self.hop_sizes, self.win_lengths)
        
        # 1. Multi-resolution STFT Loss (using pre-computed)
        stft_total, stft_dict = self.stft_loss(
            pred, target,
            pred_stft_dict=pred_stft_dict,
            target_stft_dict=target_stft_dict,
        )
        
        # 2. Mel Loss (using pre-computed 1024 magnitude)
        if 1024 in pred_stft_dict:
            pred_mag_1024 = pred_stft_dict[1024][0]
            target_mag_1024 = target_stft_dict[1024][0]
            mel_loss = self.mel_loss(pred_mag=pred_mag_1024, target_mag=target_mag_1024)
        else:
            mel_loss = self.mel_loss(pred, target)
        
        # 3. Phase-aware Complex STFT Loss (using pre-computed)
        phase_loss = self.complex_stft_loss(
            pred, target,
            pred_stft_dict=pred_stft_dict,
            target_stft_dict=target_stft_dict,
        )
        
        # 4. Time-domain L1 Loss
        time_loss = F.l1_loss(pred, target)
        
        # 5. Pre-emphasis Loss (High Frequency focus)
        pred_emp = self.pre_emphasis(pred)
        target_emp = self.pre_emphasis(target)
        pre_emp_loss = F.l1_loss(pred_emp, target_emp)

        # 5a. Short-time energy envelope loss
        energy_loss = 0.0
        if self.energy_weight > 0.0:
            if 1024 in pred_stft_dict:
                pred_mag_1024 = pred_stft_dict[1024][0]
                target_mag_1024 = target_stft_dict[1024][0]
            else:
                first_key = self.fft_sizes[0]
                pred_mag_1024 = pred_stft_dict[first_key][0]
                target_mag_1024 = target_stft_dict[first_key][0]
            pred_energy = torch.log(pred_mag_1024.pow(2).mean(dim=-2) + 1e-8)
            target_energy = torch.log(target_mag_1024.pow(2).mean(dim=-2) + 1e-8)
            energy_loss = F.l1_loss(pred_energy, target_energy)

        # 5b. High-frequency STFT loss (direct HF detail emphasis)
        hf_loss = 0.0
        if self.hf_weight > 0.0:
            hf_total = 0.0
            hf_scales = 0
            for fft_size in self.fft_sizes:
                pred_mag, _ = pred_stft_dict[fft_size]
                target_mag, _ = target_stft_dict[fft_size]
                cutoff_bin = int(pred_mag.shape[-2] * self.hf_min_freq_ratio)
                if cutoff_bin < pred_mag.shape[-2]:
                    pred_hf = pred_mag[:, cutoff_bin:, :]
                    target_hf = target_mag[:, cutoff_bin:, :]
                    hf_total += F.l1_loss(
                        torch.log(pred_hf + 1e-8),
                        torch.log(target_hf + 1e-8),
                    )
                    hf_scales += 1
            if hf_scales > 0:
                hf_loss = hf_total / hf_scales
        
        # 5c. Spectrogram smoothness matching (reduce speckle while preserving GT texture)
        spec_tv_loss = 0.0
        if self.spec_tv_weight > 0.0:
            if 1024 in pred_stft_dict:
                pred_mag_tv = pred_stft_dict[1024][0]
                target_mag_tv = target_stft_dict[1024][0]
            else:
                first_key = self.fft_sizes[0]
                pred_mag_tv = pred_stft_dict[first_key][0]
                target_mag_tv = target_stft_dict[first_key][0]

            n_bins = pred_mag_tv.shape[-2]
            start_bin = int(n_bins * self.spec_tv_freq_ratio)
            pred_log = torch.log(pred_mag_tv + 1e-8)
            target_log = torch.log(target_mag_tv + 1e-8)

            pred_sel = pred_log[:, start_bin:, :]
            target_sel = target_log[:, start_bin:, :]

            pred_tv_t = (pred_sel[:, :, 1:] - pred_sel[:, :, :-1]).abs()
            target_tv_t = (target_sel[:, :, 1:] - target_sel[:, :, :-1]).abs()
            pred_tv_f = (pred_sel[:, 1:, :] - pred_sel[:, :-1, :]).abs()
            target_tv_f = (target_sel[:, 1:, :] - target_sel[:, :-1, :]).abs()

            spec_tv_loss = F.l1_loss(pred_tv_t, target_tv_t) + F.l1_loss(pred_tv_f, target_tv_f)

        # 6. Regularization
        amp_reg = 0.0
        if model_amplitude is not None:
            amp_reg = F.relu(model_amplitude - 0.5).mean()
        
        sigma_div = 0.0
        if model_sigma is not None:
            log_sigma = torch.log(model_sigma + 1e-8)
            sigma_variance = log_sigma.var()
            sigma_div = F.relu(1.0 - sigma_variance)
        
        # 7. Phase Vector Unit-Circle Regularization
        # REFACTOR: Penalize deviation from unit circle to prevent vector collapse
        phase_reg = 0.0
        if model_phase_raw is not None:
            cos_raw, sin_raw = model_phase_raw
            radius_sq = cos_raw**2 + sin_raw**2
            phase_reg = ((radius_sq - 1.0)**2).mean()

        gamma_reg = 0.0
        if model_gamma is not None and self.gamma_reg_weight > 0.0:
            gamma_reg = (model_gamma ** 2).mean()

        if phase_reg_weight is None:
            phase_reg_weight = self.phase_reg_weight
        
        # Combine all losses
        total_loss = (
            self.stft_weight * stft_total +
            self.mel_weight * mel_loss +
            self.time_weight * time_loss +
            self.phase_weight * phase_loss +
            self.amp_reg_weight * amp_reg +
            sigma_diversity_weight * sigma_div +
            self.pre_emp_weight * pre_emp_loss +
            phase_reg_weight * phase_reg +
            self.hf_weight * hf_loss +
            self.energy_weight * energy_loss +
            self.gamma_reg_weight * gamma_reg +
            self.spec_tv_weight * spec_tv_loss
        )
        
        loss_dict = {
            "stft": stft_total.item(),
            "mel": mel_loss.item(),
            "time": time_loss.item(),
            "phase": phase_loss.item(),
            "pre_emp": pre_emp_loss.item(),
            "energy": energy_loss if isinstance(energy_loss, float) else energy_loss.item(),
            "hf": hf_loss if isinstance(hf_loss, float) else hf_loss.item(),
            "spec_tv": spec_tv_loss if isinstance(spec_tv_loss, float) else spec_tv_loss.item(),
            "phase_reg": phase_reg if isinstance(phase_reg, float) else phase_reg.item(),
            "gamma_reg": gamma_reg if isinstance(gamma_reg, float) else gamma_reg.item(),
            "total": total_loss.item(),
        }
        
        return total_loss, loss_dict
