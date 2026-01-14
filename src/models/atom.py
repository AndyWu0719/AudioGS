"""
AudioGS Model: Learnable Gabor Atoms for Audio Representation.
Updated with Phase Initialization from STFT.
"""
import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional

class AudioGSModel(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        sample_rate: int = 24000,
        audio_duration: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.nyquist_freq = sample_rate / 2.0
        self.audio_duration = audio_duration
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize placeholders
        self._init_empty_parameters(num_atoms)
        
    def _init_empty_parameters(self, num_atoms: int):
        self._amplitude_logit = nn.Parameter(torch.zeros(num_atoms, device=self.device))
        self._tau = nn.Parameter(torch.zeros(num_atoms, device=self.device))
        self._omega_logit = nn.Parameter(torch.zeros(num_atoms, device=self.device))
        self._sigma_logit = nn.Parameter(torch.zeros(num_atoms, device=self.device))
        self._phi = nn.Parameter(torch.zeros(num_atoms, device=self.device))
        self._gamma = nn.Parameter(torch.zeros(num_atoms, device=self.device))
        
        self.register_buffer("tau_grad_accum", torch.zeros(num_atoms, device=self.device))
        self.register_buffer("grad_accum_count", torch.zeros(1, device=self.device))

    def init_random(self):
        """Fallback: Random initialization."""
        with torch.no_grad():
            self._amplitude_logit.data.normal_(0, 0.1)
            self._tau.data.uniform_(0, self.audio_duration)
            self._omega_logit.data.normal_(0, 1.0)
            self._sigma_logit.data.normal_(-4.0, 0.5) 
            self._phi.data.uniform_(0, 2 * math.pi)
            
    def initialize_from_audio(self, waveform: torch.Tensor, use_f0_init: bool = True, n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize atom parameters with F0-guided or STFT-based strategy.
        
        F0-Guided Strategy:
        - 40% atoms: Initialize ω at F0 (fundamental frequency)
        - 40% atoms: Initialize ω at harmonics (2×F0, 3×F0, 4×F0, 5×F0)
        - 20% atoms: Random frequency (for unvoiced sounds)
        
        Args:
            waveform: Audio waveform tensor
            use_f0_init: If True, use F0-guided initialization
            n_fft: FFT size for STFT fallback
            hop_length: Hop length for STFT
        """
        print(f"[AudioGSModel] Initializing from audio (use_f0_init={use_f0_init})...")
        
        with torch.no_grad():
            waveform = waveform.to(self.device)
            if waveform.dim() == 2:
                waveform = waveform.squeeze(0)
            
            waveform_np = waveform.cpu().numpy()
            num_atoms = self.num_atoms
            
            # Generate random tau positions for all atoms
            taus = torch.rand(num_atoms, device=self.device) * self.audio_duration
            
            if use_f0_init:
                omegas, phis = self._f0_guided_init(waveform_np, taus)
            else:
                omegas, phis = self._stft_init(waveform, taus, n_fft, hop_length)
            
            # Assign to parameters
            self._tau.data = taus
            
            # Convert omega (Hz) to logit
            omega_normalized = (omegas / self.nyquist_freq).clamp(1e-5, 1 - 1e-5)
            self._omega_logit.data = torch.log(omega_normalized / (1 - omega_normalized))
            
            self._phi.data = phis
            
            # Amplitude: small initial values
            self._amplitude_logit.data.fill_(-2.0)  # softplus(-2) ≈ 0.13
            
            # Sigma: 2-5ms range
            self._sigma_logit.data.uniform_(math.log(0.002), math.log(0.005))
            
            # Gamma (chirp): zero initially
            self._gamma.data.zero_()
            
            print(f"[AudioGSModel] Initialized {num_atoms} atoms.")
    
    def _f0_guided_init(self, waveform_np, taus: torch.Tensor):
        """
        F0-guided frequency initialization using librosa.pyin.
        
        Returns:
            omegas: Frequency values in Hz for each atom
            phis: Phase values for each atom
        """
        import librosa
        from scipy.interpolate import interp1d
        import numpy as np
        
        num_atoms = len(taus)
        taus_np = taus.cpu().numpy()
        
        # Extract F0 using pyin
        f0, voiced_flag, voiced_prob = librosa.pyin(
            waveform_np,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=self.sample_rate,
            frame_length=2048,
            hop_length=512,
        )
        
        # Time axis for F0 frames
        f0_times = librosa.times_like(f0, sr=self.sample_rate, hop_length=512)
        
        # Handle NaN values for interpolation
        f0_valid = np.where(np.isnan(f0), 0, f0)
        voiced_mask = ~np.isnan(f0)
        
        # CONFIDENCE CHECK: If voiced ratio is too low, fallback to STFT init
        voiced_ratio = np.mean(voiced_mask)
        if voiced_ratio < 0.3:
            print(f"[F0 Init] Low voiced ratio ({voiced_ratio:.1%}), falling back to STFT init")
            return self._stft_init(
                torch.from_numpy(waveform_np).float().to(self.device),
                taus, n_fft=2048, hop_length=512
            )
        
        # Create interpolator for F0
        if np.any(voiced_mask):
            # Use valid F0 values for interpolation, fill NaN regions with 0
            f0_interp = interp1d(
                f0_times, f0_valid,
                kind='linear', bounds_error=False, fill_value=0
            )
            f0_at_taus = f0_interp(taus_np)
            
            # Create voiced/unvoiced mask for atoms
            voiced_interp = interp1d(
                f0_times, voiced_mask.astype(float),
                kind='nearest', bounds_error=False, fill_value=0
            )
            is_voiced = voiced_interp(taus_np) > 0.5
        else:
            # Entirely unvoiced
            f0_at_taus = np.zeros(num_atoms)
            is_voiced = np.zeros(num_atoms, dtype=bool)
        
        # Distribution strategy
        num_f0 = int(num_atoms * 0.4)         # 40% at F0
        num_harmonic = int(num_atoms * 0.4)   # 40% at harmonics
        num_random = num_atoms - num_f0 - num_harmonic  # 20% random
        
        omegas = np.zeros(num_atoms)
        
        # Shuffle indices for random assignment
        indices = np.random.permutation(num_atoms)
        f0_indices = indices[:num_f0]
        harmonic_indices = indices[num_f0:num_f0 + num_harmonic]
        random_indices = indices[num_f0 + num_harmonic:]
        
        # F0 atoms (40%)
        for idx in f0_indices:
            if is_voiced[idx] and f0_at_taus[idx] > 50:
                omegas[idx] = f0_at_taus[idx]
            else:
                # Unvoiced region: random high frequency (fricatives)
                omegas[idx] = np.random.uniform(2000, 8000)
        
        # Harmonic atoms (40%): 2×F0 to 5×F0
        harmonics = [2, 3, 4, 5]
        for i, idx in enumerate(harmonic_indices):
            h = harmonics[i % len(harmonics)]
            if is_voiced[idx] and f0_at_taus[idx] > 50:
                harmonic_freq = f0_at_taus[idx] * h
                # Clamp to Nyquist
                omegas[idx] = min(harmonic_freq, self.nyquist_freq * 0.95)
            else:
                # Unvoiced: random
                omegas[idx] = np.random.uniform(1000, 10000)
        
        # Random atoms (20%): full spectrum
        for idx in random_indices:
            omegas[idx] = np.random.uniform(100, self.nyquist_freq * 0.9)
        
        # =====================================================
        # Phase Initialization from STFT (Bilinear Interpolation)
        # =====================================================
        # CRITICAL: Interpolate complex vectors, NOT raw phase angles!
        # Raw phase wraps at 2π causing discontinuities.
        # We compute weighted sum of 4 nearest complex STFT bins,
        # then extract angle from the interpolated complex value.
        
        waveform_tensor = torch.from_numpy(waveform_np).float()
        stft = torch.stft(
            waveform_tensor, 
            n_fft=2048, 
            hop_length=512, 
            return_complex=True,
            window=torch.hann_window(2048),
            center=True,
        )
        # stft shape: [num_freq_bins, num_frames]
        num_freq_bins, num_frames = stft.shape
        
        # Compute continuous indices for each atom's (tau, omega)
        # Time index: tau -> frame
        hop_length = 512
        frame_indices = (taus_np * self.sample_rate / hop_length).astype(np.float32)
        frame_indices = np.clip(frame_indices, 0, num_frames - 1 - 1e-6)
        
        # Frequency index: omega -> bin
        freq_resolution = self.sample_rate / 2048  # Hz per bin
        freq_indices = (omegas / freq_resolution).astype(np.float32)
        freq_indices = np.clip(freq_indices, 0, num_freq_bins - 1 - 1e-6)
        
        phis = np.zeros(num_atoms)
        stft_np = stft.numpy()  # Complex numpy array
        
        for i in range(num_atoms):
            # Get the 4 nearest neighbors for bilinear interpolation
            f_idx = freq_indices[i]
            t_idx = frame_indices[i]
            
            # Use distinct variable names to avoid shadowing f0 (F0 pitch array)
            fb0, fb1 = int(np.floor(f_idx)), int(np.ceil(f_idx))
            tb0, tb1 = int(np.floor(t_idx)), int(np.ceil(t_idx))
            
            # Clamp to valid range
            fb1 = min(fb1, num_freq_bins - 1)
            tb1 = min(tb1, num_frames - 1)
            
            # Bilinear weights
            wf = f_idx - fb0  # Weight for fb1
            wt = t_idx - tb0  # Weight for tb1
            
            # Get 4 complex values
            c00 = stft_np[fb0, tb0]
            c01 = stft_np[fb0, tb1]
            c10 = stft_np[fb1, tb0]
            c11 = stft_np[fb1, tb1]
            
            # Bilinear interpolation of complex vectors
            c_interp = (
                (1 - wf) * (1 - wt) * c00 +
                (1 - wf) * wt * c01 +
                wf * (1 - wt) * c10 +
                wf * wt * c11
            )
            
            # Extract angle from interpolated complex value
            phis[i] = np.angle(c_interp)
        
        omegas_tensor = torch.from_numpy(omegas).float().to(self.device)
        phis_tensor = torch.from_numpy(phis).float().to(self.device)
        
        # Note: f0_pitch is the original F0 array from pyin (not the loop variable)
        print(f"[F0 Init] Voiced atoms: {np.sum(is_voiced)}/{num_atoms}, "
              f"Phase: STFT bilinear interpolation")
        
        return omegas_tensor, phis_tensor
    
    def _stft_init(self, waveform: torch.Tensor, taus: torch.Tensor, n_fft: int, hop_length: int):
        """Fallback STFT-based initialization."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        stft = torch.stft(
            waveform, n_fft=n_fft, hop_length=hop_length,
            return_complex=True, center=True,
            window=torch.hann_window(n_fft).to(self.device)
        )
        mag = stft.abs().squeeze(0)
        phase = stft.angle().squeeze(0)
        
        num_freq_bins, num_frames = mag.shape
        freqs = torch.linspace(0, self.nyquist_freq, num_freq_bins, device=self.device)
        frame_times = torch.arange(num_frames, device=self.device) * hop_length / self.sample_rate
        
        # For each atom, find nearest frame and sample frequency
        num_atoms = len(taus)
        omegas = torch.zeros(num_atoms, device=self.device)
        phis = torch.zeros(num_atoms, device=self.device)
        
        for i, t in enumerate(taus):
            frame_idx = int(t / self.audio_duration * (num_frames - 1))
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            
            # Sample from magnitude distribution
            frame_mag = mag[:, frame_idx]
            if frame_mag.sum() > 0:
                probs = frame_mag / frame_mag.sum()
                freq_idx = torch.multinomial(probs, 1).item()
            else:
                freq_idx = torch.randint(0, num_freq_bins, (1,)).item()
            
            omegas[i] = freqs[freq_idx]
            phis[i] = phase[freq_idx, frame_idx]
        
        return omegas, phis
    # --- Properties and Helpers (Same as before) ---
    @property
    def num_atoms(self) -> int: return self._amplitude_logit.shape[0]

    @property
    def amplitude(self) -> torch.Tensor: return torch.nn.functional.softplus(self._amplitude_logit)

    @property
    def tau(self) -> torch.Tensor: return self._tau

    @property
    def omega(self) -> torch.Tensor: return torch.sigmoid(self._omega_logit) * self.nyquist_freq

    @property
    def sigma(self) -> torch.Tensor: 
        min_sigma_logit = math.log(0.002)  # Allow 2ms atoms for transients
        return torch.exp(torch.clamp(self._sigma_logit, min=min_sigma_logit))

    @property
    def phi(self) -> torch.Tensor: 
        # Constrain phase to [-π, π] to prevent divergence
        return torch.remainder(self._phi + math.pi, 2 * math.pi) - math.pi

    @property
    def gamma(self) -> torch.Tensor: return self._gamma

    def get_all_params(self):
        return (self.amplitude, self.tau, self.omega, self.sigma, self.phi, self.gamma)

    def get_optimizer_param_groups(self, config: dict):
        return [
            {"params": [self._amplitude_logit], "lr": config.get("lr_amplitude", 0.01)},
            {"params": [self._tau], "lr": config.get("lr_position", 0.001)},
            {"params": [self._omega_logit], "lr": config.get("lr_frequency", 0.005)}, # Increased
            {"params": [self._sigma_logit], "lr": config.get("lr_sigma", 0.001)},
            {"params": [self._phi], "lr": config.get("lr_phase", 0.01)},
            {"params": [self._gamma], "lr": config.get("lr_chirp", 0.001)}, # Increased
        ]

    # --- Density Control Helpers ---
    def accumulate_gradients(self):
        if self._tau.grad is not None:
            self.tau_grad_accum += self._tau.grad.abs()
            self.grad_accum_count += 1
            
    def get_average_gradients(self) -> torch.Tensor:
        if self.grad_accum_count > 0: return self.tau_grad_accum / self.grad_accum_count
        return self.tau_grad_accum
        
    def reset_gradient_accumulators(self):
        self.tau_grad_accum.zero_()
        self.grad_accum_count.zero_()

    def add_atoms(self, amplitude_logit, tau, omega_logit, sigma_logit, phi, gamma):
        with torch.no_grad():
            self._amplitude_logit = nn.Parameter(torch.cat([self._amplitude_logit.data, amplitude_logit], 0))
            self._tau = nn.Parameter(torch.cat([self._tau.data, tau], 0))
            self._omega_logit = nn.Parameter(torch.cat([self._omega_logit.data, omega_logit], 0))
            self._sigma_logit = nn.Parameter(torch.cat([self._sigma_logit.data, sigma_logit], 0))
            self._phi = nn.Parameter(torch.cat([self._phi.data, phi], 0))
            self._gamma = nn.Parameter(torch.cat([self._gamma.data, gamma], 0))
            self.tau_grad_accum = torch.cat([self.tau_grad_accum, torch.zeros(amplitude_logit.shape[0], device=self.device)], 0)
    
    def remove_atoms(self, mask):
        with torch.no_grad():
            self._amplitude_logit = nn.Parameter(self._amplitude_logit.data[mask])
            self._tau = nn.Parameter(self._tau.data[mask])
            self._omega_logit = nn.Parameter(self._omega_logit.data[mask])
            self._sigma_logit = nn.Parameter(self._sigma_logit.data[mask])
            self._phi = nn.Parameter(self._phi.data[mask])
            self._gamma = nn.Parameter(self._gamma.data[mask])
            self.tau_grad_accum = self.tau_grad_accum[mask]

    def clone_atoms_by_indices(self, indices):
        if len(indices) == 0: return 0
        with torch.no_grad():
            new_amp = self._amplitude_logit.data[indices].clone()
            new_tau = self._tau.data[indices].clone() + torch.randn_like(self._tau.data[indices]) * 0.001
            # Add freq noise
            new_omega = self._omega_logit.data[indices].clone() + torch.randn_like(self._omega_logit.data[indices]) * 0.02
            new_sigma = self._sigma_logit.data[indices].clone()
            new_phi = self._phi.data[indices].clone()
            new_gamma = self._gamma.data[indices].clone()
        self.add_atoms(new_amp, new_tau, new_omega, new_sigma, new_phi, new_gamma)
        return len(indices)

    def split_atoms_by_indices(self, indices, scale_factor=1.6):
        if len(indices) == 0: return 0
        with torch.no_grad():
            sigma = self.sigma[indices]
            tau = self._tau.data[indices]
            offset = sigma * 0.5
            
            # Atom 1
            new_amp1 = self._amplitude_logit.data[indices].clone() - 0.3
            new_tau1 = tau - offset
            # Add freq noise to explore
            new_omega1 = self._omega_logit.data[indices].clone() + torch.randn_like(self._omega_logit.data[indices]) * 0.05
            new_sigma1 = self._sigma_logit.data[indices].clone() - math.log(scale_factor)
            new_phi1 = self._phi.data[indices].clone()
            new_gamma1 = self._gamma.data[indices].clone()
            
            # Atom 2
            new_amp2 = self._amplitude_logit.data[indices].clone() - 0.3
            new_tau2 = tau + offset
            new_omega2 = self._omega_logit.data[indices].clone() - torch.randn_like(self._omega_logit.data[indices]) * 0.05
            new_sigma2 = new_sigma1.clone()
            new_phi2 = self._phi.data[indices].clone() + math.pi * 0.5
            new_gamma2 = self._gamma.data[indices].clone()
            
        self.add_atoms(new_amp1, new_tau1, new_omega1, new_sigma1, new_phi1, new_gamma1)
        self.add_atoms(new_amp2, new_tau2, new_omega2, new_sigma2, new_phi2, new_gamma2)
        return len(indices) * 2

    def state_dict_full(self):
        return {
            "amplitude_logit": self._amplitude_logit.data,
            "tau": self._tau.data,
            "omega_logit": self._omega_logit.data,
            "sigma_logit": self._sigma_logit.data,
            "phi": self._phi.data,
            "gamma": self._gamma.data,
            "tau_grad_accum": self.tau_grad_accum,
            "grad_accum_count": self.grad_accum_count,
        }
    
    def load_state_dict_full(self, state):
        self._amplitude_logit = nn.Parameter(state["amplitude_logit"])
        self._tau = nn.Parameter(state["tau"])
        self._omega_logit = nn.Parameter(state["omega_logit"])
        self._sigma_logit = nn.Parameter(state["sigma_logit"])
        self._phi = nn.Parameter(state["phi"])
        self._gamma = nn.Parameter(state["gamma"])
        self.tau_grad_accum = state["tau_grad_accum"]
        self.grad_accum_count = state["grad_accum_count"]