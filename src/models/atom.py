"""
AudioGS Model: Learnable Gabor Atoms for Audio Representation.
Major Refactor: Vectorized Phase, Uncapped Harmonics, Weighted Gradient Tracking.
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
        
        # REFACTOR: Vectorized Phase [N, 2] for (cos, sin) representation
        # Initialized with random phases to break symmetry
        self._phi_vector = nn.Parameter(torch.zeros(num_atoms, 2, device=self.device))
        random_angles = torch.rand(num_atoms, device=self.device) * 2 * math.pi
        self._phi_vector.data[:, 0] = torch.cos(random_angles)  # cos(phi)
        self._phi_vector.data[:, 1] = torch.sin(random_angles)  # sin(phi)
        
        self._gamma = nn.Parameter(torch.zeros(num_atoms, device=self.device))
        
        self.register_buffer("tau_grad_accum", torch.zeros(num_atoms, device=self.device))
        self.register_buffer("grad_accum_count", torch.zeros(1, device=self.device))

    def init_random(self):
        """
        Fallback: Random initialization with CONSTANT-Q SIGMA.
        
        AUDIO PHYSICS: Sigma must scale with frequency. A 100Hz wave has
        period 10ms, so sigma must be ≥10ms to represent it without gaps.
        """
        with torch.no_grad():
            self._amplitude_logit.data.normal_(0, 0.1)
            self._tau.data.uniform_(0, self.audio_duration)
            # Init omega uniformly across full range to avoid "dead bands"
            self._omega_logit.data.uniform_(-5.0, 5.0)
            
            # CONSTANT-Q SIGMA: σ = num_cycles / ω (Hz)
            # This ensures atoms can properly represent their frequency
            N_CYCLES = 3.5  # ~3.5 cycles per atom
            omega_hz = torch.sigmoid(self._omega_logit.data) * self.nyquist_freq
            omega_hz = omega_hz.clamp(min=50.0)  # Min 50Hz for numerical stability
            constant_q_sigma = N_CYCLES / omega_hz  # seconds
            constant_q_sigma = constant_q_sigma.clamp(min=0.002, max=0.050)  # 2ms-50ms
            self._sigma_logit.data = torch.log(constant_q_sigma)
            
            # REFACTOR: Random phase vectors (unit circle)
            random_angles = torch.rand(self.num_atoms, device=self.device) * 2 * math.pi
            self._phi_vector.data[:, 0] = torch.cos(random_angles)
            self._phi_vector.data[:, 1] = torch.sin(random_angles)
            
    def initialize_from_audio(
        self, 
        waveform: torch.Tensor, 
        use_f0_init: bool = True, 
        n_fft: int = 2048, 
        hop_length: int = 512
    ):
        """
        Initialize atom parameters with F0-guided or STFT-based strategy.
        
        F0-Guided Strategy:
        - τ: Voiced-weighted sampling (dense in voiced regions, sparse in silence)
        - 40% atoms: Initialize ω at F0 (fundamental frequency)
        - 40% atoms: Initialize ω at harmonics up to 90% Nyquist (UNCAPPED)
        - 20% atoms: Random frequency (for unvoiced sounds)
        
        Args:
            waveform: Audio waveform tensor
            use_f0_init: If True, use F0-guided initialization
            n_fft: FFT size for STFT fallback
            hop_length: Hop length for STFT (PARAMETERIZED, not hardcoded)
        """
        print(f"[AudioGSModel] Initializing from audio (use_f0_init={use_f0_init})...")
        
        # Random init first to ensure no atoms are stuck at default zeros (0.5 Nyquist)
        self.init_random()
        
        with torch.no_grad():
            waveform = waveform.to(self.device)
            if waveform.dim() == 2:
                waveform = waveform.squeeze(0)
            
            waveform_np = waveform.cpu().numpy()
            num_atoms = self.num_atoms
            
            if use_f0_init:
                # F0-guided init returns voiced-weighted taus
                taus, omegas, phis = self._f0_guided_init(waveform_np, num_atoms, hop_length)
            else:
                # Uniform taus for STFT-based init
                taus = torch.rand(num_atoms, device=self.device) * self.audio_duration
                omegas, phis = self._stft_peak_init(waveform, taus, n_fft, hop_length)
            
            # Assign to parameters
            self._tau.data = taus
            
            # Convert omega (Hz) to logit
            omega_normalized = (omegas / self.nyquist_freq).clamp(1e-5, 1 - 1e-5)
            self._omega_logit.data = torch.log(omega_normalized / (1 - omega_normalized))
            
            # REFACTOR: Convert scalar phase to vectorized phase
            self._phi_vector.data[:, 0] = torch.cos(phis)
            self._phi_vector.data[:, 1] = torch.sin(phis)
            
            # Amplitude: small initial values
            self._amplitude_logit.data.fill_(-2.0)  # softplus(-2) ≈ 0.13
            
            # CONSTANT-Q SIGMA: σ inversely proportional to frequency
            N_CYCLES = 3.5
            constant_q_sigma = N_CYCLES / omegas.clamp(min=50.0)
            constant_q_sigma = constant_q_sigma.clamp(min=0.002, max=0.050)
            self._sigma_logit.data = torch.log(constant_q_sigma)
            
            # Gamma (chirp): zero initially (no frequency modulation)
            self._gamma.data.zero_()
            
            print(f"[AudioGSModel] Initialized {num_atoms} atoms (Constant-Q sigma).")
    
    def _f0_guided_init(self, waveform_np, num_atoms: int, hop_length: int = 512):
        """
        F0-guided frequency initialization using librosa.pyin.
        
        REFACTOR: Harmonics now extend to 90% Nyquist (UNCAPPED).
        
        Returns:
            taus: Time positions for each atom (voiced-weighted)
            omegas: Frequency values in Hz for each atom
            phis: Phase values for each atom
        """
        import librosa
        from scipy.interpolate import interp1d
        import numpy as np
        
        # Extract F0 using pyin (returns voiced_prob for weighting)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            waveform_np,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=self.sample_rate,
            frame_length=2048,
            hop_length=hop_length,
        )
        
        # Time axis for F0 frames
        f0_times = librosa.times_like(f0, sr=self.sample_rate, hop_length=hop_length)
        
        # Handle NaN values for interpolation
        f0_valid = np.where(np.isnan(f0), 0, f0)
        voiced_mask = ~np.isnan(f0)
        
        # CONFIDENCE CHECK: If voiced ratio is too low, fallback to uniform tau + STFT
        voiced_ratio = np.mean(voiced_mask)
        if voiced_ratio < 0.3:
            print(f"[F0 Init] Low voiced ratio ({voiced_ratio:.1%}), falling back to STFT init")
            taus = torch.rand(num_atoms, device=self.device) * self.audio_duration
            omegas, phis = self._stft_init(
                torch.from_numpy(waveform_np).float().to(self.device),
                taus, n_fft=2048, hop_length=hop_length
            )
            return taus, omegas, phis
        
        # =====================================================
        # VOICED-WEIGHTED TAU SAMPLING
        # Dense in voiced regions, sparse in silence
        # =====================================================
        voiced_weights = np.where(np.isnan(voiced_prob), 0.05, voiced_prob)
        voiced_weights = voiced_weights / (voiced_weights.sum() + 1e-8)
        
        # Sample frame indices weighted by voicing probability
        frame_indices = np.random.choice(
            len(f0_times), size=num_atoms, replace=True, p=voiced_weights
        )
        # Add small jitter within each frame's hop window
        jitter = np.random.uniform(-0.5, 0.5, size=num_atoms) * (hop_length / self.sample_rate)
        taus_np = f0_times[frame_indices] + jitter
        taus_np = np.clip(taus_np, 0, self.audio_duration - 1e-6)
        taus = torch.from_numpy(taus_np).float().to(self.device)
        
        # Create interpolators for F0 and voiced mask
        f0_interp = interp1d(
            f0_times, f0_valid,
            kind='linear', bounds_error=False, fill_value=0
        )
        f0_at_taus = f0_interp(taus_np)
        
        voiced_interp = interp1d(
            f0_times, voiced_mask.astype(float),
            kind='nearest', bounds_error=False, fill_value=0
        )
        is_voiced = voiced_interp(taus_np) > 0.5
        
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
                # Unvoiced region: mid-range frequency (fricatives)
                omegas[idx] = np.random.uniform(2000, 6000)
        
        # Harmonic atoms (40%): UNCAPPED harmonics up to 90% Nyquist
        for i, idx in enumerate(harmonic_indices):
            if is_voiced[idx] and f0_at_taus[idx] > 50:
                f0_hz = f0_at_taus[idx]
                # REFACTOR: Dynamic harmonics up to 90% Nyquist (UNCAPPED)
                max_harmonic = int(0.9 * self.nyquist_freq / f0_hz)
                if max_harmonic >= 2:
                    # Weighted selection: prefer lower harmonics (more energy in speech)
                    h_weights = np.array([1.0/h for h in range(2, max_harmonic + 1)])
                    h_weights = h_weights / h_weights.sum()
                    h = np.random.choice(range(2, max_harmonic + 1), p=h_weights)
                    omegas[idx] = f0_hz * h
                else:
                    omegas[idx] = f0_hz  # F0 too high, use fundamental
            else:
                # Unvoiced: mid-range
                omegas[idx] = np.random.uniform(2000, 6000)
        
        # Random atoms (20%): full spectrum coverage
        for idx in random_indices:
            omegas[idx] = np.random.uniform(100, 0.9 * self.nyquist_freq)
        
        # =====================================================
        # Phase Initialization from STFT (Bilinear Interpolation)
        # =====================================================
        waveform_tensor = torch.from_numpy(waveform_np).float()
        stft = torch.stft(
            waveform_tensor, 
            n_fft=2048, 
            hop_length=hop_length, 
            return_complex=True,
            window=torch.hann_window(2048),
            center=True,
        )
        num_freq_bins, num_frames = stft.shape
        
        # Compute continuous indices for each atom's (tau, omega)
        frame_indices_f = (taus_np * self.sample_rate / hop_length).astype(np.float32)
        frame_indices_f = np.clip(frame_indices_f, 0, num_frames - 1 - 1e-6)
        
        freq_resolution = self.sample_rate / 2048
        freq_indices = (omegas / freq_resolution).astype(np.float32)
        freq_indices = np.clip(freq_indices, 0, num_freq_bins - 1 - 1e-6)
        
        phis = np.zeros(num_atoms)
        stft_np = stft.numpy()
        
        for i in range(num_atoms):
            f_idx = freq_indices[i]
            t_idx = frame_indices_f[i]
            
            fb0, fb1 = int(np.floor(f_idx)), int(np.ceil(f_idx))
            tb0, tb1 = int(np.floor(t_idx)), int(np.ceil(t_idx))
            
            fb1 = min(fb1, num_freq_bins - 1)
            tb1 = min(tb1, num_frames - 1)
            
            wf = f_idx - fb0
            wt = t_idx - tb0
            
            c00 = stft_np[fb0, tb0]
            c01 = stft_np[fb0, tb1]
            c10 = stft_np[fb1, tb0]
            c11 = stft_np[fb1, tb1]
            
            c_interp = (
                (1 - wf) * (1 - wt) * c00 +
                (1 - wf) * wt * c01 +
                wf * (1 - wt) * c10 +
                wf * wt * c11
            )
            
            phis[i] = np.angle(c_interp)
        
        omegas_tensor = torch.from_numpy(omegas).float().to(self.device)
        phis_tensor = torch.from_numpy(phis).float().to(self.device)
        
        print(f"[F0 Init] Voiced-weighted τ sampling, dynamic harmonics up to 90% Nyquist")
        print(f"[F0 Init] Voiced atoms: {np.sum(is_voiced)}/{num_atoms}, Phase: STFT bilinear interpolation")
        
        return taus, omegas_tensor, phis_tensor
    
    def _stft_peak_init(self, waveform: torch.Tensor, taus: torch.Tensor, n_fft: int = 2048, hop_length: int = 512):
        """
        Robust STFT Peak-Picking Initialization (v2.0).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        stft = torch.stft(
            waveform.squeeze(0), n_fft=n_fft, hop_length=hop_length,
            return_complex=True, center=True,
            window=torch.hann_window(n_fft).to(self.device)
        )
        mag = stft.abs()
        phase = stft.angle()
        
        num_freq_bins, num_frames = mag.shape
        freqs = torch.linspace(0, self.nyquist_freq, num_freq_bins, device=self.device)
        
        num_atoms = len(taus)
        frame_times = torch.arange(num_frames, device=self.device) * hop_length / self.sample_rate
        
        omegas = torch.zeros(num_atoms, device=self.device)
        phis = torch.zeros(num_atoms, device=self.device)
        
        peaks_per_frame = min(10, num_freq_bins // 2)
        
        for i, tau in enumerate(taus):
            frame_idx = torch.argmin(torch.abs(frame_times - tau)).item()
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            
            frame_mag = mag[:, frame_idx]
            top_k_indices = torch.topk(frame_mag, k=peaks_per_frame).indices
            selected_idx = top_k_indices[torch.randint(0, len(top_k_indices), (1,)).item()]
            
            omegas[i] = freqs[selected_idx]
            phis[i] = phase[selected_idx, frame_idx]
        
        print(f"[STFT Peak Init] Initialized {num_atoms} atoms from spectral peaks")
        return omegas, phis
    
    def _stft_init(self, waveform: torch.Tensor, taus: torch.Tensor, n_fft: int, hop_length: int):
        """
        STFT-based initialization with DE-CLUMPING via temperature scaling.
        """
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
        
        num_atoms = len(taus)
        omegas = torch.zeros(num_atoms, device=self.device)
        phis = torch.zeros(num_atoms, device=self.device)
        
        for i, t in enumerate(taus):
            frame_idx = int(t / self.audio_duration * (num_frames - 1))
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            
            frame_mag = mag[:, frame_idx]
            if frame_mag.sum() > 0:
                probs = torch.sqrt(frame_mag + 1e-10)
                probs = probs / probs.sum()
                freq_idx = torch.multinomial(probs, 1).item()
            else:
                freq_idx = torch.randint(0, num_freq_bins, (1,)).item()
            
            omegas[i] = freqs[freq_idx]
            phis[i] = phase[freq_idx, frame_idx]
        
        return omegas, phis

    # --- Properties and Helpers ---
    @property
    def num_atoms(self) -> int: 
        return self._amplitude_logit.shape[0]

    @property
    def amplitude(self) -> torch.Tensor: 
        return torch.nn.functional.softplus(self._amplitude_logit)

    @property
    def tau(self) -> torch.Tensor: 
        return self._tau

    @property
    def omega(self) -> torch.Tensor: 
        return torch.sigmoid(self._omega_logit) * self.nyquist_freq

    @property
    def sigma(self) -> torch.Tensor: 
        # Use softplus for smooth positive sigma values
        # Minimum 0.5ms (~12 samples at 24kHz) to prevent gradient explosion
        return torch.nn.functional.softplus(self._sigma_logit) + 0.0005

    @property
    def phi(self) -> torch.Tensor: 
        """Phase angle from atan2 of the vectorized representation."""
        return torch.atan2(self._phi_vector[:, 1], self._phi_vector[:, 0])

    @property
    def phase_vector(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return RAW (cos, sin) vectors for loss regularization.
        
        CRITICAL: Must return UN-NORMALIZED components!
        If we returned normalized vectors (magnitude=1), the regularization term
        ((radius^2 - 1)^2) would always be 0, making the loss ineffective.
        
        By returning raw values, the loss can penalize vectors that drift away
        from the unit circle, keeping training stable.
        """
        return self._phi_vector[:, 0], self._phi_vector[:, 1]

    @property
    def gamma(self) -> torch.Tensor: 
        return self._gamma

    def get_all_params(self):
        return (self.amplitude, self.tau, self.omega, self.sigma, self.phi, self.gamma)

    def get_optimizer_param_groups(self, config: dict):
        return [
            {"params": [self._amplitude_logit], "lr": config.get("lr_amplitude", 0.01)},
            {"params": [self._tau], "lr": config.get("lr_position", 0.001)},
            {"params": [self._omega_logit], "lr": config.get("lr_frequency", 0.005)},
            {"params": [self._sigma_logit], "lr": config.get("lr_sigma", 0.001)},
            {"params": [self._phi_vector], "lr": config.get("lr_phase", 0.01)},  # REFACTORED
            {"params": [self._gamma], "lr": config.get("lr_chirp", 0.001)},
        ]

    # --- Density Control Helpers ---
    def accumulate_gradients(self):
        """
        REFACTOR: Accumulate weighted gradients from all learnable parameters.
        This provides a more holistic view of atom "activity" for density control.
        """
        grad_sum = torch.zeros(self.num_atoms, device=self.device)
        count = 0
        
        for param, weight in [
            (self._tau, 1.0),
            (self._omega_logit, 0.5),
            (self._amplitude_logit, 0.3),
            (self._phi_vector, 0.2),
        ]:
            if param.grad is not None:
                if param.dim() == 1:
                    grad_sum += weight * param.grad.abs()
                else:  # Handle phi_vector [N, 2]
                    grad_sum += weight * param.grad.abs().sum(dim=-1)
                count += 1
        
        if count > 0:
            self.tau_grad_accum += grad_sum
            self.grad_accum_count += 1
            
    def get_average_gradients(self) -> torch.Tensor:
        if self.grad_accum_count > 0: 
            return self.tau_grad_accum / self.grad_accum_count
        return self.tau_grad_accum
        
    def reset_gradient_accumulators(self):
        self.tau_grad_accum.zero_()
        self.grad_accum_count.zero_()

    def add_atoms(self, amplitude_logit, tau, omega_logit, sigma_logit, phi_vector, gamma):
        """Add new atoms with 2D phi_vector."""
        with torch.no_grad():
            self._amplitude_logit = nn.Parameter(torch.cat([self._amplitude_logit.data, amplitude_logit], 0))
            self._tau = nn.Parameter(torch.cat([self._tau.data, tau], 0))
            self._omega_logit = nn.Parameter(torch.cat([self._omega_logit.data, omega_logit], 0))
            self._sigma_logit = nn.Parameter(torch.cat([self._sigma_logit.data, sigma_logit], 0))
            self._phi_vector = nn.Parameter(torch.cat([self._phi_vector.data, phi_vector], 0))
            self._gamma = nn.Parameter(torch.cat([self._gamma.data, gamma], 0))
            self.tau_grad_accum = torch.cat([self.tau_grad_accum, torch.zeros(amplitude_logit.shape[0], device=self.device)], 0)
    
    def remove_atoms(self, mask) -> torch.Tensor:
        """
        Remove atoms and return the indices that were kept (for optimizer mapping).
        
        REFACTOR: Returns keep_indices for proper optimizer state copying.
        """
        keep_indices = torch.where(mask)[0]
        with torch.no_grad():
            self._amplitude_logit = nn.Parameter(self._amplitude_logit.data[mask])
            self._tau = nn.Parameter(self._tau.data[mask])
            self._omega_logit = nn.Parameter(self._omega_logit.data[mask])
            self._sigma_logit = nn.Parameter(self._sigma_logit.data[mask])
            self._phi_vector = nn.Parameter(self._phi_vector.data[mask])
            self._gamma = nn.Parameter(self._gamma.data[mask])
            self.tau_grad_accum = self.tau_grad_accum[mask]
        return keep_indices

    def clone_atoms_by_indices(self, indices):
        """
        Clone atoms using JITTERED HARMONIC STACKING for additive synthesis.
        
        REFACTOR: 
        - Uses real sigma / 2 -> logit conversion instead of logit - 0.693
        - No high-freq capping (let density_control handle it)
        """
        if len(indices) == 0: 
            return 0
        
        with torch.no_grad():
            # Get parent frequencies (in Hz)
            parent_omega_hz = self.omega[indices]
            
            # Compute target harmonic with JITTER for inharmonicity handling
            jitter = torch.empty_like(parent_omega_hz).uniform_(0.95, 1.05)
            target_omega_hz = parent_omega_hz * 2.0 * jitter
            
            # Safety clamp to 90% Nyquist
            max_omega = self.nyquist_freq * 0.90
            target_omega_hz = torch.clamp(target_omega_hz, min=50.0, max=max_omega)
            
            # Convert target omega to logit space
            omega_normalized = (target_omega_hz / self.nyquist_freq).clamp(1e-5, 1 - 1e-5)
            new_omega_logit = torch.log(omega_normalized / (1 - omega_normalized))
            
            # Clone other parameters
            new_amp = self._amplitude_logit.data[indices].clone() - 0.5
            new_tau = self._tau.data[indices].clone()
            
            # Add tau jitter to break positional clustering
            tau_jitter = torch.randn_like(new_tau) * 0.005  # ±5ms random displacement
            new_tau = (new_tau + tau_jitter).clamp(0, self.audio_duration - 1e-6)
            
            # REFACTOR: CONSTANT-Q sigma using real values
            # Halve real sigma → then convert back to logit
            parent_sigma_real = self.sigma[indices]
            new_sigma_real = parent_sigma_real / 2.0
            new_sigma_real = new_sigma_real.clamp(min=0.0005)  # Min 0.5ms
            new_sigma = torch.log(new_sigma_real)
            
            # Clone phi_vector (2D)
            new_phi_vector = self._phi_vector.data[indices].clone()
            new_gamma = self._gamma.data[indices].clone()
            
        self.add_atoms(new_amp, new_tau, new_omega_logit, new_sigma, new_phi_vector, new_gamma)
        return len(indices)

    def split_atoms_by_indices(self, indices, scale_factor=1.6):
        """Split atoms into two with smaller sigma and displaced tau."""
        if len(indices) == 0: 
            return 0
        with torch.no_grad():
            sigma = self.sigma[indices]
            tau = self._tau.data[indices]
            offset = sigma * 0.5
            
            tau_jitter1 = torch.randn_like(tau) * 0.003
            tau_jitter2 = torch.randn_like(tau) * 0.003
            
            # Atom 1
            new_amp1 = self._amplitude_logit.data[indices].clone() - 0.3
            new_tau1 = (tau - offset + tau_jitter1).clamp(0, self.audio_duration - 1e-6)
            new_omega1 = self._omega_logit.data[indices].clone() + torch.randn_like(self._omega_logit.data[indices]) * 0.05
            new_sigma1 = self._sigma_logit.data[indices].clone() - math.log(scale_factor)
            new_phi_vector1 = self._phi_vector.data[indices].clone()
            new_gamma1 = self._gamma.data[indices].clone()
            
            # Atom 2 (phase shifted by π/2)
            new_amp2 = self._amplitude_logit.data[indices].clone() - 0.3
            new_tau2 = (tau + offset + tau_jitter2).clamp(0, self.audio_duration - 1e-6)
            new_omega2 = self._omega_logit.data[indices].clone() - torch.randn_like(self._omega_logit.data[indices]) * 0.05
            new_sigma2 = new_sigma1.clone()
            # Rotate phase by π/2: (cos, sin) -> (-sin, cos)
            new_phi_vector2 = self._phi_vector.data[indices].clone()
            cos_orig = new_phi_vector2[:, 0].clone()
            sin_orig = new_phi_vector2[:, 1].clone()
            new_phi_vector2[:, 0] = -sin_orig
            new_phi_vector2[:, 1] = cos_orig
            new_gamma2 = self._gamma.data[indices].clone()
            
        self.add_atoms(new_amp1, new_tau1, new_omega1, new_sigma1, new_phi_vector1, new_gamma1)
        self.add_atoms(new_amp2, new_tau2, new_omega2, new_sigma2, new_phi_vector2, new_gamma2)
        return len(indices) * 2

    def state_dict_full(self):
        return {
            "amplitude_logit": self._amplitude_logit.data,
            "tau": self._tau.data,
            "omega_logit": self._omega_logit.data,
            "sigma_logit": self._sigma_logit.data,
            "phi_vector": self._phi_vector.data,  # REFACTORED
            "gamma": self._gamma.data,
            "tau_grad_accum": self.tau_grad_accum,
            "grad_accum_count": self.grad_accum_count,
        }
    
    def load_state_dict_full(self, state):
        self._amplitude_logit = nn.Parameter(state["amplitude_logit"])
        self._tau = nn.Parameter(state["tau"])
        self._omega_logit = nn.Parameter(state["omega_logit"])
        self._sigma_logit = nn.Parameter(state["sigma_logit"])
        # Handle backward compatibility
        if "phi_vector" in state:
            self._phi_vector = nn.Parameter(state["phi_vector"])
        elif "phi" in state:
            # Convert old scalar phi to vectorized
            old_phi = state["phi"]
            self._phi_vector = nn.Parameter(torch.stack([
                torch.cos(old_phi), torch.sin(old_phi)
            ], dim=-1))
        self._gamma = nn.Parameter(state["gamma"])
        self.tau_grad_accum = state["tau_grad_accum"]
        self.grad_accum_count = state["grad_accum_count"]