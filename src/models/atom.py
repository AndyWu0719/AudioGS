"""
AudioGS Model: Learnable Gabor Atoms for Audio Representation.
Major Refactor: Vectorized Phase, Uncapped Harmonics, Weighted Gradient Tracking.

Code Maintenance Update:
- Fixed sigma offset to 1ms to match CUDA kernel clamp
- Added correct inverse transforms for densification
- Wired config fields to initialization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional

from src.utils.config import InitConfig, SIGMA_OFFSET as CONFIG_SIGMA_OFFSET

# =============================================================================
# Constants for parameterization
# =============================================================================

SIGMA_OFFSET = CONFIG_SIGMA_OFFSET  # 1ms minimum sigma (matches CUDA kernel clamp)
SPLIT_AMP_SCALE = 1.0 / math.sqrt(2.0)
CLONE_AMP_SCALE = 0.5


# =============================================================================
# Inverse Transform Helpers for Densification
# =============================================================================
def _inv_softplus(y: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Numerically stable inverse softplus: log(exp(y) - 1).
    
    For large y, inv_softplus(y) ≈ y (avoiding overflow).
    For small y, uses the exact formula.
    """
    threshold = 20.0  # Above this, exp(y) overflows; use approximation
    return torch.where(
        y > threshold,
        y,
        torch.log(torch.expm1(y * beta).clamp(min=1e-8)) / beta
    )


def _sigma_real_to_logit(sigma_real: torch.Tensor) -> torch.Tensor:
    """
    Convert real sigma (seconds) to _sigma_logit.
    
    Inverse of: sigma_real = softplus(_sigma_logit) + SIGMA_OFFSET
    """
    return _inv_softplus((sigma_real - SIGMA_OFFSET).clamp(min=1e-8))


def _omega_real_to_logit(omega_real: torch.Tensor, nyquist: float) -> torch.Tensor:
    """
    Convert real omega (Hz) to _omega_logit.
    
    Inverse of: omega_real = sigmoid(_omega_logit) * 0.99 * nyquist
    Uses 0.99 * nyquist scaling to match the forward property.
    """
    max_omega = 0.99 * nyquist
    normalized = (omega_real / max_omega).clamp(1e-5, 1 - 1e-5)
    return torch.log(normalized / (1 - normalized))  # logit


def _constant_q_sigma(omega_hz: torch.Tensor, init_cfg: InitConfig) -> torch.Tensor:
    """
    Constant-Q sigma from frequency.

    If constant_q_use_2pi is enabled, treat constant_q_cycles as Q (f / BW).
    Otherwise treat it as a raw cycles-per-sigma factor.
    """
    denom = omega_hz
    if init_cfg.constant_q_use_2pi:
        denom = 2.0 * math.pi * omega_hz
    return init_cfg.constant_q_cycles / (denom + 1e-8)

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

    def init_random(self, init_config: Optional[dict] = None):
        """
        Fallback: Random initialization with CONSTANT-Q SIGMA.
        
        AUDIO PHYSICS: Sigma must scale with frequency. A 100Hz wave has
        period 10ms, so sigma must be ≥10ms to represent it without gaps.
        
        Issue 2 Fix: Uses correct inv_softplus transform and wires config values.
        
        Args:
            init_config: Optional dict with 'constant_q_cycles', 'sigma_min', 'sigma_max'
        """
        # Get config values or defaults
        init_cfg = InitConfig.from_dict(init_config)
        SIGMA_MIN = init_cfg.sigma_min
        SIGMA_MAX = init_cfg.sigma_max
        
        with torch.no_grad():
            self._amplitude_logit.data.normal_(0, 0.1)
            self._tau.data.uniform_(0, self.audio_duration)
            # Init omega uniformly across full range to avoid "dead bands"
            self._omega_logit.data.uniform_(-5.0, 5.0)
            
            # CONSTANT-Q SIGMA: σ = cycles / (2π f) if enabled, else cycles / f
            # This ensures atoms can properly represent their frequency
            omega_hz = torch.sigmoid(self._omega_logit.data) * 0.99 * self.nyquist_freq
            omega_hz = omega_hz.clamp(min=50.0)  # Min 50Hz for numerical stability
            constant_q_sigma = _constant_q_sigma(omega_hz, init_cfg)
            constant_q_sigma = constant_q_sigma.clamp(min=SIGMA_MIN, max=SIGMA_MAX)
            
            # Issue 2 Fix: Use correct inverse transform (not torch.log)
            self._sigma_logit.data = _sigma_real_to_logit(constant_q_sigma)
            
            # REFACTOR: Random phase vectors (unit circle)
            random_angles = torch.rand(self.num_atoms, device=self.device) * 2 * math.pi
            self._phi_vector.data[:, 0] = torch.cos(random_angles)
            self._phi_vector.data[:, 1] = torch.sin(random_angles)
            
    def initialize_from_audio(
        self, 
        waveform: torch.Tensor, 
        use_f0_init: bool = True, 
        n_fft: int = 2048, 
        hop_length: int = 512,
        init_config: Optional[dict] = None
    ):
        """
        Initialize atom parameters with F0-guided or STFT-based strategy.
        
        F0-Guided Strategy:
        - τ: Voiced-weighted sampling (dense in voiced regions, sparse in silence)
        - 40% atoms: Initialize ω at F0 (fundamental frequency)
        - 40% atoms: Initialize ω at harmonics up to 90% Nyquist (UNCAPPED)
        - 20% atoms: Random frequency (for unvoiced sounds)
        
        SHORT AUDIO HANDLING (<2s):
        - F0 detection is unreliable for short audio
        - Use uniform tau + STFT-based frequency instead
        
        Args:
            waveform: Audio waveform tensor
            use_f0_init: If True, use F0-guided initialization
            n_fft: FFT size for STFT fallback
            hop_length: Hop length for STFT (PARAMETERIZED, not hardcoded)
        """
        print(f"[AudioGSModel] Initializing from audio (use_f0_init={use_f0_init})...")
        
        # Read config or use defaults (Issue D: wire config fields)
        init_cfg = InitConfig.from_dict(init_config)
        SIGMA_MIN = init_cfg.sigma_min
        SIGMA_MAX = init_cfg.sigma_max
        
        # Random init first to ensure no atoms are stuck at default zeros (0.5 Nyquist)
        self.init_random(init_config)
        
        with torch.no_grad():
            waveform = waveform.to(self.device)
            if waveform.dim() == 2:
                waveform = waveform.squeeze(0)
            
            waveform_np = waveform.cpu().numpy()
            num_atoms = self.num_atoms
            
            # SHORT AUDIO HANDLING: <2s audio has unreliable F0 detection
            # Use uniform tau + STFT instead of F0-guided
            if self.audio_duration < 2.0:
                print(f"[AudioGSModel] Short audio ({self.audio_duration:.2f}s < 2s): using uniform tau + STFT init")
                # Uniform time distribution gives better coverage for short audio
                taus = torch.linspace(0.01, self.audio_duration - 0.01, num_atoms, device=self.device)
                # Add small jitter to avoid perfect grid artifacts
                taus = taus + (torch.rand(num_atoms, device=self.device) - 0.5) * 0.02
                taus = taus.clamp(0, self.audio_duration)
                omegas, phis = self._stft_init(waveform, taus, n_fft, hop_length)
            elif use_f0_init:
                # F0-guided init returns voiced-weighted taus
                taus, omegas, phis = self._f0_guided_init(waveform_np, num_atoms, hop_length)
            else:
                # Fallback: Random taus + STFT-based frequencies
                taus = torch.rand(num_atoms, device=self.device) * self.audio_duration
                omegas, phis = self._stft_init(waveform, taus, n_fft, hop_length)
            
            # Assign to parameters
            self._tau.data = taus
            
            # Convert omega (Hz) to logit (using 99% Nyquist scaling to match property)
            max_omega = 0.99 * self.nyquist_freq
            omega_normalized = (omegas / max_omega).clamp(1e-5, 1 - 1e-5)
            self._omega_logit.data = torch.log(omega_normalized / (1 - omega_normalized))
            
            # REFACTOR: Convert scalar phase to vectorized phase
            self._phi_vector.data[:, 0] = torch.cos(phis)
            self._phi_vector.data[:, 1] = torch.sin(phis)
            
            # Amplitude: small initial values
            self._amplitude_logit.data.fill_(-2.0)  # softplus(-2) ≈ 0.13
            
            # CONSTANT-Q SIGMA: σ inversely proportional to frequency
            # Uses config values (Issue D: wired from YAML)
            constant_q_sigma = _constant_q_sigma(omegas.clamp(min=50.0), init_cfg)
            constant_q_sigma = constant_q_sigma.clamp(min=SIGMA_MIN, max=SIGMA_MAX)
            # Use correct inverse transform (Issue B)
            self._sigma_logit.data = _sigma_real_to_logit(constant_q_sigma)
            
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
        # HYBRID TAU SAMPLING: 50% Uniform + 50% F0-Weighted
        # =====================================================
        # Uniform ensures coverage of unvoiced regions (consonants)
        # F0-weighted ensures dense coverage of voiced regions (vowels)
        print(f"[F0 Init] Hybrid τ: 50% uniform + 50% voiced-weighted")
        
        num_uniform = num_atoms // 2
        num_f0_weighted = num_atoms - num_uniform
        
        # Part 1: Uniform distribution (with small jitter)
        uniform_taus = torch.linspace(0.01, self.audio_duration - 0.01, num_uniform, device=self.device)
        uniform_taus = uniform_taus + (torch.rand(num_uniform, device=self.device) - 0.5) * 0.02
        
        # Part 2: F0-weighted distribution
        voiced_weights = np.where(np.isnan(voiced_prob), 0.05, voiced_prob)
        voiced_weights = voiced_weights / (voiced_weights.sum() + 1e-8)
        
        frame_indices = np.random.choice(
            len(f0_times), size=num_f0_weighted, replace=True, p=voiced_weights
        )
        jitter = np.random.uniform(-0.5, 0.5, size=num_f0_weighted) * (hop_length / self.sample_rate)
        f0_taus_np = f0_times[frame_indices] + jitter
        f0_taus_np = np.clip(f0_taus_np, 0, self.audio_duration - 1e-6)
        f0_taus = torch.from_numpy(f0_taus_np).float().to(self.device)
        
        # Combine and shuffle
        taus = torch.cat([uniform_taus, f0_taus])
        shuffle_idx = torch.randperm(num_atoms, device=self.device)
        taus = taus[shuffle_idx]
        taus_np = taus.cpu().numpy()
        
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
        
        # Harmonic atoms (40%): harmonics up to 80% Nyquist (to prevent HF band)
        for i, idx in enumerate(harmonic_indices):
            if is_voiced[idx] and f0_at_taus[idx] > 50:
                f0_hz = f0_at_taus[idx]
                # Limit to 99% Nyquist
                max_harmonic = int(0.99 * self.nyquist_freq / f0_hz)
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
        
        # Random atoms (20%): spectrum coverage up to 99% Nyquist
        for idx in random_indices:
            omegas[idx] = np.random.uniform(100, 0.99 * self.nyquist_freq)
        
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

    def add_atoms_from_residual(
        self,
        residual: torch.Tensor,
        num_atoms: int,
        init_config: Optional[dict] = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        amp_scale: float = 0.3,
        peaks_per_frame: int = 4,
        min_peak_ratio: float = 0.25,
        time_jitter_ratio: float = 0.5,
        freq_jitter_bins: float = 0.5,
        selection_strategy: str = "stft_peak",
        sigma_multiplier: float = 5.0,
        mp_amp_max: Optional[float] = None,
        mp_normalize: bool = True,
        mp_score_min: float = 0.0,
    ) -> int:
        """
        Add new atoms from residual STFT peaks (residual-guided densification).
        Optionally use MP-style selection via Gabor inner products.
        """
        if num_atoms <= 0:
            return 0
        init_cfg = InitConfig.from_dict(init_config)
        with torch.no_grad():
            if residual.dim() == 2:
                residual = residual.squeeze(0)
            residual = residual.to(self.device)

            window = torch.hann_window(n_fft, device=self.device)
            stft = torch.stft(
                residual,
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
                window=window,
                center=True,
            )
            mag = stft.abs()
            phase = stft.angle()

            num_freq_bins, num_frames = mag.shape
            if num_freq_bins == 0 or num_frames == 0:
                return 0

            frame_energy = mag.mean(dim=0)
            if frame_energy.max().item() <= 0:
                return 0

            peaks_per_frame = max(1, min(peaks_per_frame, num_freq_bins))
            frames_to_sample = min(num_frames, max(1, int(math.ceil(num_atoms / peaks_per_frame))))

            probs = frame_energy / (frame_energy.sum() + 1e-8)
            replacement = frames_to_sample > num_frames
            frame_indices = torch.multinomial(probs, frames_to_sample, replacement=replacement)

            cand_freq = []
            cand_time = []
            cand_mag = []
            cand_frame_max = []
            for frame_idx in frame_indices.tolist():
                frame_mag = mag[:, frame_idx]
                if num_freq_bins > 1:
                    frame_mag = frame_mag.clone()
                    frame_mag[0] = 0.0
                frame_max = frame_mag.max()
                if frame_max.item() <= 0:
                    continue
                topk = torch.topk(frame_mag, k=peaks_per_frame, largest=True)
                for f_idx, m_val in zip(topk.indices.tolist(), topk.values.tolist()):
                    if m_val < frame_max.item() * min_peak_ratio:
                        continue
                    cand_freq.append(f_idx)
                    cand_time.append(frame_idx)
                    cand_mag.append(m_val)
                    cand_frame_max.append(frame_max.item())

            if len(cand_mag) == 0:
                return 0

            selection = selection_strategy.lower()
            cand_freq_t = torch.tensor(cand_freq, device=self.device, dtype=torch.long)
            cand_time_t = torch.tensor(cand_time, device=self.device, dtype=torch.long)
            cand_mag_t = torch.tensor(cand_mag, device=self.device)
            cand_frame_max_t = torch.tensor(cand_frame_max, device=self.device)

            tau_cand = cand_time_t.float() * hop_length / self.sample_rate
            omega_cand = cand_freq_t.float() * (self.sample_rate / n_fft)
            if time_jitter_ratio > 0.0:
                tau_cand = tau_cand + (torch.rand_like(tau_cand) - 0.5) * (
                    hop_length / self.sample_rate
                ) * time_jitter_ratio
            if freq_jitter_bins > 0.0:
                omega_cand = omega_cand + (torch.rand_like(omega_cand) - 0.5) * (
                    self.sample_rate / n_fft
                ) * freq_jitter_bins
            tau_cand = tau_cand.clamp(0.0, self.audio_duration - 1e-6)
            omega_cand = omega_cand.clamp(min=50.0, max=0.99 * self.nyquist_freq)
            sigma_cand = _constant_q_sigma(omega_cand, init_cfg).clamp(
                min=init_cfg.sigma_min,
                max=init_cfg.sigma_max,
            )

            def _compute_inner(residual_local, tau_i, omega_i, sigma_i):
                num_samples = residual_local.shape[-1]
                window_bound = float((sigma_i * sigma_multiplier).item())
                if window_bound <= 0:
                    return None
                tau_scalar = float(tau_i.item())
                window_start = max(0, int((tau_scalar - window_bound) * self.sample_rate))
                window_end = min(num_samples - 1, int((tau_scalar + window_bound) * self.sample_rate))
                if window_start > window_end:
                    return None

                t = torch.arange(window_start, window_end + 1, device=self.device, dtype=residual_local.dtype)
                t = t / self.sample_rate - tau_i
                t_sq = t * t

                envelope = torch.exp(-t_sq / (2.0 * sigma_i * sigma_i))
                normalized_dist = torch.abs(t) / (window_bound + 1e-8)
                window_factor = torch.ones_like(normalized_dist)
                outer_mask = normalized_dist > 0.8
                if outer_mask.any():
                    edge_t = (normalized_dist[outer_mask] - 0.8) / 0.2
                    window_factor[outer_mask] = 0.5 * (1.0 + torch.cos(math.pi * edge_t))

                g = envelope * window_factor
                phase_arg = 2.0 * math.pi * omega_i * t
                cos_term = torch.cos(phase_arg)
                sin_term = torch.sin(phase_arg)

                r_seg = residual_local[window_start:window_end + 1]
                u = g * cos_term
                v = g * sin_term

                a = torch.dot(r_seg, u)
                b = torch.dot(r_seg, v)
                uu = torch.dot(u, u)
                vv = torch.dot(v, v)
                denom = 0.5 * (uu + vv)
                if denom <= 0:
                    return None

                ip_mag = torch.sqrt(a * a + b * b)
                score = ip_mag / (torch.sqrt(denom) + 1e-8) if mp_normalize else ip_mag
                A = ip_mag / (denom + 1e-8)
                phi_i = torch.atan2(-b, a)
                return score, A, phi_i, window_start, window_end, u, v

            if selection in ("mp_iterative", "mp_explicit"):
                residual_work = residual.clone()
                used = torch.zeros(tau_cand.shape[0], device=self.device, dtype=torch.bool)
                amp_list = []
                phi_list = []
                tau_list = []
                omega_list = []
                sigma_list = []

                max_steps = min(num_atoms, tau_cand.shape[0])
                for _ in range(max_steps):
                    best_idx = None
                    best_score = None
                    best_A = None
                    best_phi = None
                    best_start = None
                    best_end = None
                    best_u = None
                    best_v = None

                    for i in range(tau_cand.shape[0]):
                        if used[i]:
                            continue
                        result = _compute_inner(residual_work, tau_cand[i], omega_cand[i], sigma_cand[i])
                        if result is None:
                            continue
                        score, A, phi_i, window_start, window_end, u, v = result
                        if score < mp_score_min:
                            continue
                        if best_score is None or score > best_score:
                            best_score = score
                            best_idx = i
                            best_A = A
                            best_phi = phi_i
                            best_start = window_start
                            best_end = window_end
                            best_u = u
                            best_v = v

                    if best_idx is None:
                        break

                    if mp_amp_max is not None:
                        best_A = best_A.clamp(max=mp_amp_max)

                    amp_list.append(best_A)
                    phi_list.append(best_phi)
                    tau_list.append(tau_cand[best_idx])
                    omega_list.append(omega_cand[best_idx])
                    sigma_list.append(sigma_cand[best_idx])

                    cos_phi = torch.cos(best_phi)
                    sin_phi = torch.sin(best_phi)
                    atom = best_A * (cos_phi * best_u - sin_phi * best_v)
                    residual_work[best_start:best_end + 1] -= atom

                    used[best_idx] = True

                if len(amp_list) == 0:
                    return 0

                tau = torch.stack(tau_list)
                omega = torch.stack(omega_list)
                sigma = torch.stack(sigma_list)
                phi = torch.stack(phi_list)
                amp = torch.stack(amp_list) * amp_scale
                if mp_amp_max is not None:
                    amp = amp.clamp(max=mp_amp_max)
            elif selection in ("mp", "matching_pursuit"):
                scores = []
                amps = []
                phis = []
                taus = []
                omegas = []
                sigmas = []

                for i in range(cand_mag_t.numel()):
                    result = _compute_inner(residual, tau_cand[i], omega_cand[i], sigma_cand[i])
                    if result is None:
                        continue
                    score, A, phi_i, _, _, _, _ = result
                    if score < mp_score_min:
                        continue
                    scores.append(score)
                    amps.append(A)
                    phis.append(phi_i)
                    taus.append(tau_cand[i])
                    omegas.append(omega_cand[i])
                    sigmas.append(sigma_cand[i])

                if len(scores) == 0:
                    return 0

                scores_t = torch.stack(scores)
                k = min(num_atoms, scores_t.numel())
                sel = torch.topk(scores_t, k=k, largest=True).indices

                tau = torch.stack(taus)[sel]
                omega = torch.stack(omegas)[sel]
                sigma = torch.stack(sigmas)[sel]
                phi = torch.stack(phis)[sel]
                amp = torch.stack(amps)[sel] * amp_scale
                if mp_amp_max is not None:
                    amp = amp.clamp(max=mp_amp_max)
            else:
                k = min(num_atoms, cand_mag_t.numel())
                topk = torch.topk(cand_mag_t, k=k, largest=True)
                sel = topk.indices

                freq_idx = cand_freq_t[sel]
                time_idx = cand_time_t[sel]
                mag_vals = cand_mag_t[sel]
                frame_max_vals = cand_frame_max_t[sel]

                tau = time_idx.float() * hop_length / self.sample_rate
                if time_jitter_ratio > 0.0:
                    time_jitter = (torch.rand_like(tau) - 0.5) * (hop_length / self.sample_rate) * time_jitter_ratio
                    tau = tau + time_jitter
                tau = tau.clamp(0.0, self.audio_duration - 1e-6)

                omega = freq_idx.float() * (self.sample_rate / n_fft)
                if freq_jitter_bins > 0.0:
                    freq_jitter = (torch.rand_like(omega) - 0.5) * freq_jitter_bins * (self.sample_rate / n_fft)
                    omega = omega + freq_jitter
                omega = omega.clamp(min=50.0, max=0.99 * self.nyquist_freq)
                phi = phase[freq_idx, time_idx]

                frame_energy_vals = frame_energy[time_idx]
                frame_energy_ratio = (frame_energy_vals / (frame_energy.max() + 1e-8)).sqrt()
                amp = (mag_vals / (frame_max_vals + 1e-8)) * amp_scale * frame_energy_ratio

                sigma = _constant_q_sigma(omega, init_cfg).clamp(
                    min=init_cfg.sigma_min,
                    max=init_cfg.sigma_max,
                )

            amp = amp.clamp(min=1e-6)
            amp_logit = _inv_softplus(amp)
            sigma_logit = _sigma_real_to_logit(sigma)
            omega_logit = _omega_real_to_logit(omega, self.nyquist_freq)

            phi_vector = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
            gamma = torch.zeros_like(omega)

        self.add_atoms(amp_logit, tau, omega_logit, sigma_logit, phi_vector, gamma)
        return int(amp.shape[0])
    
    def _stft_init(self, waveform: torch.Tensor, taus: torch.Tensor, n_fft: int, hop_length: int):
        """
        STFT-based initialization with frequency limit to 80% Nyquist.
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
        
        # Limit to 99% Nyquist
        max_bin = int(num_freq_bins * 0.99)
        
        num_atoms = len(taus)
        omegas = torch.zeros(num_atoms, device=self.device)
        phis = torch.zeros(num_atoms, device=self.device)
        
        for i, t in enumerate(taus):
            frame_idx = int(t / self.audio_duration * (num_frames - 1))
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            
            # Only consider first 80% of frequency bins
            frame_mag = mag[:max_bin, frame_idx]
            if frame_mag.sum() > 0:
                probs = torch.sqrt(frame_mag + 1e-10)
                probs = probs / probs.sum()
                freq_idx = torch.multinomial(probs, 1).item()
            else:
                freq_idx = torch.randint(0, max_bin, (1,)).item()
            
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
        # Scale to 99% Nyquist to allow full frequency range
        return torch.sigmoid(self._omega_logit) * 0.99 * self.nyquist_freq

    @property
    def sigma(self) -> torch.Tensor: 
        # Use softplus for smooth positive sigma values
        # Minimum 1ms (0.001s) to match CUDA kernel clamp and prevent gradient explosion
        return F.softplus(self._sigma_logit) + SIGMA_OFFSET

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

    def clamp_parameters(self):
        """Clamp parameters to valid ranges after optimizer steps."""
        with torch.no_grad():
            self._tau.data.clamp_(0.0, self.audio_duration - 1e-6)

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

    def clone_atoms_by_indices(self, indices, clone_config: Optional[dict] = None):
        """
        Clone atoms using JITTERED HARMONIC STACKING for additive synthesis.
        
        Issue B Fix: Uses correct inverse transforms for sigma and omega logits.
        - omega uses 0.99*nyquist scaling (matches forward property)
        - sigma uses inv_softplus(sigma_real - SIGMA_OFFSET)
        """
        if len(indices) == 0: 
            return 0

        cfg = clone_config or {}
        strategy = cfg.get("strategy", "harmonic")
        harmonic_prob = cfg.get("harmonic_prob", 0.5)
        local_jitter = cfg.get("local_jitter_ratio", 0.08)
        harmonic_jitter = cfg.get("harmonic_jitter_ratio", 0.05)
        max_freq_ratio = cfg.get("max_freq_ratio", 0.95)
        tau_jitter_std = cfg.get("tau_jitter_std", 0.005)
        sigma_scale_harmonic = cfg.get("sigma_scale_harmonic", 0.5)
        sigma_scale_local = cfg.get("sigma_scale_local", 0.8)

        indices = indices.to(self.device)
        if strategy == "mixed":
            selector = torch.rand(len(indices), device=self.device) < harmonic_prob
            harmonic_indices = indices[selector]
            local_indices = indices[~selector]
        elif strategy == "local":
            harmonic_indices = torch.tensor([], device=self.device, dtype=torch.long)
            local_indices = indices
        else:
            harmonic_indices = indices
            local_indices = torch.tensor([], device=self.device, dtype=torch.long)

        def _clone_with_targets(valid_indices, target_omega_hz, sigma_scale):
            if len(valid_indices) == 0:
                return 0
            # Filter by max frequency ratio
            valid_mask = target_omega_hz < (max_freq_ratio * self.nyquist_freq)
            if not valid_mask.any():
                return 0
            valid_target_omega = target_omega_hz[valid_mask]
            valid_idx = valid_indices[valid_mask]

            new_omega_logit = _omega_real_to_logit(valid_target_omega, self.nyquist_freq)
            parent_amp = self.amplitude[valid_idx]
            new_amp_real = (parent_amp * CLONE_AMP_SCALE).clamp(min=1e-6)
            new_amp = _inv_softplus(new_amp_real)
            new_tau = self._tau.data[valid_idx].clone()
            tau_jitter = torch.randn_like(new_tau) * tau_jitter_std
            new_tau = (new_tau + tau_jitter).clamp(0, self.audio_duration - 1e-6)

            parent_sigma_real = self.sigma[valid_idx]
            new_sigma_real = (parent_sigma_real * sigma_scale).clamp(min=SIGMA_OFFSET + 1e-6)
            new_sigma = _sigma_real_to_logit(new_sigma_real)

            new_phi_vector = self._phi_vector.data[valid_idx].clone()
            new_gamma = self._gamma.data[valid_idx].clone()

            self.add_atoms(new_amp, new_tau, new_omega_logit, new_sigma, new_phi_vector, new_gamma)
            return len(valid_idx)

        num_added = 0
        if len(harmonic_indices) > 0:
            parent_omega_hz = self.omega[harmonic_indices]
            jitter = torch.empty_like(parent_omega_hz).uniform_(1.0 - harmonic_jitter, 1.0 + harmonic_jitter)
            target_omega_hz = parent_omega_hz * 2.0 * jitter
            num_added += _clone_with_targets(harmonic_indices, target_omega_hz, sigma_scale_harmonic)
        if len(local_indices) > 0:
            parent_omega_hz = self.omega[local_indices]
            jitter = torch.empty_like(parent_omega_hz).uniform_(-local_jitter, local_jitter)
            target_omega_hz = parent_omega_hz * (1.0 + jitter)
            num_added += _clone_with_targets(local_indices, target_omega_hz, sigma_scale_local)

        return num_added

    def split_atoms_by_indices(self, indices, scale_factor=1.6):
        """
        Split atoms into two with smaller sigma and displaced tau.
        
        Issue B Fix: Uses correct inverse transform for sigma logit.
        """
        if len(indices) == 0: 
            return 0
        with torch.no_grad():
            sigma = self.sigma[indices]
            tau = self._tau.data[indices]
            offset = sigma * 0.5
            
            tau_jitter1 = torch.randn_like(tau) * 0.003
            tau_jitter2 = torch.randn_like(tau) * 0.003
            
            # Compute new sigma using CORRECT inverse transform
            new_sigma_real = (sigma / scale_factor).clamp(min=SIGMA_OFFSET + 1e-6)
            new_sigma1 = _sigma_real_to_logit(new_sigma_real)
            
            # Atom 1
            parent_amp = self.amplitude[indices]
            new_amp_real = (parent_amp * SPLIT_AMP_SCALE).clamp(min=1e-6)
            new_amp1 = _inv_softplus(new_amp_real)
            new_tau1 = (tau - offset + tau_jitter1).clamp(0, self.audio_duration - 1e-6)
            new_omega1 = self._omega_logit.data[indices].clone() + torch.randn_like(self._omega_logit.data[indices]) * 0.05
            new_phi_vector1 = self._phi_vector.data[indices].clone()
            new_gamma1 = self._gamma.data[indices].clone()
            
            # Atom 2 (phase shifted by π/2)
            new_amp2 = new_amp1.clone()
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
