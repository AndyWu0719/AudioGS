"""
HF Band Diagnostic Script
==========================
Captures snapshots during training to visualize HOW and WHEN the HF band artifact appears.

Usage: python scripts/00_atom_fitting/diagnose_hf_band.py --interval 1000
"""

import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import imageio
import torchaudio

from src.models.atom import AudioGSModel
from src.losses.spectral_loss import CombinedAudioLoss
from src.utils.density_control import AdaptiveDensityController

# CUDA Renderer import (same as fit_single_audio.py)
CUDA_EXT_AVAILABLE = False
GaborRendererCUDA = None

try:
    from cuda_gabor import GaborRendererCUDA, CUDA_EXT_AVAILABLE
    print("[Renderer] Using cuda_gabor package")
except ImportError:
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))
        from cuda_gabor import GaborRendererCUDA, CUDA_EXT_AVAILABLE
        print("[Renderer] Using local cuda_gabor")
    except ImportError:
        print("[Warning] CUDA extension not available")


def render_pytorch(amplitude, tau, omega, sigma, phi, gamma, num_samples, sample_rate, device):
    """PyTorch fallback renderer."""
    t = torch.arange(num_samples, device=device, dtype=torch.float32) / sample_rate
    output = torch.zeros(num_samples, device=device)
    
    for i in range(len(amplitude)):
        t_centered = t - tau[i]
        envelope = torch.exp(-t_centered**2 / (2 * sigma[i]**2 + 1e-8))
        instant_phase = 2 * np.pi * (omega[i] * t_centered + 0.5 * gamma[i] * t_centered**2) + phi[i]
        atom = amplitude[i] * envelope * torch.cos(instant_phase)
        output = output + atom
    
    return output


def compute_mel_spectrogram(waveform, sample_rate=24000, n_mels=80, n_fft=2048, hop_length=512):
    """Compute mel spectrogram for visualization."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    mel = mel_transform(waveform.unsqueeze(0))
    mel_db = 10 * torch.log10(mel + 1e-10)
    return mel_db.squeeze(0).numpy()


def create_diagnostic_frame(model, pred_waveform, gt_waveform, iteration, sample_rate=24000):
    """Create a diagnostic visualization frame."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Ground Truth Mel
    gt_mel = compute_mel_spectrogram(gt_waveform.cpu(), sample_rate)
    axes[0, 0].imshow(gt_mel, aspect='auto', origin='lower', cmap='magma')
    axes[0, 0].set_title('Ground Truth Mel')
    axes[0, 0].set_ylabel('Mel Bin')
    
    # 2. Reconstructed Mel
    pred_mel = compute_mel_spectrogram(pred_waveform.detach().cpu(), sample_rate)
    axes[0, 1].imshow(pred_mel, aspect='auto', origin='lower', cmap='magma')
    axes[0, 1].set_title(f'Reconstructed (iter {iteration})')
    
    # 3. Omega Histogram
    omega = model.omega.detach().cpu().numpy()
    nyquist = model.nyquist_freq
    
    bins = np.linspace(0, nyquist, 50)
    axes[1, 0].hist(omega, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0.80 * nyquist, color='orange', linestyle='--', label='80% Nyquist')
    axes[1, 0].axvline(x=0.85 * nyquist, color='red', linestyle='--', label='85% Nyquist')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Atom Count')
    axes[1, 0].set_title(f'Omega Distribution (n={len(omega)})')
    axes[1, 0].legend()
    
    above_80 = (omega > 0.80 * nyquist).sum()
    above_85 = (omega > 0.85 * nyquist).sum()
    axes[1, 0].text(0.95, 0.95, f'>80%: {above_80}\n>85%: {above_85}', 
                    transform=axes[1, 0].transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Sigma vs Omega scatter
    sigma = model.sigma.detach().cpu().numpy() * 1000
    amplitude = model.amplitude.detach().cpu().numpy()
    
    scatter = axes[1, 1].scatter(omega, sigma, c=amplitude, cmap='viridis', alpha=0.5, s=5, vmin=0, vmax=0.5)
    axes[1, 1].axvline(x=0.80 * nyquist, color='orange', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=0.85 * nyquist, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Omega (Hz)')
    axes[1, 1].set_ylabel('Sigma (ms)')
    axes[1, 1].set_title('Sigma vs Omega (color=amplitude)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Amplitude')
    
    plt.suptitle(f'HF Band Diagnostic - Iteration {iteration}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig.canvas.draw()
    # Use buffer_rgba for newer matplotlib
    buf = fig.canvas.buffer_rgba()
    image = np.asarray(buf)[:, :, :3]  # Drop alpha channel
    plt.close(fig)
    
    return image


def diagnose_hf_band(audio_path: str, output_dir: str, checkpoint_interval: int = 500):
    """Run diagnostic fitting."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Diagnose] Device: {device}")
    
    # Load config
    config_path = PROJECT_ROOT / 'configs' / 'atom_fitting_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    q_multiplier = 2.0 * np.pi if config.get("initialization", {}).get("constant_q_use_2pi", True) else 1.0
    
    sample_rate = config['data']['sample_rate']
    max_iters = config['training']['max_iters']
    
    # Load audio
    gt_waveform, sr = torchaudio.load(audio_path)
    gt_waveform = gt_waveform.squeeze(0)
    if sr != sample_rate:
        gt_waveform = torchaudio.functional.resample(gt_waveform, sr, sample_rate)
    gt_waveform = gt_waveform.to(device)
    num_samples = len(gt_waveform)
    audio_duration = num_samples / sample_rate
    
    filename = Path(audio_path).stem
    print(f"[Diagnose] {filename} ({audio_duration:.2f}s, {num_samples} samples)")
    
    # Initialize model
    initial_atoms = int(audio_duration * config['model']['initial_atoms_per_second'])
    max_atoms = int(audio_duration * config['model']['max_atoms_per_second'])
    initial_atoms = max(256, min(initial_atoms, 8192))
    max_atoms = max(1024, min(max_atoms, 32768))
    
    model = AudioGSModel(
        num_atoms=initial_atoms, sample_rate=sample_rate,
        audio_duration=audio_duration, device=device,
    )
    model.initialize_from_audio(gt_waveform.cpu(), init_config=config.get("initialization", {}))
    
    # Renderer
    renderer = None
    if CUDA_EXT_AVAILABLE and GaborRendererCUDA is not None:
        renderer = GaborRendererCUDA(sample_rate=sample_rate)
        print("[Diagnose] Using CUDA renderer")
    else:
        print("[Diagnose] Using PyTorch fallback (slower)")
    
    # Setup optimizer
    lr_config = {k: v for k, v in config['training'].items() if k.startswith('lr_')}
    param_groups = model.get_optimizer_param_groups(lr_config)
    optimizer = torch.optim.Adam(param_groups)
    
    # Setup loss
    loss_config = config['loss']
    loss_fn = CombinedAudioLoss(
        sample_rate=sample_rate,
        fft_sizes=loss_config['fft_sizes'],
        hop_sizes=loss_config['hop_sizes'],
        win_lengths=loss_config['win_lengths'],
        stft_weight=loss_config.get('spectral_weight', 1.0),
        mel_weight=loss_config.get('mel_weight', 5.0),
        time_weight=loss_config.get('time_domain_weight', 0.1),
        phase_weight=loss_config.get('phase_weight', 1.0),
        pre_emp_weight=loss_config.get('pre_emp_weight', 2.0),
        hf_weight=loss_config.get('hf_weight', 0.0),
        hf_min_freq_ratio=loss_config.get('hf_min_freq_ratio', 0.6),
        energy_weight=loss_config.get('energy_weight', 0.0),
        gamma_reg_weight=loss_config.get('gamma_reg_weight', 0.0),
        phase_reg_weight=loss_config.get('phase_reg_weight', 0.1),
        spec_tv_weight=loss_config.get('spec_tv_weight', 0.0),
        spec_tv_freq_ratio=loss_config.get('spec_tv_freq_ratio', 0.5),
    )
    
    # Density controller
    dc = config['density_control']
    densify_from_iter = dc["densify_from_iter"]
    densify_until_iter = dc["densify_until_iter"]
    densification_interval = dc["densification_interval"]
    ref_duration = dc.get("densify_reference_duration_s", 3.0)
    min_scale = dc.get("densify_min_scale", 0.25)
    max_scale = dc.get("densify_max_scale", 2.0)
    scale = max(min(audio_duration / ref_duration, max_scale), min_scale)
    densify_from_iter = max(1, int(densify_from_iter * scale))
    densification_interval = max(1, int(densification_interval * scale))
    densify_until_iter = max(densify_until_iter, densify_from_iter + densification_interval)
    density_controller = AdaptiveDensityController(
        grad_threshold=dc['grad_threshold'],
        prune_amplitude_threshold=dc['prune_amplitude_threshold'],
        max_num_atoms=max_atoms,
        init_config=config.get("initialization", {}),
        clone_sigma_ratio_max=dc.get("clone_sigma_ratio_max", 1.25),
        clone_sigma_max=dc.get("clone_sigma_max", 0.03),
        prune_sigma_exponent=dc.get("prune_sigma_exponent", 0.5),
        prune_sigma_max_boost=dc.get("prune_sigma_max_boost", 4.0),
    )
    residual_spawn = dc.get("residual_spawn", True)
    residual_spawn_energy_ratio = dc.get("residual_spawn_energy_ratio", 0.05)
    residual_spawn_max = dc.get("residual_spawn_max", 128)
    residual_spawn_fft = dc.get("residual_spawn_fft", 2048)
    residual_spawn_hop = dc.get("residual_spawn_hop", 512)
    residual_amp_scale = dc.get("residual_amp_scale", 0.3)
    residual_peaks_per_frame = dc.get("residual_spawn_peaks_per_frame", 4)
    residual_min_peak_ratio = dc.get("residual_spawn_min_peak_ratio", 0.25)
    residual_time_jitter_ratio = dc.get("residual_spawn_time_jitter_ratio", 0.5)
    residual_freq_jitter_bins = dc.get("residual_spawn_freq_jitter_bins", 0.5)
    residual_selection_strategy = dc.get("residual_selection_strategy", "stft_peak")
    residual_mp_sigma_multiplier = dc.get("residual_mp_sigma_multiplier", 5.0)
    residual_mp_amp_max = dc.get("residual_mp_amp_max", None)
    residual_mp_normalize = dc.get("residual_mp_normalize", True)
    residual_mp_score_min = dc.get("residual_mp_score_min", 0.0)
    if isinstance(residual_mp_amp_max, (int, float)) and residual_mp_amp_max <= 0:
        residual_mp_amp_max = None

    clone_config = {
        "strategy": dc.get("clone_strategy", "harmonic"),
        "harmonic_prob": dc.get("clone_harmonic_prob", 0.5),
        "local_jitter_ratio": dc.get("clone_local_jitter_ratio", 0.08),
        "harmonic_jitter_ratio": dc.get("clone_harmonic_jitter_ratio", 0.05),
        "max_freq_ratio": dc.get("clone_max_freq_ratio", 0.95),
        "tau_jitter_std": dc.get("clone_tau_jitter_std", 0.005),
        "sigma_scale_harmonic": dc.get("clone_sigma_scale_harmonic", 0.5),
        "sigma_scale_local": dc.get("clone_sigma_scale_local", 0.8),
    }
    
    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect frames
    frames = []
    hf_counts = []
    
    print(f"[Diagnose] Capturing frames every {checkpoint_interval} iterations")
    
    # Training loop
    pbar = tqdm(range(max_iters), desc="Diagnosing")
    
    for iteration in pbar:
        amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
        
        # Render
        if renderer is not None:
            pred_waveform = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
        else:
            pred_waveform = render_pytorch(amplitude, tau, omega, sigma, phi, gamma,
                                           num_samples, sample_rate, device)
        
        # Loss
        phase_vector = model.phase_vector
        loss, loss_dict = loss_fn(
            pred_waveform,
            gt_waveform,
            model_amplitude=amplitude,
            model_sigma=sigma,
            model_phase_raw=phase_vector,
            model_gamma=gamma,
            phase_reg_weight=loss_config.get("phase_reg_weight", 0.1),
        )
        
        # CQ regularization
        cq_limit = dc.get('cq_cycle_limit', 50.0)
        cq_weight = dc.get('cq_reg_weight', 0.01)
        cq_loss = cq_weight * F.relu(q_multiplier * sigma * omega - cq_limit).mean()
        loss = loss + cq_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        model.accumulate_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.clamp_parameters()
        
        # Density control
        density_controller.update_thresholds(loss.item())
        if densify_from_iter <= iteration < densify_until_iter:
            if iteration % densification_interval == 0 and iteration > 0:
                density_controller.densify_and_prune(model, optimizer, clone_config=clone_config)
                if residual_spawn:
                    with torch.no_grad():
                        residual = (gt_waveform - pred_waveform).detach()
                        residual_energy = (residual ** 2).mean()
                        target_energy = (gt_waveform ** 2).mean()
                        ratio = residual_energy / (target_energy + 1e-8)
                    if ratio > residual_spawn_energy_ratio:
                        remaining_budget = max_atoms - model.num_atoms
                        num_new = min(remaining_budget, residual_spawn_max)
                        if num_new > 0:
                            density_controller.add_atoms_from_residual(
                                model,
                                optimizer,
                                residual,
                                num_new,
                                init_config=config.get("initialization", {}),
                                n_fft=residual_spawn_fft,
                                hop_length=residual_spawn_hop,
                                amp_scale=residual_amp_scale,
                                peaks_per_frame=residual_peaks_per_frame,
                                min_peak_ratio=residual_min_peak_ratio,
                                time_jitter_ratio=residual_time_jitter_ratio,
                                freq_jitter_bins=residual_freq_jitter_bins,
                                selection_strategy=residual_selection_strategy,
                                sigma_multiplier=residual_mp_sigma_multiplier,
                                mp_amp_max=residual_mp_amp_max,
                                mp_normalize=residual_mp_normalize,
                                mp_score_min=residual_mp_score_min,
                            )
        
        # Capture checkpoint
        if iteration % checkpoint_interval == 0 or iteration == max_iters - 1:
            omega_np = model.omega.detach().cpu().numpy()
            above_85 = (omega_np > 0.85 * model.nyquist_freq).sum()
            hf_counts.append((iteration, above_85))
            
            with torch.no_grad():
                amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
                if renderer is not None:
                    pred = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
                else:
                    pred = render_pytorch(amplitude, tau, omega, sigma, phi, gamma, num_samples, sample_rate, device)
            
            frame = create_diagnostic_frame(model, pred, gt_waveform, iteration, sample_rate)
            frames.append(frame)
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'atoms': model.num_atoms, 'HF>85%': above_85})
    
    # Save animation
    gif_path = output_path / f'{filename}_hf_diagnostic.gif'
    print(f"[Diagnose] Saving animation to {gif_path}")
    imageio.mimsave(gif_path, frames, duration=0.5)
    
    # Save HF count plot
    fig, ax = plt.subplots(figsize=(10, 5))
    iters, counts = zip(*hf_counts)
    ax.plot(iters, counts, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Atoms > 85% Nyquist')
    ax.set_title('HF Atom Count Over Training')
    ax.grid(True, alpha=0.3)
    plt.savefig(output_path / f'{filename}_hf_count.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Diagnose] Complete! Check {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, default=None)
    parser.add_argument('--output', type=str, default='logs/00_atom_fitting/diagnostic')
    parser.add_argument('--interval', type=int, default=500)
    args = parser.parse_args()
    
    if args.audio is None:
        target = PROJECT_ROOT / 'data/raw/LibriTTS_R/train/train-clean-100/426/122821/426_122821_000040_000000.wav'
        if target.exists():
            args.audio = str(target)
        else:
            print("Error: No audio file specified")
            sys.exit(1)
    
    diagnose_hf_band(args.audio, args.output, args.interval)
