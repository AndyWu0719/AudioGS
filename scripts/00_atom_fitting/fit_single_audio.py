"""
AudioGS Atom Fitting Script (Physics Pillars Compliant)

Major Refactor:
- Removed HF Sparsity hack (no longer needed with proper phase + density control)
- Integrated phase_vector for unit-circle regularization
- Added Constant-Q Regularization to prevent non-physical atoms
- Increased mel_weight for better spectral envelope

Implements:
- Pillar 1: atomicAdd (Linear Superposition)
- Pillar 3: Constant-Q Sigma Initialization (σ ∝ 1/ω)
- Pillar 4: STFT Loss > 90%
- Pillar 5: Harmonic Cloning (now density-controlled, no HF cloning)
- Pillar 6: Constant-Q Regularization (NEW)

Usage:
    conda activate qwen2_CALM
    cd /data0/determined/users/andywu/GS-TS
    python scripts/00_atom_fitting/fit_single_audio.py
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

from src.models.atom import AudioGSModel
from src.losses.spectral_loss import CombinedAudioLoss
from src.utils.density_control import AdaptiveDensityController, rebuild_optimizer_from_model
from src.utils.visualization import Visualizer
from src.tools.metrics import compute_mss_loss, compute_pesq, compute_sisdr

# Try to import CUDA renderer
CUDA_EXT_AVAILABLE = False
GaborRendererCUDA = None

# Try installed package first
try:
    from cuda_gabor import GaborRendererCUDA, CUDA_EXT_AVAILABLE
    print("[Renderer] Using installed cuda_gabor package")
except ImportError:
    # Try local path
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))
        from cuda_gabor import GaborRendererCUDA, CUDA_EXT_AVAILABLE
        print("[Renderer] Using local cuda_gabor")
    except ImportError:
        print("[Warning] CUDA extension not available, will use PyTorch fallback")

# ============================================================
# BENCHMARK TEST FILES (Pre-selected via torchaudio.info)
# ============================================================
TEST_FILES = [None, None, None]


def find_test_files_by_duration(
    root_path: str,
    target_durations: List[float] = [1.0, 3.0, 5.0],
    sample_rate: int = 24000,
) -> List[str]:
    """
    Find audio files closest to target durations using torchaudio.info.
    """
    print(f"[Benchmark] Scanning {root_path} for test files...")
    
    root = Path(root_path)
    all_files = list(root.rglob("*.wav"))
    
    if len(all_files) == 0:
        raise ValueError(f"No WAV files found in {root_path}")
    
    print(f"[Benchmark] Found {len(all_files)} WAV files, filtering by duration...")
    
    file_durations = []
    for i, fpath in enumerate(all_files[:500]):
        try:
            info = torchaudio.info(str(fpath))
            duration = info.num_frames / info.sample_rate
            file_durations.append((fpath, duration))
        except Exception:
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Scanned {i+1} files...")
    
    result = []
    for target in target_durations:
        best_file = None
        best_diff = float('inf')
        
        for fpath, duration in file_durations:
            diff = abs(duration - target)
            if diff < best_diff:
                best_diff = diff
                best_file = (fpath, duration)
        
        if best_file:
            result.append(str(best_file[0]))
            print(f"  Target {target}s: {best_file[0].name} ({best_file[1]:.2f}s)")
        else:
            result.append(None)
    
    return result


def load_config(path: str) -> Dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_audio(path: str, sample_rate: int = 24000) -> Tuple[torch.Tensor, int]:
    """Load audio file and resample if needed."""
    waveform, sr = torchaudio.load(path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    
    return waveform.squeeze(0), sample_rate


def compute_metrics(
    gt_waveform: torch.Tensor,
    pred_waveform: torch.Tensor,
    sample_rate: int = 24000,
) -> Dict[str, float]:
    """Compute reconstruction quality metrics."""
    # Ensure same length
    min_len = min(len(gt_waveform), len(pred_waveform))
    gt = gt_waveform[:min_len]
    pred = pred_waveform[:min_len]
    
    # MSS Loss (Multi-Scale Spectral)
    mss = compute_mss_loss(gt, pred, sample_rate)
    
    # L1 Loss
    l1 = torch.nn.functional.l1_loss(pred, gt).item()
    
    # SNR
    noise = pred - gt
    signal_power = (gt ** 2).mean()
    noise_power = (noise ** 2).mean()
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
    
    # Mel L1 (approximate)
    try:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80
        ).to(gt.device)
        mel_gt = mel_transform(gt.unsqueeze(0))
        mel_pred = mel_transform(pred.unsqueeze(0))
        mel_l1 = torch.nn.functional.l1_loss(
            torch.log(mel_pred + 1e-5), torch.log(mel_gt + 1e-5)
        ).item()
    except Exception:
        mel_l1 = float('nan')
    

    pesq_score = compute_pesq(gt, pred, sample_rate)
    sisdr = compute_sisdr(gt, pred)
        
    return {
        'mss_loss': mss,
        'l1_loss': l1,
        'snr_db': snr,
        'mel_l1': mel_l1,
        'pesq': pesq_score,
        'sisdr': sisdr,
    }


def fit_single_audio(
    audio_path: str,
    config: Dict,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """
    Fit Gabor atoms to a single audio file.
    
    REFACTOR: 
    - Removed HF Sparsity hack (density control now handles this)
    - Integrated phase_vector for unit-circle regularization
    - Added Constant-Q Regularization to prevent non-physical atoms
    
    Returns:
        Dict of metrics
    """
    sample_rate = config["data"]["sample_rate"]
    max_iters = config["training"]["max_iters"]
    
    # Load audio
    gt_waveform, _ = load_audio(audio_path, sample_rate)
    gt_waveform = gt_waveform.to(device)
    num_samples = len(gt_waveform)
    audio_duration = num_samples / sample_rate
    
    filename = Path(audio_path).stem
    print(f"\n[Fit] {filename} ({audio_duration:.2f}s, {num_samples} samples)")
    
    # DYNAMIC DENSITY: Compute atom counts based on duration
    initial_atoms_per_sec = config["model"].get("initial_atoms_per_second", 1024)
    max_atoms_per_sec = config["model"].get("max_atoms_per_second", 4096)
    initial_num_atoms = int(audio_duration * initial_atoms_per_sec)
    max_num_atoms = int(audio_duration * max_atoms_per_sec)
    
    # Clamp to reasonable bounds
    initial_num_atoms = max(256, min(initial_num_atoms, 8192))
    max_num_atoms = max(1024, min(max_num_atoms, 32768))
    
    print(f"[Fit] Density: {initial_num_atoms} init → {max_num_atoms} max ({max_atoms_per_sec}/sec)")
    
    # Initialize model
    model = AudioGSModel(
        num_atoms=initial_num_atoms,
        sample_rate=sample_rate,
        audio_duration=audio_duration,
        device=device,
    )
    
    # Initialize from audio (uses Constant-Q sigma internally)
    model.initialize_from_audio(gt_waveform)
    
    # Renderer
    if CUDA_EXT_AVAILABLE:
        try:
            renderer = GaborRendererCUDA(sample_rate=sample_rate, sigma_multiplier=5.0)
            print("[Fit] Using CUDA renderer (5σ truncation)")
        except Exception as e:
            print(f"[Fit] CUDA renderer failed ({e}), using PyTorch fallback")
            renderer = None
    else:
        renderer = None
    
    # Loss Configuration
    # REFACTOR: Increased mel_weight from 1.0 to 5.0 for better spectral envelope
    loss_config = config["loss"]
    loss_fn = CombinedAudioLoss(
        sample_rate=sample_rate,
        fft_sizes=loss_config["fft_sizes"],
        hop_sizes=loss_config["hop_sizes"],
        win_lengths=loss_config["win_lengths"],
        stft_weight=loss_config.get("spectral_weight", 1.0),
        mel_weight=loss_config.get("mel_weight", 5.0),  # INCREASED from 1.0 to 5.0
        time_weight=loss_config.get("time_domain_weight", 0.1),
        phase_weight=loss_config.get("phase_weight", 1.0),
        amp_reg_weight=loss_config.get("amp_reg_weight", 0.0001),
        pre_emp_weight=loss_config.get("pre_emp_weight", 2.0),
    )
    
    # Optimizer
    optimizer = rebuild_optimizer_from_model(model, torch.optim.Adam, config["training"])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[4000, 7000, 9000], gamma=0.5
    )
    
    # Density Controller - Simplified speech-aware design
    dc = config["density_control"]
    density_controller = AdaptiveDensityController(
        grad_threshold=dc["grad_threshold"],
        prune_amplitude_threshold=dc["prune_amplitude_threshold"],
        max_num_atoms=max_num_atoms,
    )
    enable_split = dc.get("enable_split", True)
    
    # Linear CQ Regularization (soft penalty, NOT exponential barrier)
    cq_cycle_limit = dc.get("cq_cycle_limit", 50.0)
    cq_reg_weight = dc.get("cq_reg_weight", 0.01)  # Gentle weight
    
    # Training loop
    pbar = tqdm(range(max_iters), desc=f"Fitting {filename}", leave=False)
    
    for iteration in pbar:
        amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
        
        # Render
        if renderer is not None:
            pred_waveform = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
        else:
            # PyTorch fallback (slower)
            pred_waveform = render_pytorch(amplitude, tau, omega, sigma, phi, gamma, 
                                           num_samples, sample_rate, device)
        
        if torch.isnan(pred_waveform).any():
            print(f"[Warning] NaNs detected in renderer output at iter {iteration}!")
            pred_waveform = torch.nan_to_num(pred_waveform, 0.0)
        
        # Get phase_vector for unit-circle regularization
        phase_vector = model.phase_vector  # Returns RAW (cos, sin) tuple
        
        # Main loss with phase regularization
        loss, loss_dict = loss_fn(
            pred_waveform, gt_waveform,
            model_amplitude=amplitude,
            model_sigma=sigma,
            model_phase_raw=phase_vector,
            sigma_diversity_weight=dc.get("sigma_diversity_weight", 0.001),
        )
        
        # =====================================================
        # LINEAR CQ REGULARIZATION (Soft penalty)
        # =====================================================
        # Gently penalize atoms exceeding cycle limit
        # Linear is more stable than exponential barrier
        
        current_cycles = sigma * omega
        cq_reg_loss = cq_reg_weight * F.relu(current_cycles - cq_cycle_limit).mean()
        loss = loss + cq_reg_loss
        loss_dict['cq_reg'] = cq_reg_loss.item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        model.accumulate_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Density control (simplified, no late-stage freeze)
        density_controller.update_thresholds(loss.item())
        if dc["densify_from_iter"] <= iteration < dc["densify_until_iter"]:
            if iteration % dc["densification_interval"] == 0 and iteration > 0:
                stats = density_controller.densify_and_prune(
                    model, optimizer,
                    do_split=enable_split,
                    do_clone=True,
                    do_prune=True,
                )
        
        # Logging
        if iteration % config["training"]["log_interval"] == 0:
            pbar.set_postfix({
                "L": f"{loss_dict['total']:.3f}",
                "mel": f"{loss_dict['mel']:.2f}",
                "ph": f"{loss_dict['phase']:.2f}",
                "cq": f"{loss_dict['cq_reg']:.4f}",  # NEW
                "pR": f"{loss_dict['phase_reg']:.3f}",
                "n": model.num_atoms
            })
    
    # Final render
    with torch.no_grad():
        amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
        if renderer is not None:
            pred_waveform = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
        else:
            pred_waveform = render_pytorch(amplitude, tau, omega, sigma, phi, gamma,
                                           num_samples, sample_rate, device)
    
    # Save reconstructed audio
    audio_output = output_dir / f"{filename}_recon.wav"
    torchaudio.save(str(audio_output), pred_waveform.unsqueeze(0).cpu(), sample_rate)
    print(f"[Fit] Saved: {audio_output}")
    
    # Generate visualization
    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(exist_ok=True)
    
    visualizer = Visualizer(str(vis_dir), sample_rate)
    visualizer.plot_spectrogram_comparison(gt_waveform, pred_waveform, f"{filename}_vis")
    print(f"[Fit] Saved visualization: {vis_dir}/{filename}_vis.png")
    
    # Compute metrics
    metrics = compute_metrics(gt_waveform, pred_waveform, sample_rate)
    metrics['final_atoms'] = model.num_atoms
    metrics['duration_s'] = audio_duration
    
    return metrics


def render_pytorch(
    amplitude, tau, omega, sigma, phi, gamma,
    num_samples, sample_rate, device
) -> torch.Tensor:
    """PyTorch fallback renderer (slower than CUDA)."""
    t = torch.arange(num_samples, device=device, dtype=torch.float32) / sample_rate
    output = torch.zeros(num_samples, device=device)
    
    for i in range(len(amplitude)):
        A = amplitude[i]
        tau_i = tau[i]
        omega_i = omega[i]
        sigma_i = sigma[i]
        phi_i = phi[i]
        gamma_i = gamma[i]
        
        t_centered = t - tau_i
        envelope = torch.exp(-t_centered**2 / (2 * sigma_i**2 + 1e-8))
        phase = 2 * 3.14159 * (omega_i * t_centered + 0.5 * gamma_i * t_centered**2) + phi_i
        carrier = torch.cos(phase)
        
        output += A * envelope * carrier
    
    return output


def main():
    parser = argparse.ArgumentParser(description="AudioGS Atom Fitting (Physics Pillars)")
    parser.add_argument("--config", type=str, default="configs/atom_fitting_config.yaml")
    parser.add_argument("--scan_files", action="store_true", help="Scan for test files")
    args = parser.parse_args()
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    config = load_config(str(config_path))
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")
    
    # Output directory
    output_dir = PROJECT_ROOT / config["output"]["root_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Main] Output: {output_dir}")
    
    # Find test files
    global TEST_FILES
    if args.scan_files or TEST_FILES[0] is None:
        TEST_FILES = find_test_files_by_duration(
            config["data"]["dataset_path"],
            target_durations=[1.0, 3.0, 5.0],
            sample_rate=config["data"]["sample_rate"],
        )
    
    # Filter out None entries
    test_files = [f for f in TEST_FILES if f is not None]
    if len(test_files) == 0:
        raise ValueError("No test files found. Check dataset path.")
    
    print(f"\n{'='*60}")
    print("AudioGS Atom Fitting Benchmark (Frequency-Adaptive Splitting)")
    print(f"{'='*60}")
    print(f"Files: {len(test_files)}")
    print(f"{'='*60}\n")
    
    # Run fitting for each test file
    all_metrics = []
    
    for audio_path in test_files:
        try:
            metrics = fit_single_audio(audio_path, config, output_dir, device)
            metrics['file'] = Path(audio_path).name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"[Error] Failed on {audio_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary table
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'File':<30} {'Duration':<10} {'SNR (dB)':<12} {'Mel-L1':<12} {'PESQ':<10} {'SI-SDR':<12} {'Atoms':<10}")
    print(f"{'-'*110}")
    
    for m in all_metrics:
        print(f"{m['file']:<30} {m['duration_s']:<10.2f} {m['snr_db']:<12.2f} {m['mel_l1']:<12.4f} {m['pesq']:<10.2f} {m['sisdr']:<12.2f} {m['final_atoms']:<10}")
    
    print(f"{'='*80}")
    
    # Save metrics to file
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("AudioGS Atom Fitting Benchmark Results (Frequency-Adaptive Splitting)\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        
        for m in all_metrics:
            f.write(f"File: {m['file']}\n")
            f.write(f"  Duration: {m['duration_s']:.2f}s\n")
            f.write(f"  SNR: {m['snr_db']:.2f} dB\n")
            f.write(f"  Mel-L1: {m['mel_l1']:.4f}\n")
            f.write(f"  PESQ: {m['pesq']:.2f}\n")
            f.write(f"  SI-SDR: {m['sisdr']:.2f} dB\n")
            f.write(f"  MSS Loss: {m['mss_loss']:.4f}\n")
            f.write(f"  L1 Loss: {m['l1_loss']:.6f}\n")
            f.write(f"  Final Atoms: {m['final_atoms']}\n")
            f.write("\n")
    
    print(f"\n[Main] Metrics saved to: {metrics_path}")
    print("[Main] Complete!")


if __name__ == "__main__":
    main()