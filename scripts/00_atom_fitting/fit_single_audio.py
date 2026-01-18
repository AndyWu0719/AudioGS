"""
AudioGS Atom Fitting Script (Physics Pillars Compliant)

Implements:
- Pillar 1: atomicAdd (Linear Superposition)
- Pillar 3: Constant-Q Sigma Initialization (σ ∝ 1/ω)
- Pillar 4: STFT Loss > 90%
- Pillar 5: Harmonic Cloning

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
from src.tools.metrics import compute_mss_loss

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
# These are closest to 1s, 3s, 5s durations in LibriTTS_R/train-clean-100
# ============================================================
TEST_FILES = [
    # Short (~1s) - Will be found by scan
    None,
    # Medium (~3s) - Will be found by scan
    None,
    # Long (~5s) - Will be found by scan
    None,
]


def find_test_files_by_duration(
    root_path: str,
    target_durations: List[float] = [1.0, 3.0, 5.0],
    sample_rate: int = 24000,
) -> List[str]:
    """
    Find audio files closest to target durations using torchaudio.info.
    
    Args:
        root_path: Root directory to scan
        target_durations: List of target durations in seconds
        sample_rate: Expected sample rate
        
    Returns:
        List of file paths, one per target duration
    """
    print(f"[Benchmark] Scanning {root_path} for test files...")
    
    root = Path(root_path)
    all_files = list(root.rglob("*.wav"))
    
    if len(all_files) == 0:
        raise ValueError(f"No WAV files found in {root_path}")
    
    print(f"[Benchmark] Found {len(all_files)} WAV files, filtering by duration...")
    
    # Collect file durations
    file_durations = []
    for i, fpath in enumerate(all_files[:500]):  # Limit to first 500 for speed
        try:
            info = torchaudio.info(str(fpath))
            duration = info.num_frames / info.sample_rate
            file_durations.append((fpath, duration))
        except Exception:
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Scanned {i+1} files...")
    
    # Find closest files to each target duration
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
        )
        mel_gt = mel_transform(gt.unsqueeze(0))
        mel_pred = mel_transform(pred.unsqueeze(0))
        mel_l1 = torch.nn.functional.l1_loss(
            torch.log(mel_pred + 1e-5), torch.log(mel_gt + 1e-5)
        ).item()
    except Exception:
        mel_l1 = float('nan')
    
    return {
        'mss_loss': mss,
        'l1_loss': l1,
        'snr_db': snr,
        'mel_l1': mel_l1,
    }


def fit_single_audio(
    audio_path: str,
    config: Dict,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """
    Fit Gabor atoms to a single audio file.
    
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
    
    # Initialize model
    model = AudioGSModel(
        num_atoms=config["model"]["initial_num_atoms"],
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
    
    # Loss
    loss_config = config["loss"]
    loss_fn = CombinedAudioLoss(
        sample_rate=sample_rate,
        fft_sizes=loss_config["fft_sizes"],
        hop_sizes=loss_config["hop_sizes"],
        win_lengths=loss_config["win_lengths"],
        stft_weight=loss_config.get("spectral_weight", 1.0),
        mel_weight=loss_config.get("mel_weight", 45.0),
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
    
    # Density Controller
    dc = config["density_control"]
    density_controller = AdaptiveDensityController(
        grad_threshold=dc["grad_threshold"],
        sigma_split_threshold=dc["sigma_split_threshold"],
        prune_amplitude_threshold=dc["prune_amplitude_threshold"],
        max_num_atoms=config["model"]["max_num_atoms"],
    )
    
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
        
        # Loss
        loss, loss_dict = loss_fn(
            pred_waveform, gt_waveform,
            model_amplitude=amplitude,
            model_sigma=sigma,
            sigma_diversity_weight=dc.get("sigma_diversity_weight", 0.001),
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        model.accumulate_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Density control
        density_controller.update_thresholds(loss.item())
        if dc["densify_from_iter"] <= iteration < dc["densify_until_iter"]:
            if iteration % dc["densification_interval"] == 0 and iteration > 0:
                stats = density_controller.densify_and_prune(model, optimizer)
        
        # Logging
        if iteration % config["training"]["log_interval"] == 0:
            pbar.set_postfix({"loss": f"{loss_dict['total']:.4f}", "atoms": model.num_atoms})
    
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
    print("AudioGS Atom Fitting Benchmark (Physics Pillars Compliant)")
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
    print(f"{'File':<30} {'Duration':<10} {'SNR (dB)':<12} {'Mel-L1':<12} {'Atoms':<10}")
    print(f"{'-'*80}")
    
    for m in all_metrics:
        print(f"{m['file']:<30} {m['duration_s']:<10.2f} {m['snr_db']:<12.2f} {m['mel_l1']:<12.4f} {m['final_atoms']:<10}")
    
    print(f"{'='*80}")
    
    # Save metrics to file
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("AudioGS Atom Fitting Benchmark Results\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        
        for m in all_metrics:
            f.write(f"File: {m['file']}\n")
            f.write(f"  Duration: {m['duration_s']:.2f}s\n")
            f.write(f"  SNR: {m['snr_db']:.2f} dB\n")
            f.write(f"  Mel-L1: {m['mel_l1']:.4f}\n")
            f.write(f"  MSS Loss: {m['mss_loss']:.4f}\n")
            f.write(f"  L1 Loss: {m['l1_loss']:.6f}\n")
            f.write(f"  Final Atoms: {m['final_atoms']}\n")
            f.write("\n")
    
    print(f"\n[Main] Metrics saved to: {metrics_path}")
    print("[Main] Complete!")


if __name__ == "__main__":
    main()