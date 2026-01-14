"""
Audio Quality Evaluation Script for Gabor Atom Reconstruction.

Evaluates the quality of atom-reconstructed audio using standard metrics:
- PESQ: Perceptual Evaluation of Speech Quality
- STOI: Short-Time Objective Intelligibility
- MCD: Mel Cepstral Distortion
- SI-SDR: Scale-Invariant Signal-to-Distortion Ratio

Usage:
    conda activate qwen2_CALM
    pip install pesq pystoi  # Install dependencies if needed
    python scripts/evaluate_atoms.py --data_dir data/atoms/LibriTTS_R/train/train-clean-100
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

# Import CUDA renderer
try:
    from cuda_gabor import get_cuda_gabor_renderer
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    print("[Warning] CUDA renderer not available")

# Import quality metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("[Warning] PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("[Warning] STOI not available. Install with: pip install pystoi")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate atom reconstruction quality")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/AudioGS_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/atoms/LibriTTS_R/train/train-clean-100",
        help="Directory containing .pt atom files"
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        default=None,  # Will be loaded from config
        help="Directory containing original .wav files"
    )
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate")
    parser.add_argument("--sample_rate", type=int, default=None, help="Sample rate (from config if not set)")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================
# Metric Implementations
# ============================================================

def compute_pesq(original: np.ndarray, reconstructed: np.ndarray, sr: int = 24000) -> float:
    """Compute PESQ score (Perceptual Evaluation of Speech Quality)."""
    if not PESQ_AVAILABLE:
        return float('nan')
    
    try:
        # PESQ requires 16kHz or 8kHz
        if sr != 16000:
            # Resample to 16kHz
            original_16k = torchaudio.functional.resample(
                torch.from_numpy(original), sr, 16000
            ).numpy()
            reconstructed_16k = torchaudio.functional.resample(
                torch.from_numpy(reconstructed), sr, 16000
            ).numpy()
        else:
            original_16k = original
            reconstructed_16k = reconstructed
        
        # Ensure same length
        min_len = min(len(original_16k), len(reconstructed_16k))
        original_16k = original_16k[:min_len]
        reconstructed_16k = reconstructed_16k[:min_len]
        
        score = pesq(16000, original_16k, reconstructed_16k, 'wb')
        return score
    except Exception as e:
        return float('nan')


def compute_stoi(original: np.ndarray, reconstructed: np.ndarray, sr: int = 24000) -> float:
    """Compute STOI score (Short-Time Objective Intelligibility)."""
    if not STOI_AVAILABLE:
        return float('nan')
    
    try:
        # Ensure same length
        min_len = min(len(original), len(reconstructed))
        score = stoi(original[:min_len], reconstructed[:min_len], sr, extended=False)
        return score
    except Exception as e:
        return float('nan')


def compute_mcd(original: np.ndarray, reconstructed: np.ndarray, sr: int = 24000, n_mfcc: int = 13) -> float:
    """
    Compute MCD (Mel Cepstral Distortion).
    
    MCD = (10 / ln(10)) * sqrt(2 * sum((mfcc1 - mfcc2)^2))
    """
    try:
        # Compute MFCCs
        original_t = torch.from_numpy(original).float().unsqueeze(0)
        reconstructed_t = torch.from_numpy(reconstructed).float().unsqueeze(0)
        
        # Ensure same length
        min_len = min(original_t.shape[-1], reconstructed_t.shape[-1])
        original_t = original_t[..., :min_len]
        reconstructed_t = reconstructed_t[..., :min_len]
        
        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80},
        )
        
        mfcc1 = mfcc_transform(original_t)  # [1, n_mfcc, T]
        mfcc2 = mfcc_transform(reconstructed_t)
        
        # Ensure same time frames
        min_t = min(mfcc1.shape[-1], mfcc2.shape[-1])
        mfcc1 = mfcc1[..., :min_t]
        mfcc2 = mfcc2[..., :min_t]
        
        # Compute MCD (excluding c0)
        diff = mfcc1[:, 1:, :] - mfcc2[:, 1:, :]  # Exclude DC component
        mcd = (10.0 / np.log(10)) * torch.sqrt(2 * (diff ** 2).sum(dim=1)).mean()
        
        return mcd.item()
    except Exception as e:
        return float('nan')


def compute_sisdr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio).
    
    SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    where s_target = (<s', s> / ||s||^2) * s
    """
    try:
        # Ensure same length
        min_len = min(len(original), len(reconstructed))
        ref = original[:min_len].astype(np.float64)
        est = reconstructed[:min_len].astype(np.float64)
        
        # Zero mean
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        
        # SI-SDR calculation
        dot = np.dot(ref, est)
        s_target = dot * ref / (np.dot(ref, ref) + 1e-8)
        e_noise = est - s_target
        
        si_sdr = 10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8))
        
        return float(si_sdr)
    except Exception as e:
        return float('nan')


# ============================================================
# Audio Reconstruction
# ============================================================

def reconstruct_audio_from_atoms(
    atoms: torch.Tensor,
    audio_duration: float,
    sample_rate: int,
    device: torch.device,
    renderer,
) -> np.ndarray:
    """
    Reconstruct audio from atom parameters.
    
    Args:
        atoms: [N, 6] tensor (tau_norm, omega, sigma, amplitude, phi, gamma)
        audio_duration: Duration in seconds
        sample_rate: Sample rate
        device: CUDA device
        renderer: CUDA Gabor renderer
        
    Returns:
        audio: Numpy array of reconstructed audio
    """
    num_samples = int(audio_duration * sample_rate)
    
    # Denormalize tau from [0,1] to seconds
    atoms = atoms.to(device)
    tau = atoms[:, 0] * audio_duration  # Denormalize tau
    omega = atoms[:, 1]
    sigma = atoms[:, 2]
    amplitude = atoms[:, 3]
    phi = atoms[:, 4]
    gamma = atoms[:, 5]
    
    # Render audio
    with torch.no_grad():
        audio = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
    
    return audio.cpu().numpy()


def load_original_audio(original_path: str, sample_rate: int) -> Optional[np.ndarray]:
    """Load original audio file."""
    try:
        waveform, sr = torchaudio.load(original_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        
        return waveform.numpy()
    except Exception as e:
        return None


# ============================================================
# Baseline Comparisons
# ============================================================

def get_baseline_metrics() -> Dict:
    """
    Reference baseline metrics for comparison.
    
    These are typical values from literature for speech reconstruction.
    """
    return {
        "original": {
            "PESQ": 4.5,
            "STOI": 1.0,
            "MCD": 0.0,
            "SI-SDR": float('inf'),
            "description": "Original audio (perfect)"
        },
        "high_quality_vocoder": {
            "PESQ": 3.8,
            "STOI": 0.95,
            "MCD": 3.0,
            "SI-SDR": 15.0,
            "description": "HiFi-GAN / BigVGAN level"
        },
        "medium_quality": {
            "PESQ": 3.0,
            "STOI": 0.85,
            "MCD": 5.0,
            "SI-SDR": 10.0,
            "description": "Standard neural vocoder"
        },
        "low_quality": {
            "PESQ": 2.0,
            "STOI": 0.70,
            "MCD": 8.0,
            "SI-SDR": 5.0,
            "description": "Low quality / artifacts"
        },
    }


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_sample(
    atom_file: Path,
    original_dir: Path,
    sample_rate: int,
    device: torch.device,
    renderer,
) -> Optional[Dict]:
    """Evaluate a single sample."""
    try:
        # Load atoms
        data = torch.load(atom_file, weights_only=False)
        atoms = data['atoms']
        audio_duration = data['audio_duration']
        source_path = data.get('source_path', '')
        
        # Find original audio
        if source_path and Path(source_path).exists():
            original_path = source_path
        else:
            # Try to reconstruct path from relative structure
            relative = atom_file.relative_to(Path(atom_file).parents[2])
            original_path = original_dir / relative.with_suffix('.wav')
        
        # Load original
        original = load_original_audio(str(original_path), sample_rate)
        if original is None:
            return None
        
        # Reconstruct from atoms
        reconstructed = reconstruct_audio_from_atoms(
            atoms, audio_duration, sample_rate, device, renderer
        )
        
        # Ensure same length
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]
        
        # Normalize
        original = original / (np.abs(original).max() + 1e-8)
        reconstructed = reconstructed / (np.abs(reconstructed).max() + 1e-8)
        
        # Compute metrics
        metrics = {
            "file": str(atom_file.name),
            "duration": audio_duration,
            "num_atoms": atoms.shape[0],
            "reconstruction_loss": data.get('final_loss', float('nan')),
            "PESQ": compute_pesq(original, reconstructed, sample_rate),
            "STOI": compute_stoi(original, reconstructed, sample_rate),
            "MCD": compute_mcd(original, reconstructed, sample_rate),
            "SI-SDR": compute_sisdr(original, reconstructed),
        }
        
        return metrics
        
    except Exception as e:
        print(f"[Error] {atom_file}: {e}")
        return None


def main():
    args = parse_args()
    
    # Load config file
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"[Config] Loaded from {args.config}")
    else:
        config = {}
        print(f"[Warning] Config file not found: {args.config}")
    
    # Get parameters from config or args
    sample_rate = args.sample_rate or config.get('data', {}).get('sample_rate', 24000)
    original_dir = args.original_dir or config.get('data', {}).get('dataset_path', '') + '/train/train-clean-100'
    
    print("=" * 70)
    print("Audio Gaussian Splatting - Quality Evaluation")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Sample rate: {sample_rate}")
    print(f"Data dir: {args.data_dir}")
    print(f"Original dir: {original_dir}")
    print(f"Max samples: {args.max_samples}")
    print("=" * 70)
    
    # Check dependencies
    if not RENDERER_AVAILABLE:
        print("[Error] CUDA renderer required. Install cuda_gabor extension.")
        return
    
    if not PESQ_AVAILABLE:
        print("[Warning] PESQ not available. Install: pip install pesq")
    
    if not STOI_AVAILABLE:
        print("[Warning] STOI not available. Install: pip install pystoi")
    
    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Initialize renderer (suppress prints)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        renderer = get_cuda_gabor_renderer(sample_rate=sample_rate)
    
    # Collect atom files
    data_dir = PROJECT_ROOT / args.data_dir
    original_dir_path = Path(original_dir)
    
    atom_files = list(data_dir.rglob("*.pt"))
    print(f"Found {len(atom_files)} atom files")
    
    if args.max_samples and len(atom_files) > args.max_samples:
        import random
        random.shuffle(atom_files)
        atom_files = atom_files[:args.max_samples]
        print(f"Sampling {args.max_samples} files for evaluation")
    
    # Evaluate
    results = []
    
    for atom_file in tqdm(atom_files, desc="Evaluating"):
        metrics = evaluate_sample(
            atom_file, original_dir_path, sample_rate, device, renderer
        )
        if metrics:
            results.append(metrics)
    
    print(f"\nEvaluated {len(results)} samples successfully")
    
    # Aggregate statistics
    if results:
        # Compute mean and std for each metric
        metric_names = ["PESQ", "STOI", "MCD", "SI-SDR", "reconstruction_loss"]
        
        summary = {}
        for name in metric_names:
            values = [r[name] for r in results if not np.isnan(r.get(name, float('nan')))]
            if values:
                summary[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
        
        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 70)
        for name in metric_names:
            if name in summary:
                s = summary[name]
                print(f"{name:<20} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")
        
        # Compare with baselines
        print("\n" + "=" * 70)
        print("COMPARISON WITH BASELINES")
        print("=" * 70)
        baselines = get_baseline_metrics()
        
        print(f"{'Method':<30} {'PESQ':>8} {'STOI':>8} {'MCD':>8} {'SI-SDR':>8}")
        print("-" * 70)
        
        # Our method
        our_pesq = summary.get('PESQ', {}).get('mean', float('nan'))
        our_stoi = summary.get('STOI', {}).get('mean', float('nan'))
        our_mcd = summary.get('MCD', {}).get('mean', float('nan'))
        our_sisdr = summary.get('SI-SDR', {}).get('mean', float('nan'))
        print(f"{'AudioGS (Ours)':<30} {our_pesq:>8.3f} {our_stoi:>8.3f} {our_mcd:>8.2f} {our_sisdr:>8.2f}")
        
        for name, baseline in baselines.items():
            if name != 'original':
                print(f"{baseline['description']:<30} {baseline['PESQ']:>8.3f} {baseline['STOI']:>8.3f} {baseline['MCD']:>8.2f} {baseline['SI-SDR']:>8.2f}")
        
        # Quality assessment
        print("\n" + "=" * 70)
        print("QUALITY ASSESSMENT")
        print("=" * 70)
        
        if our_pesq >= 3.5:
            quality = "HIGH - Suitable for production TTS"
        elif our_pesq >= 2.5:
            quality = "MEDIUM - Acceptable for training, may need refinement"
        else:
            quality = "LOW - Consider increasing iterations or adjusting parameters"
        
        print(f"Overall Quality: {quality}")
        print(f"Recommendation: ", end="")
        if our_pesq >= 3.0 and our_stoi >= 0.85:
            print("✓ Ready for Flow Matching training")
        else:
            print("⚠ Consider improving atom fitting before training")
        
        # Save results
        output_file = args.output_file or str(PROJECT_ROOT / "logs" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "config": {
                "data_dir": str(args.data_dir),
                "max_samples": args.max_samples,
                "sample_rate": args.sample_rate,
            },
            "summary": summary,
            "baselines": baselines,
            "samples": results[:100],  # Save first 100 samples
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        print("=" * 70)


if __name__ == "__main__":
    main()
