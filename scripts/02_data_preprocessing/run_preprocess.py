"""
Dataset Generation Script for Audio Gaussian Splatting Phase 2.

Processes LibriTTS-R train-clean-100 into atom parameter dataset using
multi-GPU processing with the CUDA extension for maximum speed.

VRAM Usage (Measured):
    - Per sample (short audio): ~500MB
    - Per sample (long audio 10s): ~2-3GB peak
    - RTX 4090 (24GB): 16 workers per GPU = ~8GB average, 16GB headroom

Parallelism:
    - 4 GPUs x 16 workers = 64 parallel processes
    - Expected speed: ~2s per sample = ~18 hours for 33K files

Usage:
    conda activate qwen2_CALM
    python scripts/preprocess_dataset.py --output_dir data/atoms_v1

    # Resume after interruption:
    python scripts/preprocess_dataset.py --resume --output_dir data/atoms_v1

Expected output: ~28,000 .pt files containing atom parameters.
"""



import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torchaudio
import yaml
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/02_data_prep -> scripts -> GS-TS
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

from models.atom import AudioGSModel
from cuda_gabor import get_cuda_gabor_renderer
from losses.spectral_loss import CombinedAudioLoss
from utils.density_control import AdaptiveDensityController, rebuild_optimizer_from_model


# ============================================================
# Configuration
# ============================================================

def load_config_from_yaml(config_path: str) -> Dict:
    """
    Load configuration from AudioGS_config.yaml and flatten for preprocessing.
    
    This ensures consistency between training and preprocessing.
    """
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)
    
    # Flatten the nested config structure
    config = {
        # Data
        "sample_rate": yaml_config.get("data", {}).get("sample_rate", 24000),
        "max_audio_length": yaml_config.get("data", {}).get("max_audio_length", 10.0),
        "min_audio_length": 1.0,  # Not in yaml, use default
        
        # Model
        "initial_num_atoms": yaml_config.get("model", {}).get("initial_num_atoms", 4000),
        "max_num_atoms": yaml_config.get("model", {}).get("max_num_atoms", 20000),
        
        # Training
        "num_iters": yaml_config.get("training", {}).get("max_iters", 10000),
        "lr_amplitude": yaml_config.get("training", {}).get("lr_amplitude", 0.01),
        "lr_position": yaml_config.get("training", {}).get("lr_position", 0.0001),
        "lr_frequency": yaml_config.get("training", {}).get("lr_frequency", 0.01),
        "lr_sigma": yaml_config.get("training", {}).get("lr_sigma", 0.01),
        "lr_phase": yaml_config.get("training", {}).get("lr_phase", 0.02),
        "lr_chirp": yaml_config.get("training", {}).get("lr_chirp", 0.002),
        
        # Loss weights
        "spectral_weight": yaml_config.get("loss", {}).get("spectral_weight", 1.5),
        "mel_weight": yaml_config.get("loss", {}).get("mel_weight", 0.2),
        "time_weight": yaml_config.get("loss", {}).get("time_domain_weight", 0.5),
        "pre_emp_weight": yaml_config.get("loss", {}).get("pre_emp_weight", 20.0),
        "amp_reg_weight": yaml_config.get("loss", {}).get("amp_reg_weight", 0.01),
        
        # Density control
        # Note: densify timing is handled by 3-stage training (Stage 2: 20%-70%)
        "grad_threshold": yaml_config.get("density_control", {}).get("grad_threshold", 0.0002),
        "sigma_split_threshold": yaml_config.get("density_control", {}).get("sigma_split_threshold", 0.01),
        "prune_amplitude_threshold": yaml_config.get("density_control", {}).get("prune_amplitude_threshold", 0.001),
        "densification_interval": yaml_config.get("density_control", {}).get("densification_interval", 100),
        
        # Preprocessing specific
        "loss_threshold": 3.0,      # Discard if final loss > threshold
        "high_freq_ratio": 0.2,     # 20% high-frequency atoms
    }
    
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate atom dataset from LibriTTS-R")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/AudioGS_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None,  # Will use config if not specified
        help="Path to train-clean-100 (overrides config)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/atoms/LibriTTS_R/train/train-clean-100",
        help="Output directory for .pt files"
    )
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--num_workers_per_gpu", type=int, default=24, 
                        help="Workers per GPU (24 recommended for RTX 4090 24GB)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (for testing)")
    parser.add_argument("--resume", action="store_true", help="Skip already processed files")
    return parser.parse_args()


def collect_audio_files(data_path: str) -> List[Dict]:
    """Collect all .wav files with transcripts and speaker IDs."""
    files = []
    data_path = Path(data_path)
    
    # LibriTTS-R structure: speaker_id/chapter_id/speaker_chapter_utterance.wav
    for wav_path in data_path.rglob("*.wav"):
        # Parse speaker ID from path
        parts = wav_path.relative_to(data_path).parts
        if len(parts) >= 2:
            speaker_id = parts[0]
        else:
            speaker_id = "unknown"
        
        # Look for transcript (normalized.txt)
        txt_path = wav_path.with_suffix(".normalized.txt")
        if not txt_path.exists():
            txt_path = wav_path.with_suffix(".original.txt")
        
        transcript = ""
        if txt_path.exists():
            transcript = txt_path.read_text().strip()
        
        # Preserve relative path for directory structure
        relative_path = wav_path.relative_to(data_path)
        
        files.append({
            "audio_path": str(wav_path),
            "speaker_id": speaker_id,
            "transcript": transcript,
            "file_id": wav_path.stem,
            "relative_path": str(relative_path),  # e.g., "103/1241/103_1241_000001_000000.wav"
        })
    
    return files


def load_audio(audio_path: str, target_sr: int, max_length: float) -> Optional[torch.Tensor]:
    """Load and preprocess audio file."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        
        # Trim/pad to max length
        max_samples = int(max_length * target_sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        
        return waveform
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None


def fit_atoms_to_audio(
    waveform: torch.Tensor,
    device: torch.device,
    config: Dict,
    renderer,
    loss_fn,
    verbose: bool = False,
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Fit Gabor atoms to audio waveform.
    
    Args:
        waveform: Audio waveform tensor
        device: CUDA device
        config: Configuration dict
        renderer: Pre-initialized CUDA Gabor renderer (reused across files)
        loss_fn: Pre-initialized loss function (reused across files)
        verbose: Enable verbose output
    
    Returns:
        atoms: [N, 6] tensor (tau, omega, sigma, amplitude, phi, gamma) or None if failed
        final_loss: Final reconstruction loss
    """
    sample_rate = config["sample_rate"]
    num_samples = len(waveform)
    audio_duration = num_samples / sample_rate
    
    # Move waveform to device
    gt_waveform = waveform.to(device)
    
    # Initialize model with F0-guided initialization
    model = AudioGSModel(
        num_atoms=config["initial_num_atoms"],
        sample_rate=sample_rate,
        audio_duration=audio_duration,
        device=device,
    )
    model.initialize_from_audio(gt_waveform, use_f0_init=True)
    
    # Density controller
    density_controller = AdaptiveDensityController(
        grad_threshold=config["grad_threshold"],
        sigma_split_threshold=config["sigma_split_threshold"],
        prune_amplitude_threshold=config["prune_amplitude_threshold"],
        max_num_atoms=config["max_num_atoms"],
    )
    
    # Multi-stage training configuration
    total_iters = config["num_iters"]
    stage1_end = int(total_iters * 0.2)    # 0-20%: Structure finding
    stage2_end = int(total_iters * 0.7)    # 20-70%: Joint optimization
    # Stage 3: 70-100%: Polishing
    
    final_loss = float('inf')
    optimizer = None
    current_stage = 0
    
    for iteration in range(total_iters):
        # Determine current stage
        if iteration < stage1_end:
            stage = 1
        elif iteration < stage2_end:
            stage = 2
        else:
            stage = 3
        
        # Stage transition: reset optimizer and configure parameters
        if stage != current_stage:
            optimizer = _configure_stage(
                model, stage, config, loss_fn,
                densify_active=(stage == 2)
            )
            current_stage = stage
        
        # Forward pass
        amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
        pred_waveform = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
        
        # Loss with stage-specific weights
        if stage == 3:
            # Polish stage: emphasize STFT for phase alignment
            loss, loss_dict = loss_fn(
                pred_waveform, gt_waveform,
                model_amplitude=amplitude,
                model_sigma=sigma,
                sigma_diversity_weight=0.0,  # No regularization in polish
            )
        else:
            loss, loss_dict = loss_fn(
                pred_waveform, gt_waveform,
                model_amplitude=amplitude,
                model_sigma=sigma,
                sigma_diversity_weight=0.001,
            )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if stage != 1:  # Only accumulate gradients in stage 2+
            model.accumulate_gradients()
        
        # Gradient clipping (stricter in stage 3)
        max_norm = 0.5 if stage == 3 else 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()
        final_loss = loss_dict["total"]
        
        # Density control (only in stage 2)
        if stage == 2:
            density_controller.update_thresholds(final_loss)
            # Stage 2 is the densify window (20%-70%), so just check interval
            if iteration % config["densification_interval"] == 0:
                density_controller.densify_and_prune(model, optimizer)
    
    # Extract final atom parameters
    with torch.no_grad():
        amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
        
        # Normalize tau to [0, 1]
        tau_normalized = tau / audio_duration
        tau_normalized = tau_normalized.clamp(0, 1)
        
        # Stack: [N, 6] in order (tau, omega, sigma, amplitude, phi, gamma)
        atoms = torch.stack([
            tau_normalized,  # Normalized to [0, 1]
            omega,           # Hz
            sigma,           # seconds
            amplitude,       # Amplitude
            phi,             # Phase (radians)
            gamma,           # Chirp rate
        ], dim=1)
        
        # Sort by tau (ascending) - makes it a sequence!
        sorted_indices = torch.argsort(atoms[:, 0])
        atoms = atoms[sorted_indices]
    
    return atoms.cpu(), final_loss, audio_duration


def _configure_stage(
    model: AudioGSModel,
    stage: int,
    config: Dict,
    loss_fn: CombinedAudioLoss,
    densify_active: bool = False,
) -> torch.optim.Optimizer:
    """
    Configure optimizer and parameter freezing for each training stage.
    
    Stage 1 (Structure): Train τ, ω, σ. Freeze amp, φ.
    Stage 2 (Joint): Train ALL. Density control active.
    Stage 3 (Polish): Train amp, φ. Freeze τ, ω, σ. Update loss weights.
    """
    if stage == 1:
        # Freeze amplitude and phase
        model._amplitude_logit.requires_grad = False
        model._phi.requires_grad = False
        model._tau.requires_grad = True
        model._omega_logit.requires_grad = True
        model._sigma_logit.requires_grad = True
        model._gamma.requires_grad = True
        
        param_groups = [
            {"params": [model._tau], "lr": config["lr_position"]},
            {"params": [model._omega_logit], "lr": config["lr_frequency"]},
            {"params": [model._sigma_logit], "lr": config["lr_sigma"]},
            {"params": [model._gamma], "lr": config["lr_chirp"]},
        ]
        
    elif stage == 2:
        # Unfreeze all parameters
        model._amplitude_logit.requires_grad = True
        model._phi.requires_grad = True
        model._tau.requires_grad = True
        model._omega_logit.requires_grad = True
        model._sigma_logit.requires_grad = True
        model._gamma.requires_grad = True
        
        param_groups = model.get_optimizer_param_groups(config)
        
    else:  # stage == 3
        # Freeze structure, train amplitude and phase
        model._amplitude_logit.requires_grad = True
        model._phi.requires_grad = True
        model._tau.requires_grad = False
        model._omega_logit.requires_grad = False
        model._sigma_logit.requires_grad = False
        model._gamma.requires_grad = False
        
        param_groups = [
            {"params": [model._amplitude_logit], "lr": config["lr_amplitude"] * 0.5},
            {"params": [model._phi], "lr": config["lr_phase"] * 0.5},
        ]
        
        # Update loss weights for polishing
        loss_fn.stft_weight = 2.0   # Emphasize linear magnitude
        loss_fn.mel_weight = 0.2    # Reduce perceptual
        loss_fn.time_weight = 0.5   # Force phase alignment
    
    return torch.optim.Adam(param_groups)


def process_single_file(
    file_info: Dict,
    output_dir: Path,
    device: torch.device,
    config: Dict,
    renderer,
    loss_fn,
    resume: bool = False,
) -> Dict:
    """Process a single audio file."""
    file_id = file_info["file_id"]
    
    # Preserve directory structure: speaker_id/chapter_id/filename.pt
    relative_path = Path(file_info["relative_path"])
    output_subdir = output_dir / relative_path.parent  # speaker_id/chapter_id/
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_path = output_subdir / f"{file_id}.pt"
    
    # Skip if already processed
    if resume and output_path.exists():
        return {"status": "skipped", "file_id": file_id}
    
    # Load audio
    waveform = load_audio(
        file_info["audio_path"],
        config["sample_rate"],
        config["max_audio_length"],
    )
    
    if waveform is None:
        return {"status": "load_error", "file_id": file_id}
    
    # Check minimum length
    if len(waveform) < config["min_audio_length"] * config["sample_rate"]:
        return {"status": "too_short", "file_id": file_id}
    
    # Fit atoms (renderer and loss_fn are reused across files)
    try:
        atoms, final_loss, audio_duration = fit_atoms_to_audio(
            waveform, device, config, renderer, loss_fn
        )
    except Exception as e:
        return {"status": "fit_error", "file_id": file_id, "error": str(e)}
    
    # Check loss threshold
    if final_loss > config["loss_threshold"]:
        return {"status": "high_loss", "file_id": file_id, "loss": final_loss}
    
    # Save result
    result = {
        "atoms": atoms,                      # [N, 6] tensor
        "transcript": file_info["transcript"],
        "speaker_id": file_info["speaker_id"],
        "audio_duration": audio_duration,
        "num_atoms": atoms.shape[0],
        "final_loss": final_loss,
        "file_id": file_id,
        "source_path": file_info["audio_path"],
    }
    
    torch.save(result, output_path)
    
    return {
        "status": "success", 
        "file_id": file_id, 
        "loss": final_loss,
        "num_atoms": atoms.shape[0],
    }


def worker_fn(
    gpu_id: int,
    file_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: Path,
    config: Dict,
    resume: bool,
):
    """Worker function for multiprocessing."""
    # Set device
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    # Suppress CUDA gabor print
    import io
    import contextlib
    
    # =========================================================
    # OPTIMIZATION: Initialize heavy objects ONCE per worker
    # (not per file) to avoid CUDA memory allocation overhead
    # =========================================================
    
    with contextlib.redirect_stdout(io.StringIO()):
        # Initialize CUDA renderer once
        renderer = get_cuda_gabor_renderer(sample_rate=config["sample_rate"])
        
        # Initialize loss function once
        loss_fn = CombinedAudioLoss(
            sample_rate=config["sample_rate"],
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_lengths=[512, 1024, 2048],
            stft_weight=config["spectral_weight"],
            mel_weight=config["mel_weight"],
            time_weight=config["time_weight"],
            amp_reg_weight=config["amp_reg_weight"],
            pre_emp_weight=config["pre_emp_weight"],
        ).to(device)
        
        # NOTE: torch.compile disabled - causes long startup delay
        # try:
        #     loss_fn = torch.compile(loss_fn)
        # except Exception:
        #     pass  # Fallback to eager mode if compile fails
    
    # Process files from queue
    while True:
        try:
            file_info = file_queue.get(timeout=1)
        except:
            break
        
        if file_info is None:
            break
        
        # Redirect stdout to suppress prints
        with contextlib.redirect_stdout(io.StringIO()):
            result = process_single_file(
                file_info, output_dir, device, config, 
                renderer, loss_fn, resume
            )
        
        result_queue.put(result)



def main():
    args = parse_args()
    
    # Load config from YAML
    config_path = PROJECT_ROOT / args.config
    print(f"[Config] Loading from {args.config}")
    config = load_config_from_yaml(str(config_path))
    
    # Get data path from args or config
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = config.get("dataset_path", "") + "/train/train-clean-100"
        # Fallback if dataset_path not in flattened config
        if not data_path or data_path == "/train/train-clean-100":
            # Load raw yaml to get dataset_path
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
            data_path = yaml_config.get("data", {}).get("dataset_path", "") + "/train/train-clean-100"
    
    print("=" * 70)
    print("Audio Gaussian Splatting - Dataset Generation")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Data path: {data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Num GPUs: {args.num_gpus}")
    print(f"Iterations: {config['num_iters']}")
    print(f"Sample rate: {config['sample_rate']}")
    print("=" * 70)
    
    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect files
    print("Collecting audio files...")
    files = collect_audio_files(data_path)
    print(f"Found {len(files)} audio files")
    
    if args.max_samples:
        files = files[:args.max_samples]
        print(f"Limited to {len(files)} samples for testing")
    
    # Save config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Setup multiprocessing
    mp.set_start_method("spawn", force=True)
    
    file_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add files to queue
    for f in files:
        file_queue.put(f)
    
    # Add poison pills
    for _ in range(args.num_gpus * args.num_workers_per_gpu):
        file_queue.put(None)
    
    # Start workers
    workers = []
    for gpu_id in range(args.num_gpus):
        for _ in range(args.num_workers_per_gpu):
            p = mp.Process(
                target=worker_fn,
                args=(gpu_id, file_queue, result_queue, output_dir, config, args.resume),
            )
            p.start()
            workers.append(p)
    
    # Collect results with progress bar
    stats = {"success": 0, "skipped": 0, "high_loss": 0, "error": 0}
    
    pbar = tqdm(total=len(files), desc="Processing")
    processed = 0
    timeout_count = 0
    max_timeouts = 5  # Allow multiple timeouts before giving up
    
    while processed < len(files):
        try:
            # Long timeout for worker initialization (CUDA context takes time)
            result = result_queue.get(timeout=300)
            processed += 1
            timeout_count = 0  # Reset on success
            pbar.update(1)
            
            status = result["status"]
            if status == "success":
                stats["success"] += 1
                pbar.set_postfix({
                    "ok": stats["success"], 
                    "loss": f"{result.get('loss', 0):.2f}"
                })
            elif status == "skipped":
                stats["skipped"] += 1
            elif status == "high_loss":
                stats["high_loss"] += 1
            else:
                stats["error"] += 1
                
        except Exception as e:
            timeout_count += 1
            # Check if workers are still alive
            alive_workers = sum(1 for p in workers if p.is_alive())
            print(f"\n[Warning] Queue timeout ({timeout_count}/{max_timeouts}), {alive_workers} workers alive")
            
            if timeout_count >= max_timeouts or alive_workers == 0:
                print(f"\n[Error] Too many timeouts or no workers alive, stopping...")
                break

    
    pbar.close()
    
    # Wait for workers
    for p in workers:
        p.join(timeout=10)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Success: {stats['success']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"High Loss (discarded): {stats['high_loss']}")
    print(f"Errors: {stats['error']}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Save stats
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
