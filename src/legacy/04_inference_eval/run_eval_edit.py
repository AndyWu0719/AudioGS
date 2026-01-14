"""
Editability Evaluation Script.

Evaluate atom editing capabilities (pitch shift, time stretch).

Usage:
    python scripts/04_inference_eval/run_eval_edit.py \
        --checkpoint logs/flow/best.pt \
        --data_dir data/atoms/LibriTTS_R/train/train-clean-100 \
        --num_samples 20
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

from tools.inference import AGSInferenceEngine
from tools.metrics import compute_mss_loss, compute_f0_error, compute_duration_error


def parse_args():
    parser = argparse.ArgumentParser(description="Editability Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True, help="Original audio directory")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def evaluate_reconstruction(engine, audio_paths, output_dir):
    """Evaluate reconstruction quality via invert_audio."""
    print("\n[1] Reconstruction Quality...")
    
    mss_losses = []
    
    for i, audio_path in enumerate(tqdm(audio_paths, desc="Inverting")):
        try:
            # Invert audio
            atoms = engine.invert_audio(audio_path, steps=2000)
            
            # Load original
            wf_orig, sr = torchaudio.load(audio_path)
            if wf_orig.shape[0] > 1:
                wf_orig = wf_orig.mean(dim=0)
            else:
                wf_orig = wf_orig.squeeze(0)
            if sr != engine.sample_rate:
                wf_orig = torchaudio.functional.resample(wf_orig, sr, engine.sample_rate)
            
            duration = len(wf_orig) / engine.sample_rate
            
            # Render back
            wf_recon = engine.atoms_to_audio(atoms, duration)
            
            # Compute MSS loss
            mss = compute_mss_loss(wf_orig, wf_recon)
            mss_losses.append(mss)
            
            # Save for inspection
            recon_path = output_dir / "reconstruction" / f"recon_{i:03d}.wav"
            engine.save_audio(wf_recon, str(recon_path))
            
        except Exception as e:
            print(f"  [Error] {audio_path}: {e}")
            continue
    
    return {
        "mean": float(np.mean(mss_losses)) if mss_losses else float('nan'),
        "std": float(np.std(mss_losses)) if mss_losses else float('nan'),
        "samples": len(mss_losses),
    }


def evaluate_pitch_shift(engine, atoms_list, durations, output_dir, shift=2):
    """Evaluate pitch shifting (+2 semitones)."""
    print(f"\n[2] Pitch Shift (+{shift} semitones)...")
    
    errors = []
    
    for i, (atoms, dur) in enumerate(tqdm(zip(atoms_list, durations), desc="Pitch Shift", total=len(atoms_list))):
        try:
            # Original audio
            audio_orig = engine.atoms_to_audio(atoms, dur)
            orig_path = output_dir / "pitch" / f"orig_{i:03d}.wav"
            engine.save_audio(audio_orig, str(orig_path))
            
            # Pitch shifted
            atoms_shifted = engine.edit_atoms(atoms, pitch_shift=shift)
            audio_shifted = engine.atoms_to_audio(atoms_shifted, dur)
            shifted_path = output_dir / "pitch" / f"shifted_{i:03d}.wav"
            engine.save_audio(audio_shifted, str(shifted_path))
            
            # Compute F0 error
            f0_result = compute_f0_error(str(orig_path), str(shifted_path), expected_shift=shift)
            errors.append(f0_result['mean_error'])
            
        except Exception as e:
            print(f"  [Error] sample {i}: {e}")
            continue
    
    return {
        "shift_semitones": shift,
        "mean_error": float(np.mean(errors)) if errors else float('nan'),
        "std_error": float(np.std(errors)) if errors else float('nan'),
        "samples": len(errors),
    }


def evaluate_time_stretch(engine, atoms_list, durations, output_dir, rate=1.2):
    """Evaluate time stretching (1.2x = 20% faster)."""
    print(f"\n[3] Time Stretch ({rate}x)...")
    
    errors = []
    
    for i, (atoms, dur) in enumerate(tqdm(zip(atoms_list, durations), desc="Time Stretch", total=len(atoms_list))):
        try:
            # Original audio
            audio_orig = engine.atoms_to_audio(atoms, dur)
            orig_path = output_dir / "speed" / f"orig_{i:03d}.wav"
            engine.save_audio(audio_orig, str(orig_path))
            
            # Time stretched
            atoms_stretched = engine.edit_atoms(atoms, speed_rate=rate)
            new_dur = dur / rate
            audio_stretched = engine.atoms_to_audio(atoms_stretched, new_dur)
            stretched_path = output_dir / "speed" / f"stretched_{i:03d}.wav"
            engine.save_audio(audio_stretched, str(stretched_path))
            
            # Compute duration error
            dur_result = compute_duration_error(str(orig_path), str(stretched_path), expected_rate=rate)
            errors.append(dur_result['error_percent'])
            
        except Exception as e:
            print(f"  [Error] sample {i}: {e}")
            continue
    
    return {
        "speed_rate": rate,
        "mean_error_percent": float(np.mean(errors)) if errors else float('nan'),
        "std_error_percent": float(np.std(errors)) if errors else float('nan'),
        "samples": len(errors),
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Editability Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Samples: {args.num_samples}")
    print("=" * 60)
    
    # Initialize engine
    engine = AGSInferenceEngine(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    
    # Output directory
    output_dir = PROJECT_ROOT / "outputs" / f"eval_edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reconstruction").mkdir(exist_ok=True)
    (output_dir / "pitch").mkdir(exist_ok=True)
    (output_dir / "speed").mkdir(exist_ok=True)
    
    # Collect audio files
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else PROJECT_ROOT / args.data_dir
    
    # If data_dir contains .pt files, load atoms; if .wav files, collect audio paths
    pt_files = list(data_dir.rglob("*.pt"))[:args.num_samples]
    wav_files = list(data_dir.rglob("*.wav"))[:args.num_samples]
    
    results = {}
    
    # 1. Reconstruction (using audio files if available)
    if wav_files:
        recon_results = evaluate_reconstruction(engine, [str(f) for f in wav_files], output_dir)
        results["reconstruction"] = recon_results
        print(f"  MSS Loss: {recon_results['mean']:.4f} ± {recon_results['std']:.4f}")
    
    # For pitch/speed tests, we need atoms
    if pt_files:
        atoms_list = []
        durations = []
        
        for pt_file in pt_files:
            data = torch.load(pt_file, weights_only=False)
            atoms_list.append(data['atoms'])
            durations.append(data['audio_duration'])
        
        # 2. Pitch shift
        pitch_results = evaluate_pitch_shift(engine, atoms_list, durations, output_dir, shift=2)
        results["pitch_shift"] = pitch_results
        print(f"  F0 Error: {pitch_results['mean_error']:.2f} ± {pitch_results['std_error']:.2f} semitones")
        
        # 3. Time stretch
        speed_results = evaluate_time_stretch(engine, atoms_list, durations, output_dir, rate=1.2)
        results["time_stretch"] = speed_results
        print(f"  Duration Error: {speed_results['mean_error_percent']:.2f}%")
    
    # Save results
    output_path = args.output or str(output_dir / "results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EDITABILITY SUMMARY")
    print("=" * 60)
    if "reconstruction" in results:
        print(f"Reconstruction MSS: {results['reconstruction']['mean']:.4f}")
    if "pitch_shift" in results:
        print(f"Pitch Shift Error: {results['pitch_shift']['mean_error']:.2f} semitones")
    if "time_stretch" in results:
        print(f"Time Stretch Error: {results['time_stretch']['mean_error_percent']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
