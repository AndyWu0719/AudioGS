#!/usr/bin/env python
"""
Encoder Evaluation Script for GaborGridEncoder

Evaluates a trained encoder checkpoint by running real-time inference
on validation audio and computing quality metrics.

Usage:
    python scripts/03_encoder_eval/run_encoder_eval.py \
        --checkpoint logs/encoder_v1/best_model.pt \
        --data_path data/raw/LibriTTS_R/dev-clean \
        --num_samples 50
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
import yaml

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

from models.encoder import build_encoder

# Quality metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

# CUDA renderer
try:
    from cuda_gabor import get_cuda_gabor_renderer
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GaborGridEncoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config override (else from ckpt)")
    parser.add_argument("--data_path", type=str, default="data/raw/LibriTTS_R/dev-clean")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None, help="Save sample audio")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def compute_si_sdr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute SI-SDR."""
    min_len = min(len(pred), len(target))
    pred = pred[:min_len].astype(np.float64)
    target = target[:min_len].astype(np.float64)
    
    pred = pred - np.mean(pred)
    target = target - np.mean(target)
    
    dot = np.dot(pred, target)
    s_target = dot * target / (np.dot(target, target) + 1e-8)
    e_noise = pred - s_target
    
    return float(10 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8) + 1e-8
    ))


def compute_pesq_score(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Compute PESQ (requires 16kHz)."""
    if not PESQ_AVAILABLE:
        return float('nan')
    try:
        if sr != 16000:
            ref = torchaudio.functional.resample(
                torch.from_numpy(ref), sr, 16000
            ).numpy()
            deg = torchaudio.functional.resample(
                torch.from_numpy(deg), sr, 16000
            ).numpy()
        min_len = min(len(ref), len(deg))
        return pesq(16000, ref[:min_len], deg[:min_len], 'wb')
    except:
        return float('nan')


def compute_stoi_score(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Compute STOI."""
    if not STOI_AVAILABLE:
        return float('nan')
    try:
        min_len = min(len(ref), len(deg))
        return stoi(ref[:min_len], deg[:min_len], sr, extended=False)
    except:
        return float('nan')


class EncoderEvaluator:
    """Evaluator for GaborGridEncoder."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_override: str = None,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        print(f"[Eval] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Config
        if config_override:
            with open(config_override, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = checkpoint.get('config', {})
        
        self.config = config
        self.sample_rate = config.get('data', {}).get('sample_rate', 24000)
        
        # Build encoder
        self.encoder = build_encoder(config).to(self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.encoder.eval()
        
        print(f"[Eval] Encoder loaded: {sum(p.numel() for p in self.encoder.parameters()):,} params")
        
        # Renderer
        if not RENDERER_AVAILABLE:
            raise RuntimeError("CUDA renderer required!")
        self.renderer = get_cuda_gabor_renderer(sample_rate=self.sample_rate)
        
    def load_audio(self, path: str, max_length: float = 10.0) -> torch.Tensor:
        """Load and preprocess audio."""
        waveform, sr = torchaudio.load(path)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        # Resample
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Truncate
        max_samples = int(max_length * self.sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        
        return waveform
    
    @torch.no_grad()
    def reconstruct(self, audio: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio through encoder."""
        # Add batch dim
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        audio = audio.to(self.device)
        num_samples = audio.shape[-1]
        
        # Forward
        enc_output = self.encoder(audio)
        
        # Render
        recon = self.renderer(
            enc_output['amplitude'][0].contiguous(),
            enc_output['tau'][0].contiguous(),
            enc_output['omega'][0].contiguous(),
            enc_output['sigma'][0].contiguous(),
            enc_output['phi'][0].contiguous(),
            enc_output['gamma'][0].contiguous(),
            num_samples,
        )
        
        # Metadata
        active_ratio = (enc_output['existence_prob'][0] > 0.5).float().mean().item()
        num_atoms = enc_output['amplitude'].shape[-1]
        
        return recon, {'active_ratio': active_ratio, 'num_atoms': num_atoms}
    
    def evaluate_file(self, audio_path: str) -> Dict:
        """Evaluate single file."""
        # Load
        original = self.load_audio(audio_path)
        
        # Reconstruct
        reconstructed, meta = self.reconstruct(original)
        
        # Convert to numpy
        orig_np = original.numpy()
        recon_np = reconstructed.cpu().numpy()
        
        # Normalize for metrics
        orig_np = orig_np / (np.abs(orig_np).max() + 1e-8)
        recon_np = recon_np / (np.abs(recon_np).max() + 1e-8)
        
        # Metrics
        return {
            'file': Path(audio_path).name,
            'duration': len(original) / self.sample_rate,
            'num_atoms': meta['num_atoms'],
            'active_ratio': meta['active_ratio'],
            'si_sdr': compute_si_sdr(recon_np, orig_np),
            'pesq': compute_pesq_score(orig_np, recon_np, self.sample_rate),
            'stoi': compute_stoi_score(orig_np, recon_np, self.sample_rate),
        }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("GaborGridEncoder Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data path: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print("=" * 60)
    
    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Evaluator
    evaluator = EncoderEvaluator(
        checkpoint_path=args.checkpoint,
        config_override=args.config,
        device=device,
    )
    
    # Collect audio files
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    
    audio_files = list(data_path.rglob("*.wav"))[:args.num_samples]
    print(f"Found {len(audio_files)} files")
    
    # Evaluate
    results = []
    for audio_path in tqdm(audio_files, desc="Evaluating"):
        try:
            metrics = evaluator.evaluate_file(str(audio_path))
            results.append(metrics)
        except Exception as e:
            print(f"[Error] {audio_path}: {e}")
    
    # Aggregate
    if results:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        metrics_to_agg = ['si_sdr', 'pesq', 'stoi', 'active_ratio']
        for name in metrics_to_agg:
            values = [r[name] for r in results if not np.isnan(r.get(name, float('nan')))]
            if values:
                print(f"{name.upper():>12}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        print("=" * 60)
        
        # Quality assessment
        avg_pesq = np.mean([r['pesq'] for r in results if not np.isnan(r['pesq'])])
        if not np.isnan(avg_pesq):
            if avg_pesq >= 3.5:
                print("Quality: HIGH - Production ready")
            elif avg_pesq >= 2.5:
                print("Quality: MEDIUM - May need refinement")
            else:
                print("Quality: LOW - Consider more training")
        
        # Save results
        output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "logs" / "eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'checkpoint': args.checkpoint,
                'num_samples': len(results),
                'summary': {
                    name: {'mean': float(np.nanmean([r[name] for r in results])),
                           'std': float(np.nanstd([r[name] for r in results]))}
                    for name in metrics_to_agg
                },
                'samples': results[:20],  # First 20
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
