#!/usr/bin/env python
"""
WandB Sweep Agent for Atom Fitting Hyperparameter Search.
This script is called by wandb sweep agent. It reads hyperparameters from
wandb.config and runs a single atom fitting trial on representative audio samples.
Key Features:
1. Logical dependency: densify_until_iter = max_iters * densify_until_ratio
2. Composite score: PESQ - 0.2 * (num_atoms / 10000) to balance quality vs efficiency
Usage:
    # First, create the sweep
    wandb sweep configs/sweep_atom_fitting.yaml
    
    # Then run the agent
    wandb agent <sweep_id>
"""
import sys
import os
from pathlib import Path
# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/sweeps -> scripts -> GS-TS
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))
import argparse
import random
import time
import torch
import torchaudio
import wandb
import numpy as np
from tqdm import tqdm
from models.atom import AudioGSModel
from losses.spectral_loss import CombinedAudioLoss
from utils.density_control import AdaptiveDensityController, rebuild_optimizer_from_model
from torch.optim.lr_scheduler import LambdaLR

# Optional quality metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("[Warning] pesq not available, using MSS as proxy")

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

def compute_composite_score(pesq_score: float, num_atoms: int) -> float:
    """
    Compute composite score that balances quality vs efficiency.
    
    Formula: Score = PESQ - 0.2 * (num_atoms / 10000)
    
    Examples:
        PESQ 4.0, Atoms 8000  -> Score = 4.0 - 0.16 = 3.84
        PESQ 4.1, Atoms 16000 -> Score = 4.1 - 0.32 = 3.78
        Result: Lower atom count wins when PESQ is similar
    """
    atom_penalty = 0.2 * (num_atoms / 10000.0)
    return pesq_score - atom_penalty

def get_test_audios(data_path: str, num_samples: int = 3) -> list:
    """Get a few representative audio files for testing."""
    data_path = Path(data_path)
    
    # Find audio files
    audio_files = list(data_path.rglob("*.wav"))[:100]
    
    if len(audio_files) < num_samples:
        raise ValueError(f"Not enough audio files in {data_path}")
    
    # Select diverse samples
    random.seed(42)
    selected = random.sample(audio_files, num_samples)
    
    return selected

def run_trial(config: dict) -> dict:
    """Run a single atom fitting trial with the given config."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = 24000
    
    if not RENDERER_AVAILABLE:
        raise RuntimeError("CUDA renderer not available")
    
    renderer = get_cuda_gabor_renderer(sample_rate=sample_rate)
    
    # Get test audio files
    data_path = config.get('data_path', '/data0/determined/users/andywu/GS-TS/data/raw/LibriTTS_R/train/train-clean-100')
    test_audios = get_test_audios(data_path, num_samples=3)
    
    # ========================================
    # FLEXIBLE TRAINING (Searchable Parameters)
    # ========================================
    max_iters = config['max_iters']
    
    densify_from_iter = config.get('densify_from_iter', 500)
    densify_until_ratio = config.get('densify_until_ratio', 0.7)
    densify_until_iter = int(max_iters * densify_until_ratio)
    densification_interval = config.get('densification_interval', 100)
    
    # Soft LR decay settings
    lr_decay_start_ratio = config.get('lr_decay_start_ratio', 0.8)
    lr_decay_factor = config.get('lr_decay_factor', 0.1)
    
    print(f"[Sweep] Flexible training: max={max_iters}, densify=[{densify_from_iter}, {densify_until_iter}] (ratio={densify_until_ratio})")
    
    all_metrics = {
        'pesq': [], 'stoi': [], 'mss_loss': [],
        'final_loss': [], 'num_atoms': [], 'time_per_audio': []
    }
    
    for audio_idx, audio_path in enumerate(test_audios):
        print(f"[Sweep] Processing audio {audio_idx + 1}/{len(test_audios)}: {audio_path.name}")
        
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        
        # Truncate to max 5 seconds for speed
        max_samples = 5 * sample_rate
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        
        waveform = waveform.to(device)
        audio_duration = len(waveform) / sample_rate
        num_samples_audio = len(waveform)
        
        start_time = time.time()
        
        # Initialize model
        model = AudioGSModel(
            num_atoms=config['initial_num_atoms'],
            sample_rate=sample_rate,
            audio_duration=audio_duration,
            device=device,
        )
        model.initialize_from_audio(waveform)
        
        # Loss function - with phase_weight for PESQ improvement
        loss_fn = CombinedAudioLoss(
            sample_rate=sample_rate,
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_lengths=[512, 1024, 2048],
            stft_weight=config.get('spectral_weight', 1.0),
            mel_weight=config.get('mel_weight', 0.5),
            time_weight=config.get('time_domain_weight', 0.1),  # Aligned with config
            phase_weight=config.get('phase_weight', 0.8),  # Phase alignment
        ).to(device)
        
        # Optimizer
        lr_config = {
            'lr_amplitude': config['lr_amplitude'],
            'lr_position': config['lr_position'],
            'lr_frequency': config['lr_frequency'],
            'lr_sigma': config.get('lr_sigma', 0.01),
            'lr_phase': config.get('lr_phase', 0.02),
            'lr_chirp': config.get('lr_chirp', 0.002),
        }
        optimizer = rebuild_optimizer_from_model(model, torch.optim.Adam, lr_config)
        
        # Density controller
        density_controller = AdaptiveDensityController(
            grad_threshold=config['grad_threshold'],
            sigma_split_threshold=config.get('sigma_split_threshold', 0.003),  # Aligned with config
            prune_amplitude_threshold=config.get('prune_amplitude_threshold', 0.0005),
            max_num_atoms=config['max_num_atoms'],
        )
        
        # Soft LR decay scheduler (no hard freezing)
        decay_start = int(max_iters * lr_decay_start_ratio)
        def lr_lambda(iteration: int) -> float:
            if iteration < decay_start:
                return 1.0
            progress = (iteration - decay_start) / (max_iters - decay_start + 1e-8)
            return 1.0 - progress * (1.0 - lr_decay_factor)
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        # Training loop
        final_loss = float('inf')
        
        for iteration in range(max_iters):
            amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
            pred_waveform = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples_audio)
            
            loss, _ = loss_fn(pred_waveform, waveform, model_amplitude=amplitude, model_sigma=sigma)
            
            optimizer.zero_grad()
            loss.backward()
            model.accumulate_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Soft LR decay
            
            # Density control (FLEXIBLE boundaries)
            if densify_from_iter <= iteration < densify_until_iter and iteration % densification_interval == 0:
                density_controller.densify_and_prune(model, optimizer)
            
            final_loss = loss.item()
            
            # Log progress
            if iteration % 1000 == 0:
                wandb.log({
                    'iteration': iteration,
                    'loss': final_loss,
                    'num_atoms': model.num_atoms,
                    'audio_idx': audio_idx,
                })
        
        elapsed = time.time() - start_time
        
        # Final reconstruction
        with torch.no_grad():
            amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
            recon = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples_audio)
        
        # Compute quality metrics
        metrics = {'final_loss': final_loss, 'num_atoms': model.num_atoms, 'time': elapsed}
        
        # MSS loss
        mss = 0
        for fft_size in [512, 1024, 2048]:
            window = torch.hann_window(fft_size, device=device)
            spec_orig = torch.stft(waveform, fft_size, fft_size//4, window=window, return_complex=True)
            spec_recon = torch.stft(recon, fft_size, fft_size//4, window=window, return_complex=True)
            mss += (spec_orig.abs() - spec_recon.abs()).abs().mean().item()
        metrics['mss'] = mss / 3
        
        # PESQ
        if PESQ_AVAILABLE:
            try:
                orig_16k = torchaudio.functional.resample(waveform.cpu(), sample_rate, 16000).numpy()
                recon_16k = torchaudio.functional.resample(recon.cpu(), sample_rate, 16000).numpy()
                metrics['pesq'] = pesq(16000, orig_16k, recon_16k, 'wb')
            except Exception as e:
                print(f"[Warning] PESQ failed: {e}")
                metrics['pesq'] = 2.5  # Default mediocre score
        else:
            # Use MSS as proxy (lower is better, convert to PESQ-like scale)
            metrics['pesq'] = max(1.0, 4.5 - metrics['mss'] * 10)
        
        # STOI
        if STOI_AVAILABLE:
            try:
                metrics['stoi'] = stoi(waveform.cpu().numpy(), recon.cpu().numpy(), sample_rate)
            except Exception:
                metrics['stoi'] = 0.5
        
        all_metrics['final_loss'].append(metrics['final_loss'])
        all_metrics['num_atoms'].append(metrics['num_atoms'])
        all_metrics['time_per_audio'].append(elapsed)
        all_metrics['mss_loss'].append(metrics['mss'])
        all_metrics['pesq'].append(metrics['pesq'])
        if 'stoi' in metrics:
            all_metrics['stoi'].append(metrics['stoi'])
        
        print(f"  Loss: {final_loss:.4f}, Atoms: {model.num_atoms}, PESQ: {metrics['pesq']:.2f}")
    
    # Aggregate results
    final_pesq = np.mean(all_metrics['pesq'])
    final_num_atoms = np.mean(all_metrics['num_atoms'])
    
    results = {
        'final_loss': np.mean(all_metrics['final_loss']),
        'final_num_atoms': final_num_atoms,
        'final_pesq': final_pesq,
        'mss_loss': np.mean(all_metrics['mss_loss']),
        'time_per_audio': np.mean(all_metrics['time_per_audio']),
        
        # COMPOSITE SCORE: The main optimization target
        'composite_score': compute_composite_score(final_pesq, final_num_atoms),
    }
    
    if all_metrics['stoi']:
        results['final_stoi'] = np.mean(all_metrics['stoi'])
    
    return results

def main():
    # Initialize wandb
    wandb.init()
    config = dict(wandb.config)
    
    print(f"\n{'='*60}")
    print(f"[Sweep] Running trial with config:")
    print(f"{'='*60}")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")
    
    # Run the trial
    try:
        results = run_trial(config)
        
        # Log final results
        wandb.log(results)
        
        print(f"\n{'='*60}")
        print(f"[Sweep] Trial completed:")
        print(f"{'='*60}")
        print(f"  PESQ:            {results['final_pesq']:.3f}")
        print(f"  Num Atoms:       {results['final_num_atoms']:.0f}")
        print(f"  Composite Score: {results['composite_score']:.3f}")
        print(f"{'='*60}\n")
            
    except Exception as e:
        print(f"[Sweep] Trial failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.log({'error': str(e), 'composite_score': 0, 'final_pesq': 0})
        raise

if __name__ == "__main__":
    main()