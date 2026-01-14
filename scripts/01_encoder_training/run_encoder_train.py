#!/usr/bin/env python
"""
Encoder Training Script for GaborGridEncoder (DDP Enabled)

Trains the encoder to predict Gabor atom parameters directly from audio.
Uses the differentiable GaborRenderer as the decoder for reconstruction loss.

Usage:
    torchrun --nproc_per_node=4 scripts/02_encoder_training/train_encoder.py --config configs/AudioGS_config.yaml
"""

import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import wandb
import numpy as np

from models.encoder import build_encoder
from losses.spectral_loss import CombinedAudioLoss

# Optional quality metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("[Warning] pesq not available")

# CUDA renderer
try:
    from cuda_gabor import get_cuda_gabor_renderer
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    print("[Error] CUDA renderer not available!")


def setup_ddp():
    """Initialize Distributed Data Parallel."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Initialized Rank {rank}/{world_size} (Local {local_rank})")
        return rank, local_rank, world_size
    else:
        print("[DDP] Not using distributed mode (Single GPU)")
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0


class AudioDataset(Dataset):
    """Dataset for loading audio files from one or more directories."""
    
    def __init__(
        self, 
        data_paths,  # Can be str or list of str
        sample_rate: int = 24000,
        max_length_sec: float = 5.0,
        min_length_sec: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_sec * sample_rate)
        self.min_samples = int(min_length_sec * sample_rate)
        
        # Handle single path or list of paths
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        # Find all wav files from all paths
        self.files = []
        for data_path in data_paths:
            path = Path(data_path)
            if path.exists():
                found = list(path.rglob("*.wav"))
                self.files.extend(found)
                if is_rank_zero():
                    print(f"[Dataset] Found {len(found)} audio files in {data_path}")
            else:
                if is_rank_zero():
                    print(f"[Dataset] WARNING: Path does not exist: {data_path}")
        
        if is_rank_zero():
            print(f"[Dataset] Total: {len(self.files)} audio files")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        # Load audio
        # Note: In a real DDP scenario with huge datasets, ensure file opening is safe
        try:
            waveform, sr = torchaudio.load(str(audio_path))
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.min_samples), self.min_samples
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Truncate or pad
        if len(waveform) > self.max_samples:
            # Random crop
            start = torch.randint(0, len(waveform) - self.max_samples, (1,)).item()
            waveform = waveform[start:start + self.max_samples]
        elif len(waveform) < self.min_samples:
            # Pad with zeros
            waveform = F.pad(waveform, (0, self.min_samples - len(waveform)))
        
        return waveform, len(waveform)


def collate_fn(batch):
    """Collate function for variable length audio."""
    waveforms, lengths = zip(*batch)
    
    # Pad to max length in batch
    max_len = max(lengths)
    padded = torch.stack([
        F.pad(w, (0, max_len - len(w))) for w in waveforms
    ])
    
    return padded, torch.tensor(lengths)


def compute_si_sdr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Scale-Invariant Signal-to-Distortion Ratio."""
    # Ensure shapes match
    if pred.shape != target.shape:
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

    # Remove mean
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # Compute SI-SDR
    dot = (pred * target).sum(dim=-1, keepdim=True)
    s_target = dot * target / (target.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
    e_noise = pred - s_target
    
    si_sdr = 10 * torch.log10(
        s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + 1e-8) + 1e-8
    )
    
    return si_sdr.mean()


class EncoderTrainer:
    """Training manager for GaborGridEncoder."""
    
    def __init__(
        self,
        config: dict,
        device: torch.device,
        rank: int,
    ):
        self.config = config
        self.device = device
        self.rank = rank
        self.is_master = (rank == 0)
        
        # Extract configs
        self.data_config = config.get('data', {})
        self.enc_config = config.get('encoder_model', {})
        self.train_config = config.get('training', {})
        self.loss_config = config.get('loss', {})
        
        self.sample_rate = self.data_config.get('sample_rate', 24000)
        
        # Build encoder
        self.encoder = build_encoder(config).to(device)
        
        if self.is_master:
            print(f"[Encoder] Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
            print(f"[Encoder] Atoms/second capacity: {self.encoder.atoms_per_second:,}")
        
        # Wrap DDP
        if dist.is_initialized():
            self.encoder = DDP(
                self.encoder, 
                device_ids=[device.index],
                find_unused_parameters=config.get('distributed', {}).get('find_unused_parameters', False)
            )
        
        # Build renderer
        if not RENDERER_AVAILABLE:
            raise RuntimeError("CUDA renderer required!")
        self.renderer = get_cuda_gabor_renderer(sample_rate=self.sample_rate)
        
        # Build loss
        self.loss_fn = CombinedAudioLoss(
            sample_rate=self.sample_rate,
            fft_sizes=self.loss_config.get('fft_sizes', [2048, 1024, 512]),
            hop_sizes=self.loss_config.get('hop_sizes', [512, 256, 128]),
            win_lengths=self.loss_config.get('win_lengths', [2048, 1024, 512]),
            stft_weight=self.loss_config.get('spectral_weight', 1.0),
            mel_weight=self.loss_config.get('mel_weight', 0.5),
            time_weight=self.loss_config.get('time_domain_weight', 0.1),
            phase_weight=self.loss_config.get('phase_weight', 0.8),
            amp_reg_weight=self.loss_config.get('amp_reg_weight', 0.01),
            pre_emp_weight=self.loss_config.get('pre_emp_weight', 20.0),
        ).to(device)
        
        self.sparsity_weight = self.loss_config.get('sparsity_weight', 0.0)
        
        # Optimizer
        learning_rate = float(self.train_config.get('learning_rate', 5e-4))
        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        
        # Schedulers
        warmup_steps = self.train_config.get('warmup_steps', 1000)
        max_epochs = self.train_config.get('max_epochs', 100)
        
        self.warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.01, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        self.main_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=max_epochs - (warmup_steps // 1000 if warmup_steps >= 1000 else 0) 
        )
        
        self.global_step = 0
        self.epoch = 0
        
    def render_batch(
        self, 
        enc_output: dict, 
        num_samples: int,
    ) -> torch.Tensor:
        """Render audio from encoder output (batch)."""
        batch_size = enc_output['amplitude'].shape[0]
        reconstructed = []
        
        for b in range(batch_size):
            # Renderer expects contiguous tensors
            recon = self.renderer(
                enc_output['amplitude'][b].contiguous(),
                enc_output['tau'][b].contiguous(),
                enc_output['omega'][b].contiguous(),
                enc_output['sigma'][b].contiguous(),
                enc_output['phi'][b].contiguous(),
                enc_output['gamma'][b].contiguous(),
                num_samples,
            )
            reconstructed.append(recon)
        
        return torch.stack(reconstructed, dim=0)
    
    def train_step(self, batch: torch.Tensor) -> dict:
        """Single training step."""
        audio, lengths = batch
        audio = audio.to(self.device, non_blocking=True)
        num_samples = audio.shape[-1]
        
        # Forward through encoder
        enc_output = self.encoder(audio)
        
        # Render reconstruction
        reconstructed = self.render_batch(enc_output, num_samples)
        
        # 1. Reconstruction Loss
        loss, loss_dict = self.loss_fn(reconstructed, audio, 
                                       model_amplitude=enc_output['amplitude'],
                                       model_sigma=enc_output['sigma'])
        
        # 2. Sparsity Loss [CRITICAL]
        # Penalize the mean probability of atoms existing
        existence_prob_mean = enc_output['existence_prob'].mean()
        sparsity_loss = self.sparsity_weight * existence_prob_mean
        
        loss = loss + sparsity_loss
        
        # Update dict
        loss_dict['sparsity'] = sparsity_loss.item()
        loss_dict['total'] = loss.item() # Update total
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Step scheduler
        if self.global_step < self.train_config.get('warmup_steps', 1000):
            self.warmup_scheduler.step()
        
        self.global_step += 1
        
        # Compute active atom ratio (for logging)
        with torch.no_grad():
            active_ratio = (enc_output['existence_prob'] > 0.5).float().mean().item()
        
        return {
            'loss': loss.item(),
            'active_ratio': active_ratio,
            **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()},
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, max_batches: int = 20) -> dict:
        """Run validation."""
        self.encoder.eval()
        
        all_losses = []
        all_si_sdr = []
        all_pesq = []
        all_active_ratio = []
        
        # To avoid DDP hanging, minimal validation on rank 0 is often sufficient 
        # for these metrics, or gather. Here we just run on each rank and average.
        
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
                
            audio, lengths = batch
            audio = audio.to(self.device)
            num_samples = audio.shape[-1]
            
            # Forward
            enc_output = self.encoder(audio)
            reconstructed = self.render_batch(enc_output, num_samples)
            
            # Loss
            loss, _ = self.loss_fn(reconstructed, audio)
            all_losses.append(loss.item())
            
            # SI-SDR
            si_sdr = compute_si_sdr(reconstructed, audio)
            all_si_sdr.append(si_sdr.item())
            
            # Active ratio
            active = (enc_output['existence_prob'] > 0.5).float().mean().item()
            all_active_ratio.append(active)
            
            # PESQ (Expensive, only compute on Rank 0 and a few samples)
            if self.is_master and PESQ_AVAILABLE and i < 5:
                try:
                    ref_16k = torchaudio.functional.resample(
                        audio[0].cpu(), self.sample_rate, 16000
                    ).numpy()
                    deg_16k = torchaudio.functional.resample(
                        reconstructed[0].cpu(), self.sample_rate, 16000
                    ).numpy()
                    pesq_score = pesq(16000, ref_16k, deg_16k, 'wb')
                    all_pesq.append(pesq_score)
                except Exception as e:
                    # print(f"PESQ failed: {e}")
                    pass
        
        self.encoder.train()
        
        # Aggregate metrics from all ranks would be ideal, but minimal simple avg is okay for now
        results = {
            'val_loss': np.mean(all_losses) if all_losses else 0.0,
            'val_si_sdr': np.mean(all_si_sdr) if all_si_sdr else 0.0,
            'val_active_ratio': np.mean(all_active_ratio) if all_active_ratio else 0.0,
        }
        
        if all_pesq:
            results['val_pesq'] = np.mean(all_pesq)
            
        return results
    
    def save_checkpoint(self, path: str, extra: dict = None):
        """Save training checkpoint (Rank 0 only)."""
        if not self.is_master:
            return
            
        # Unwrap DDP
        model_state = self.encoder.module.state_dict() if isinstance(self.encoder, DDP) else self.encoder.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'encoder_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved to {path}")
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Unwrap DDP for loading
        if isinstance(self.encoder, DDP):
            self.encoder.module.load_state_dict(checkpoint['encoder_state'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        if self.is_master:
            print(f"[Checkpoint] Loaded from epoch {self.epoch}, step {self.global_step}")


def train(args, config):
    """Main training function."""
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize wandb (Rank 0 only)
    if is_rank_zero() and args.use_wandb:
        wandb.init(
            project="AudioGS-Encoder",
            config=config,
            name=args.exp_name,
        )
    
    # Create trainer
    trainer = EncoderTrainer(config, device, rank)
    
    # Create datasets
    data_path = config['data']['dataset_path']  # e.g., data/raw/LibriTTS_R
    train_config = config.get('training', {})
    
    # Training: train/train-clean-100 + train/train-clean-360
    train_paths = [
        os.path.join(data_path, "train", "train-clean-100"),
        os.path.join(data_path, "train", "train-clean-360"),
    ]
    train_dataset = AudioDataset(
        data_paths=train_paths,
        sample_rate=config['data']['sample_rate'],
        max_length_sec=config['data'].get('max_audio_length', 5.0),
    )
    
    # Validation: dev/dev-clean
    val_dataset = AudioDataset(
        data_paths=os.path.join(data_path, "dev", "dev-clean"),
        sample_rate=config['data']['sample_rate'],
        max_length_sec=config['data'].get('max_audio_length', 5.0),
    )
    
    # DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 16),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get('batch_size', 16) // 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    # Training loop
    max_epochs = train_config.get('max_epochs', 100)
    val_interval = train_config.get('val_interval', 1000)
    save_interval = train_config.get('save_interval', 5000)
    log_interval = train_config.get('log_interval', 10)
    
    output_dir = Path(f"logs/encoder_{args.exp_name}")
    if is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_rank_zero():
        print(f"\n{'='*60}")
        print(f"GaborGridEncoder Training")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"GPUs: {world_size}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            
        trainer.epoch = epoch
        
        if is_rank_zero():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        else:
            pbar = train_loader
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            
            if is_rank_zero():
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'active': f"{metrics['active_ratio']:.1%}",
                    'spars': f"{metrics.get('sparsity', 0):.4f}",
                })
                
                # Log to wandb
                if args.use_wandb and trainer.global_step % log_interval == 0:
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/active_ratio': metrics['active_ratio'],
                        'train/sparsity_loss': metrics.get('sparsity', 0.0),
                        'train/phase_loss': metrics.get('phase', 0.0),
                        'train/spectral_loss': metrics.get('stft', 0.0),
                        'train/mel_loss': metrics.get('mel', 0.0),
                        'train/lr': trainer.optimizer.param_groups[0]['lr'],
                        'step': trainer.global_step,
                    })
            
            # Validation
            if trainer.global_step % val_interval == 0:
                val_metrics = trainer.validate(val_loader)
                
                if is_rank_zero():
                    print(f"\n[Val] Step {trainer.global_step}: "
                          f"Loss={val_metrics['val_loss']:.4f}, "
                          f"SI-SDR={val_metrics['val_si_sdr']:.2f}, "
                          f"Active={val_metrics['val_active_ratio']:.1%}")
                    
                    if 'val_pesq' in val_metrics:
                        print(f"       PESQ={val_metrics['val_pesq']:.3f}")
                    
                    if args.use_wandb:
                        wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
                    
                    # Save best
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        trainer.save_checkpoint(
                            str(output_dir / "best_model.pt"),
                            extra={'val_metrics': val_metrics}
                        )
            
            # Save checkpoint
            if trainer.global_step % save_interval == 0:
                trainer.save_checkpoint(
                    str(output_dir / f"checkpoint_{trainer.global_step}.pt")
                )
        
        # End of epoch
        trainer.main_scheduler.step()
    
    # Final save
    trainer.save_checkpoint(str(output_dir / "final_model.pt"))
    
    if is_rank_zero() and args.use_wandb:
        wandb.finish()
    
    cleanup_ddp()
    if is_rank_zero():
        print("Training Complete!")


def main():
    parser = argparse.ArgumentParser(description="Train GaborGridEncoder")
    parser.add_argument("--config", type=str, default="configs/AudioGS_config.yaml")
    parser.add_argument("--exp_name", type=str, default="v1")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(args, config)


if __name__ == "__main__":
    main()