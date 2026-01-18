#!/usr/bin/env python
"""
Pure Autoencoder Training Script for GaborGridEncoder

Simplified training pipeline:
- Multi-Scale STFT reconstruction loss only
- No GAN, no existence detection, no complex sparsity
- Deterministic encoder for downstream Flow Matching

Usage:
    torchrun --nproc_per_node=4 scripts/01_encoder_training/run_encoder_train.py \
        --config configs/AudioGS_config.yaml --use_wandb
"""

import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb
import numpy as np

from models.encoder import build_encoder
from losses.spectral_loss import CombinedAudioLoss
from utils.visualization import Visualizer

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from cuda_gabor import get_cuda_gabor_renderer
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False


def setup_ddp():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0


class AudioDataset(Dataset):
    def __init__(self, data_paths, sample_rate=24000, max_length_sec=5.0, min_length_sec=1.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_sec * sample_rate)
        self.min_samples = int(min_length_sec * sample_rate)
        self.files = []
        
        for p in data_paths:
            if os.path.isdir(p):
                found = list(Path(p).rglob("*.wav"))
                self.files.extend(found)
                if is_rank_zero():
                    print(f"[Dataset] Found {len(found)} files in {p}")
        if is_rank_zero():
            print(f"[Dataset] Total: {len(self.files)}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.files[idx])
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        
        audio = audio.mean(dim=0)  # Convert to mono
        
        # Normalize
        audio = audio / (audio.abs().max() + 1e-8)
        
        # Pad or crop
        if audio.shape[0] > self.max_samples:
            start = torch.randint(0, audio.shape[0] - self.max_samples, (1,)).item()
            audio = audio[start:start + self.max_samples]
        elif audio.shape[0] < self.min_samples:
            audio = F.pad(audio, (0, self.min_samples - audio.shape[0]))
        
        return audio, audio.shape[0]


def collate_fn(batch):
    audios, lengths = zip(*batch)
    max_len = max(lengths)
    padded = torch.stack([F.pad(a, (0, max_len - a.shape[0])) for a in audios])
    return padded, torch.tensor(lengths)


def compute_si_sdr(pred, target):
    """Scale-Invariant SDR."""
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    s_target = (torch.sum(pred * target, dim=-1, keepdim=True) / 
                (torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8)) * target
    e_noise = pred - s_target
    return 10 * torch.log10(s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + 1e-8) + 1e-8).mean()


class AutoencoderTrainer:
    """Pure Autoencoder Trainer - No GAN, No Existence Detection."""
    
    def __init__(self, config, device, rank, output_dir):
        self.config = config
        self.device = device
        self.rank = rank
        self.is_master = (rank == 0)
        self.output_dir = output_dir
        
        train_cfg = config.get('training', {})
        loss_cfg = config.get('loss', {})
        self.sample_rate = config['data']['sample_rate']
        
        # ========== Encoder ==========
        self.encoder = build_encoder(config).to(device)
        if self.is_master:
            num_params = sum(p.numel() for p in self.encoder.parameters())
            print(f"[Encoder] {num_params:,} params")
        
        if dist.is_initialized():
            self.encoder = DDP(self.encoder, device_ids=[device.index])
        
        # ========== Renderer ==========
        if not RENDERER_AVAILABLE:
            raise RuntimeError("CUDA renderer required!")
        self.renderer = get_cuda_gabor_renderer(sample_rate=self.sample_rate)
        
        # ========== Loss ==========
        self.recon_loss_fn = CombinedAudioLoss(
            sample_rate=self.sample_rate,
            fft_sizes=loss_cfg.get('fft_sizes', [2048, 1024, 512]),
            hop_sizes=loss_cfg.get('hop_sizes', [512, 256, 128]),
            win_lengths=loss_cfg.get('win_lengths', [2048, 1024, 512]),
            stft_weight=loss_cfg.get('spectral_weight', 1.0),
            mel_weight=loss_cfg.get('mel_weight', 45.0),
            time_weight=loss_cfg.get('time_domain_weight', 0.1),
            phase_weight=loss_cfg.get('phase_weight', 2.0),
            amp_reg_weight=loss_cfg.get('amp_reg_weight', 0.0),  # Disabled
            pre_emp_weight=loss_cfg.get('pre_emp_weight', 2.0),
        ).to(device)
        
        # Optional L1 regularization on amplitude (very light)
        self.amp_reg_weight = loss_cfg.get('amp_l1_weight', 1e-4)
        
        # ========== Optimizer ==========
        lr = float(train_cfg.get('learning_rate', 2e-4))
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr, betas=(0.9, 0.99))
        
        # ========== Scheduler ==========
        warmup_steps = train_cfg.get('warmup_steps', 1000)
        max_steps = train_cfg.get('max_steps', 100000)
        
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max_steps - warmup_steps, eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], [warmup_steps])
        
        # ========== Gradient Accumulation ==========
        self.accum_steps = train_cfg.get('accumulation_steps', 1)
        self.accum_count = 0
        self.global_step = 0
        
        # ========== Visualization ==========
        if self.is_master:
            self.visualizer = Visualizer(str(output_dir / "visualizations"), sample_rate=self.sample_rate)
        else:
            self.visualizer = None
        
        self.ref_audio = None
    
    def set_reference_sample(self, audio):
        self.ref_audio = audio.to(self.device)
    
    def render_batch(self, enc_output, num_samples, prune_low_amp=False, amp_threshold=1e-4):
        """Render Gabor atoms to waveform.
        
        Args:
            enc_output: Encoder output dictionary
            num_samples: Number of audio samples to render
            prune_low_amp: If True, skip atoms with amplitude < amp_threshold (for inference speedup)
            amp_threshold: Minimum amplitude to keep (default 1e-4)
        """
        batch_size = enc_output['amplitude'].shape[0]
        reconstructed = []
        
        for b in range(batch_size):
            amp = enc_output['amplitude'][b].contiguous()
            tau = enc_output['tau'][b].contiguous()
            omega = enc_output['omega'][b].contiguous()
            sigma = enc_output['sigma'][b].contiguous()
            phi = enc_output['phi'][b].contiguous()
            gamma = enc_output['gamma'][b].contiguous()
            
            # Inference pruning: skip low-amplitude atoms
            if prune_low_amp:
                mask = amp > amp_threshold
                amp = amp[mask]
                tau = tau[mask]
                omega = omega[mask]
                sigma = sigma[mask]
                phi = phi[mask]
                gamma = gamma[mask]
            
            recon = self.renderer(amp, tau, omega, sigma, phi, gamma, num_samples)
            reconstructed.append(recon)
        
        return torch.stack(reconstructed, dim=0)
    
    def train_step(self, batch):
        """Simple reconstruction training step."""
        audio, lengths = batch
        audio = audio.to(self.device, non_blocking=True)
        num_samples = audio.shape[-1]
        
        is_accumulating = (self.accum_count + 1) % self.accum_steps != 0
        
        # Forward pass (FP32)
        enc_output = self.encoder(audio.float())
        
        # Render
        fake_audio = self.render_batch(enc_output, num_samples)
        
        # NaN protection
        if not torch.isfinite(fake_audio).all():
            if self.is_master:
                print(f"[Warning] Step {self.global_step}: NaN in output, skipping")
            self.accum_count += 1
            return self._nan_metrics()
        
        # Reconstruction loss with phase vector regularization
        recon_loss, loss_dict = self.recon_loss_fn(
            fake_audio.float(), audio.float(),
            model_amplitude=enc_output['amplitude'].float(),
            model_sigma=enc_output['sigma'].float(),
            model_phase_raw=(enc_output['cos_phi_raw'].float(), enc_output['sin_phi_raw'].float())
        )
        
        # Optional: Light L1 on amplitude (encourages sparsity naturally)
        amp_reg = self.amp_reg_weight * enc_output['amplitude'].abs().mean()
        
        # Total loss
        total_loss = (recon_loss + amp_reg) / self.accum_steps
        
        # Backward
        total_loss.backward()
        
        if not is_accumulating:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        self.global_step += 1
        self.accum_count += 1
        
        # Metrics
        with torch.no_grad():
            si_sdr = compute_si_sdr(fake_audio, audio).item()
            avg_amp = enc_output['amplitude'].mean().item()
        
        return {
            'loss': total_loss.item() * self.accum_steps,
            'recon_loss': recon_loss.item(),
            'mel_loss': loss_dict.get('mel', 0.0),
            'si_sdr': si_sdr,
            'amp_reg': amp_reg.item(),
            'avg_amp': avg_amp,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    def _nan_metrics(self):
        return {
            'loss': 0.0, 'recon_loss': 0.0, 'mel_loss': 0.0,
            'si_sdr': 0.0, 'amp_reg': 0.0, 'avg_amp': 0.0,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    def validate(self, val_loader, max_batches=20):
        self.encoder.eval()
        
        total_loss = 0.0
        total_si_sdr = 0.0
        total_pesq = 0.0
        count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_batches:
                    break
                
                audio, lengths = batch
                audio = audio.to(self.device)
                num_samples = audio.shape[-1]
                
                enc_output = self.encoder(audio.float())
                fake_audio = self.render_batch(enc_output, num_samples)
                
                recon_loss, _ = self.recon_loss_fn(
                    fake_audio.float(), audio.float(),
                    model_amplitude=enc_output['amplitude'].float(),
                    model_sigma=enc_output['sigma'].float()
                )
                
                total_loss += recon_loss.item()
                total_si_sdr += compute_si_sdr(fake_audio, audio).item()
                
                if PESQ_AVAILABLE and i < 5:
                    try:
                        for b in range(min(audio.shape[0], 2)):
                            ref = audio[b].cpu()
                            deg = fake_audio[b].detach().cpu()
                            # PESQ only supports 8000Hz (NB) or 16000Hz (WB)
                            # Resample from 24000Hz to 16000Hz
                            if self.sample_rate != 16000:
                                ref = torchaudio.functional.resample(ref, self.sample_rate, 16000).numpy()
                                deg = torchaudio.functional.resample(deg, self.sample_rate, 16000).numpy()
                            else:
                                ref = ref.numpy()
                                deg = deg.numpy()
                            total_pesq += pesq(16000, ref, deg, 'wb')
                    except Exception as e:
                        pass  # Silently skip PESQ errors
                
                count += 1
        
        self.encoder.train()
        
        results = {
            'val_loss': total_loss / max(count, 1),
            'val_si_sdr': total_si_sdr / max(count, 1),
        }
        if total_pesq > 0:
            results['val_pesq'] = total_pesq / (count * 2)
        
        # Visualize reference sample
        if self.is_master and self.ref_audio is not None:
            enc_out = self.encoder(self.ref_audio.unsqueeze(0).float())
            recon = self.render_batch(enc_out, self.ref_audio.shape[-1])
            recon = recon.squeeze(0)
            
            step_str = f"step_{self.global_step:06d}"
            self.visualizer.save_audio(recon, f"{step_str}_recon")
            self.visualizer.plot_spectrogram_comparison(self.ref_audio, recon, f"{step_str}_spec",
                                                        title=f"Step {self.global_step}")
        
        return results
    
    def save_checkpoint(self, path, extra=None):
        if not self.is_master:
            return
        
        enc_state = self.encoder.module.state_dict() if isinstance(self.encoder, DDP) else self.encoder.state_dict()
        
        ckpt = {
            'global_step': self.global_step,
            'encoder_state': enc_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f"[Checkpoint] Saved to {path}")
    
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        
        if isinstance(self.encoder, DDP):
            self.encoder.module.load_state_dict(ckpt['encoder_state'])
        else:
            self.encoder.load_state_dict(ckpt['encoder_state'])
        
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.global_step = ckpt.get('global_step', 0)
        
        if self.is_master:
            print(f"[Checkpoint] Loaded from step {self.global_step}")


def train(args, config):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    output_dir = Path(f"logs/encoder_{args.exp_name}")
    if is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if dist.is_initialized():
        dist.barrier()
    
    if is_rank_zero() and args.use_wandb:
        wandb.init(project="AudioGS-AE", config=config, name=args.exp_name)
    
    trainer = AutoencoderTrainer(config, device, rank, output_dir)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Datasets
    data_path = config['data']['dataset_path']
    train_subsets = config['data'].get('subsets', ['train-clean-100'])
    train_paths = [os.path.join(data_path, "train", s) for s in train_subsets]
    train_dataset = AudioDataset(train_paths, config['data']['sample_rate'],
                                  config['data'].get('max_audio_length', 3.0))
    
    val_subsets = config['data'].get('val_subsets', ['dev-clean'])
    val_paths = [os.path.join(data_path, "dev", s) for s in val_subsets]
    val_dataset = AudioDataset(val_paths, config['data']['sample_rate'],
                                config['data'].get('max_audio_length', 3.0))
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    
    batch_size = config['training'].get('batch_size', 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=4, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size // 2),
                            sampler=val_sampler, num_workers=2, collate_fn=collate_fn)
    
    if is_rank_zero() and len(val_dataset) > 0:
        ref_audio, _ = val_dataset[0]
        trainer.set_reference_sample(ref_audio)
        trainer.visualizer.save_audio(ref_audio, "reference_gt")
    
    max_steps = config['training'].get('max_steps', 100000)
    val_interval = config['training'].get('val_interval', 2000)
    save_interval = config['training'].get('save_interval', 10000)
    log_interval = config['training'].get('log_interval', 10)
    
    if is_rank_zero():
        print(f"\n{'='*60}")
        print("AudioGS Pure Autoencoder Training")
        print(f"{'='*60}")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, GPUs: {world_size}")
        print(f"Batch: {batch_size}, Accum: {trainer.accum_steps}, Max steps: {max_steps}")
        print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    epoch = 0
    train_iter = iter(train_loader)
    
    if is_rank_zero():
        pbar = tqdm(total=max_steps, initial=trainer.global_step, desc="Training")
    
    while trainer.global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if dist.is_initialized():
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        metrics = trainer.train_step(batch)
        
        if is_rank_zero():
            pbar.update(1)
            pbar.set_postfix({
                'L': f"{metrics['loss']:.1f}",
                'Mel': f"{metrics['mel_loss']:.1f}",
                'SDR': f"{metrics['si_sdr']:.1f}",
                'amp': f"{metrics['avg_amp']:.3f}",
            })
            
            if args.use_wandb and trainer.global_step % log_interval == 0:
                wandb.log({
                    'train/loss': metrics['loss'],
                    'train/recon_loss': metrics['recon_loss'],
                    'train/mel_loss': metrics['mel_loss'],
                    'train/si_sdr': metrics['si_sdr'],
                    'train/avg_amp': metrics['avg_amp'],
                    'train/lr': metrics['lr'],
                    'step': trainer.global_step,
                })
        
        if trainer.global_step % val_interval == 0 and trainer.global_step > 0:
            val_metrics = trainer.validate(val_loader)
            if is_rank_zero():
                pesq_str = f", PESQ={val_metrics['val_pesq']:.2f}" if val_metrics.get('val_pesq', 0) > 0 else ""
                tqdm.write(f"\n[Val] Step {trainer.global_step}: "
                          f"Loss={val_metrics['val_loss']:.2f}, "
                          f"SI-SDR={val_metrics['val_si_sdr']:.1f}dB{pesq_str}")
                
                if args.use_wandb:
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()}, step=trainer.global_step)
                
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    trainer.save_checkpoint(str(output_dir / "best_model.pt"),
                                            extra={'val_metrics': val_metrics})
        
        if trainer.global_step % save_interval == 0 and trainer.global_step > 0:
            if is_rank_zero():
                trainer.save_checkpoint(str(output_dir / f"checkpoint_{trainer.global_step}.pt"))
    
    if is_rank_zero():
        trainer.save_checkpoint(str(output_dir / "final_model.pt"))
        pbar.close()
        print("\nTraining complete!")
        if args.use_wandb:
            wandb.finish()
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Train AudioGS Autoencoder")
    parser.add_argument("--config", type=str, default="configs/AudioGS_config.yaml")
    parser.add_argument("--exp_name", type=str, default="pure_ae_v1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(args, config)


if __name__ == "__main__":
    main()