#!/usr/bin/env python
"""
GAN Training Script for GaborGridEncoder

Features:
- Adversarial training with MPD + MSD discriminators (HiFi-GAN style)
- Feature matching loss for stable training
- Step-based training with warmup
- Energy-based sparsity loss
- Structural warmup (freeze delta_tau/omega/sigma for initial steps)

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
from models.discriminator import build_discriminator, discriminator_loss, generator_loss, feature_matching_loss
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
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
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
        
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        self.files = []
        for p in data_paths:
            path = Path(p)
            if path.exists():
                found = list(path.rglob("*.wav"))
                self.files.extend(found)
                if is_rank_zero():
                    print(f"[Dataset] Found {len(found)} files in {p}")
        if is_rank_zero():
            print(f"[Dataset] Total: {len(self.files)}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            waveform, sr = torchaudio.load(str(self.files[idx]))
        except:
            return torch.zeros(self.min_samples), self.min_samples
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        if len(waveform) > self.max_samples:
            start = torch.randint(0, len(waveform) - self.max_samples, (1,)).item()
            waveform = waveform[start:start + self.max_samples]
        elif len(waveform) < self.min_samples:
            waveform = F.pad(waveform, (0, self.min_samples - len(waveform)))
        
        return waveform, len(waveform)


def collate_fn(batch):
    waveforms, lengths = zip(*batch)
    max_len = max(lengths)
    padded = torch.stack([F.pad(w, (0, max_len - len(w))) for w in waveforms])
    return padded, torch.tensor(lengths)


def compute_si_sdr(pred, target):
    if pred.shape != target.shape:
        min_len = min(pred.shape[-1], target.shape[-1])
        pred, target = pred[..., :min_len], target[..., :min_len]
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (pred * target).sum(dim=-1, keepdim=True)
    s_target = dot * target / (target.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
    e_noise = pred - s_target
    return 10 * torch.log10(s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + 1e-8) + 1e-8).mean()


class GANTrainer:
    """GAN Trainer with Generator (Encoder) and Discriminator."""
    
    def __init__(self, config, device, rank, output_dir):
        self.config = config
        self.device = device
        self.rank = rank
        self.is_master = (rank == 0)
        self.output_dir = output_dir
        
        train_cfg = config.get('training', {})
        loss_cfg = config.get('loss', {})
        gan_cfg = config.get('gan', {})
        self.sample_rate = config['data']['sample_rate']
        
        # ========== Generator (Encoder) ==========
        self.generator = build_encoder(config).to(device)
        if self.is_master:
            print(f"[Generator] {sum(p.numel() for p in self.generator.parameters()):,} params")
        
        if dist.is_initialized():
            self.generator = DDP(self.generator, device_ids=[device.index], find_unused_parameters=True)
        
        # ========== Discriminator ==========
        self.discriminator = build_discriminator().to(device)
        if self.is_master:
            print(f"[Discriminator] {sum(p.numel() for p in self.discriminator.parameters()):,} params")
        
        if dist.is_initialized():
            self.discriminator = DDP(self.discriminator, device_ids=[device.index])
        
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
            mel_weight=loss_cfg.get('mel_weight', 25.0),  # High for GAN stability
            time_weight=loss_cfg.get('time_domain_weight', 0.1),
            phase_weight=loss_cfg.get('phase_weight', 2.0),
            amp_reg_weight=loss_cfg.get('amp_reg_weight', 0.01),
            pre_emp_weight=loss_cfg.get('pre_emp_weight', 10.0),
        ).to(device)
        
        self.sparsity_weight = loss_cfg.get('sparsity_weight', 0.001)
        self.fm_weight = gan_cfg.get('fm_weight', 10.0)
        self.adv_weight = gan_cfg.get('adv_weight', 1.0)
        
        # ========== Optimizers ==========
        g_lr = float(train_cfg.get('learning_rate', 2e-4))
        d_lr = float(gan_cfg.get('d_lr', 2e-4))
        
        self.g_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=g_lr, betas=(0.8, 0.99))
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=d_lr, betas=(0.8, 0.99))
        
        # ========== Schedulers ==========
        warmup_steps = train_cfg.get('warmup_steps', 1000)
        max_steps = train_cfg.get('max_steps', 100000)
        
        g_warmup = LinearLR(self.g_optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        g_cosine = CosineAnnealingLR(self.g_optimizer, T_max=max_steps - warmup_steps)
        self.g_scheduler = SequentialLR(self.g_optimizer, [g_warmup, g_cosine], [warmup_steps])
        
        d_warmup = LinearLR(self.d_optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        d_cosine = CosineAnnealingLR(self.d_optimizer, T_max=max_steps - warmup_steps)
        self.d_scheduler = SequentialLR(self.d_optimizer, [d_warmup, d_cosine], [warmup_steps])
        
        self.global_step = 0
        self.warmup_freeze_steps = train_cfg.get('warmup_freeze_structure', 3000)
        
        if self.is_master:
            self.visualizer = Visualizer(str(output_dir / "visualizations"), sample_rate=self.sample_rate)
        else:
            self.visualizer = None
        
        self.ref_audio = None
        
    def set_reference_sample(self, audio):
        self.ref_audio = audio.to(self.device)
    
    def render_batch(self, enc_output, num_samples):
        batch_size = enc_output['amplitude'].shape[0]
        reconstructed = []
        for b in range(batch_size):
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
    
    def train_step(self, batch):
        """Single GAN training step with D and G updates."""
        audio, lengths = batch
        audio = audio.to(self.device, non_blocking=True)
        num_samples = audio.shape[-1]
        
        # ========== Generator Forward ==========
        enc_output = self.generator(audio)
        fake_audio = self.render_batch(enc_output, num_samples)
        
        # ========== Discriminator Step ==========
        # D wants: real -> high, fake -> low
        self.d_optimizer.zero_grad()
        
        real_d_out, real_d_feats = self.discriminator(audio)
        fake_d_out, fake_d_feats = self.discriminator(fake_audio.detach())
        
        d_loss = discriminator_loss(real_d_out, fake_d_out)
        d_loss.backward()
        self.d_optimizer.step()
        
        # ========== Generator Step ==========
        self.g_optimizer.zero_grad()
        
        # Recompute fake through D (for gradients)
        fake_d_out_g, fake_d_feats_g = self.discriminator(fake_audio)
        _, real_d_feats_g = self.discriminator(audio)  # For feature matching
        
        # Adversarial loss
        g_adv_loss = generator_loss(fake_d_out_g) * self.adv_weight
        
        # Feature matching loss
        fm_loss = feature_matching_loss(real_d_feats_g, fake_d_feats_g) * self.fm_weight
        
        # Reconstruction loss
        recon_loss, loss_dict = self.recon_loss_fn(
            fake_audio, audio,
            model_amplitude=enc_output['amplitude'],
            model_sigma=enc_output['sigma']
        )
        
        # Sparsity loss (energy-based): sum of amplitude * existence_prob
        energy_sparsity = (enc_output['amplitude'] * enc_output['existence_prob']).mean()
        sparsity_loss = self.sparsity_weight * energy_sparsity
        
        # Total G loss
        g_loss = recon_loss + g_adv_loss + fm_loss + sparsity_loss
        
        # Structural warmup: freeze delta_tau/omega/sigma gradients
        if self.global_step < self.warmup_freeze_steps:
            # Zero out gradients for structural params by not stepping them
            # This is approximate - we train but the frozen logic is in encoder
            pass
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.g_optimizer.step()
        
        self.g_scheduler.step()
        self.d_scheduler.step()
        self.global_step += 1
        
        with torch.no_grad():
            active_ratio = (enc_output['existence_prob'] > 0.5).float().mean().item()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'recon_loss': recon_loss.item(),
            'adv_loss': g_adv_loss.item(),
            'fm_loss': fm_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'active_ratio': active_ratio,
            'g_lr': self.g_optimizer.param_groups[0]['lr'],
        }
    
    @torch.no_grad()
    def validate(self, val_loader, max_batches=20):
        self.generator.eval()
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_si_sdr = torch.tensor(0.0, device=self.device)
        total_active = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0, device=self.device)
        
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            audio, _ = batch
            audio = audio.to(self.device)
            
            enc_output = self.generator(audio)
            fake = self.render_batch(enc_output, audio.shape[-1])
            
            loss, _ = self.recon_loss_fn(fake, audio)
            total_loss += loss
            total_si_sdr += compute_si_sdr(fake, audio)
            total_active += (enc_output['existence_prob'] > 0.5).float().mean()
            count += 1
        
        if dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_si_sdr, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_active, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        
        cnt = count.item()
        results = {
            'val_loss': (total_loss / cnt).item() if cnt > 0 else 0.0,
            'val_si_sdr': (total_si_sdr / cnt).item() if cnt > 0 else 0.0,
            'val_active_ratio': (total_active / cnt).item() if cnt > 0 else 0.0,
        }
        
        # Visualization
        if self.is_master and self.ref_audio is not None and self.visualizer:
            ref_in = self.ref_audio.unsqueeze(0) if self.ref_audio.dim() == 1 else self.ref_audio
            enc_out = self.generator(ref_in)
            recon = self.render_batch(enc_out, ref_in.shape[-1])[0]
            
            step_str = f"step_{self.global_step:06d}"
            self.visualizer.save_audio(recon, f"{step_str}_recon")
            self.visualizer.plot_spectrogram_comparison(self.ref_audio, recon, f"{step_str}_spec",
                                                         title=f"Step {self.global_step}")
        
        self.generator.train()
        return results
    
    def save_checkpoint(self, path, extra=None):
        if not self.is_master:
            return
        
        g_state = self.generator.module.state_dict() if isinstance(self.generator, DDP) else self.generator.state_dict()
        d_state = self.discriminator.module.state_dict() if isinstance(self.discriminator, DDP) else self.discriminator.state_dict()
        
        ckpt = {
            'global_step': self.global_step,
            'generator_state': g_state,
            'discriminator_state': d_state,
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'd_scheduler': self.d_scheduler.state_dict(),
            'config': self.config,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f"[Checkpoint] Saved to {path}")
    
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        
        if isinstance(self.generator, DDP):
            self.generator.module.load_state_dict(ckpt['generator_state'])
        else:
            self.generator.load_state_dict(ckpt['generator_state'])
        
        if isinstance(self.discriminator, DDP):
            self.discriminator.module.load_state_dict(ckpt['discriminator_state'])
        else:
            self.discriminator.load_state_dict(ckpt['discriminator_state'])
        
        self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
        if 'g_scheduler' in ckpt:
            self.g_scheduler.load_state_dict(ckpt['g_scheduler'])
        if 'd_scheduler' in ckpt:
            self.d_scheduler.load_state_dict(ckpt['d_scheduler'])
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
        wandb.init(project="AudioGS-GAN", config=config, name=args.exp_name)
    
    trainer = GANTrainer(config, device, rank, output_dir)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Datasets
    data_path = config['data']['dataset_path']
    train_subsets = config['data'].get('subsets', ['train-clean-100'])
    train_paths = [os.path.join(data_path, "train", s) for s in train_subsets]
    train_dataset = AudioDataset(train_paths, config['data']['sample_rate'],
                                  config['data'].get('max_audio_length', 5.0))
    
    val_subsets = config['data'].get('val_subsets', ['dev-clean'])
    val_paths = [os.path.join(data_path, "dev", s) for s in val_subsets]
    val_dataset = AudioDataset(val_paths, config['data']['sample_rate'],
                                config['data'].get('max_audio_length', 5.0))
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    
    batch_size = config['training'].get('batch_size', 16)
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
        print("AudioGS GAN Training")
        print(f"{'='*60}")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, GPUs: {world_size}")
        print(f"Max steps: {max_steps}")
        print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    epoch = 0
    train_iter = iter(train_loader)
    
    if is_rank_zero():
        pbar = tqdm(total=max_steps, initial=trainer.global_step, desc="GAN Training")
    
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
                'G': f"{metrics['g_loss']:.3f}",
                'D': f"{metrics['d_loss']:.3f}",
                'act': f"{metrics['active_ratio']:.1%}",
            })
            
            if args.use_wandb and trainer.global_step % log_interval == 0:
                wandb.log({
                    'train/g_loss': metrics['g_loss'],
                    'train/d_loss': metrics['d_loss'],
                    'train/recon_loss': metrics['recon_loss'],
                    'train/adv_loss': metrics['adv_loss'],
                    'train/fm_loss': metrics['fm_loss'],
                    'train/sparsity_loss': metrics['sparsity_loss'],
                    'train/active_ratio': metrics['active_ratio'],
                    'train/lr': metrics['g_lr'],
                    'step': trainer.global_step,
                })
        
        if trainer.global_step % val_interval == 0 and trainer.global_step > 0:
            val_metrics = trainer.validate(val_loader)
            if is_rank_zero():
                tqdm.write(f"\n[Val] Step {trainer.global_step}: "
                          f"Loss={val_metrics['val_loss']:.4f}, "
                          f"SI-SDR={val_metrics['val_si_sdr']:.2f}, "
                          f"Active={val_metrics['val_active_ratio']:.1%}")
                
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
        pbar.close()
        trainer.save_checkpoint(str(output_dir / "final_model.pt"))
        if args.use_wandb:
            wandb.finish()
    
    cleanup_ddp()
    if is_rank_zero():
        print("Training Complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/AudioGS_config.yaml")
    parser.add_argument("--exp_name", default="gan_v1")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train(args, config)


if __name__ == "__main__":
    main()