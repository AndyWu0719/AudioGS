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
from torch.amp import GradScaler
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
    """GAN Trainer with Generator (Encoder) and Discriminator.
    
    Supports staged training:
    - Phase 1: Pure reconstruction, no D (larger batch size)
    - Phase 2: Full GAN training (smaller batch size due to D memory)
    """
    
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
        
        # ========== Batch Size Configuration ==========
        self.batch_size_phase1 = train_cfg.get('batch_size_phase1', train_cfg.get('batch_size', 4))
        self.batch_size_phase2 = train_cfg.get('batch_size_phase2', train_cfg.get('batch_size', 2))
        self.current_batch_size = self.batch_size_phase1  # Start with Phase 1
        
        # ========== Generator (Encoder) ==========
        self.generator = build_encoder(config).to(device)
        if self.is_master:
            print(f"[Generator] {sum(p.numel() for p in self.generator.parameters()):,} params")
        
        if dist.is_initialized():
            self.generator = DDP(self.generator, device_ids=[device.index])
        
        # ========== Discriminator (DELAYED LOADING) ==========
        # D is NOT initialized in Phase 1 to save memory
        self.discriminator = None
        self.d_optimizer = None
        self.d_scheduler = None
        self._d_initialized = False
        self._gan_cfg = gan_cfg  # Store for later initialization
        self._train_cfg = train_cfg
        
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
            mel_weight=loss_cfg.get('mel_weight', 25.0),
            time_weight=loss_cfg.get('time_domain_weight', 0.1),
            phase_weight=loss_cfg.get('phase_weight', 2.0),
            amp_reg_weight=loss_cfg.get('amp_reg_weight', 0.01),
            pre_emp_weight=loss_cfg.get('pre_emp_weight', 10.0),
        ).to(device)
        
        self.sparsity_weight = loss_cfg.get('sparsity_weight', 0.1)
        self.target_active_ratio = loss_cfg.get('target_active_ratio', 0.15)
        self.fm_weight = gan_cfg.get('fm_weight', 10.0)
        self.adv_weight = gan_cfg.get('adv_weight', 1.0)
        
        # ========== Generator Optimizer ==========
        g_lr = float(train_cfg.get('learning_rate', 2e-4))
        self.g_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=g_lr, betas=(0.8, 0.99))
        
        # ========== Generator Scheduler ==========
        warmup_steps = train_cfg.get('warmup_steps', 1000)
        max_steps = train_cfg.get('max_steps', 100000)
        
        g_warmup = LinearLR(self.g_optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        g_cosine = CosineAnnealingLR(self.g_optimizer, T_max=max_steps - warmup_steps)
        self.g_scheduler = SequentialLR(self.g_optimizer, [g_warmup, g_cosine], [warmup_steps])
        
        # ========== AMP GradScaler ==========
        self.g_scaler = GradScaler()
        
        # ========== Gradient Accumulation ==========
        self.accum_steps = train_cfg.get('accumulation_steps', 1)
        self.accum_count = 0
        
        self.global_step = 0
        self.warmup_freeze_steps = train_cfg.get('warmup_freeze_structure', 3000)
        
        # ========== Staged Training ==========
        self.gan_warmup_steps = gan_cfg.get('warmup_steps', 15000)
        if self.is_master:
            print(f"[Training] Phase 1: {self.gan_warmup_steps} steps (pure recon, batch={self.batch_size_phase1})")
            print(f"[Training] Phase 2: GAN training (batch={self.batch_size_phase2})")
            print(f"[Training] Target active ratio: {self.target_active_ratio*100:.0f}%")
        
        if self.is_master:
            self.visualizer = Visualizer(str(output_dir / "visualizations"), sample_rate=self.sample_rate)
        else:
            self.visualizer = None
        
        self.ref_audio = None
    
    def _init_discriminator(self):
        """Initialize discriminator at Phase 2 start (delayed loading to save memory)."""
        if self._d_initialized:
            return
        
        if self.is_master:
            print(f"\n[Phase 2] Initializing Discriminator...")
        
        # Build and move D to device
        self.discriminator = build_discriminator().to(self.device)
        if self.is_master:
            print(f"[Discriminator] {sum(p.numel() for p in self.discriminator.parameters()):,} params")
        
        if dist.is_initialized():
            self.discriminator = DDP(self.discriminator, device_ids=[self.device.index])
        
        # Create D optimizer
        d_lr = float(self._gan_cfg.get('d_lr', 2e-4))
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=d_lr, betas=(0.8, 0.99))
        
        # Create D scheduler (start from current step)
        max_steps = self._train_cfg.get('max_steps', 100000)
        remaining_steps = max_steps - self.global_step
        self.d_scheduler = CosineAnnealingLR(self.d_optimizer, T_max=remaining_steps)
        
        # Update batch size flag
        self.current_batch_size = self.batch_size_phase2
        
        self._d_initialized = True
        
        if self.is_master:
            print(f"[Phase 2] Batch size reduced: {self.batch_size_phase1} -> {self.batch_size_phase2}")
            print(f"[Phase 2] GAN training active!\n")
        
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
        """
        Staged GAN training step:
        - Phase 1 (step < gan_warmup_steps): Pure reconstruction, no GAN
        - Phase 2 (step >= gan_warmup_steps): Full GAN training
        """
        audio, lengths = batch
        audio = audio.to(self.device, non_blocking=True)
        num_samples = audio.shape[-1]
        
        is_accumulating = (self.accum_count + 1) % self.accum_steps != 0
        use_gan = self.global_step >= self.gan_warmup_steps
        
        # ========== Phase 2 Start: Initialize Discriminator ==========
        if use_gan and not self._d_initialized:
            self._init_discriminator()
        
        # ========== Generator Forward (FP32 for Conformer stability) ==========
        with torch.amp.autocast('cuda', enabled=False):
            enc_output = self.generator(audio.float())
        
        # Progressive unfreezing - freeze structural params in early training
        if self.global_step < self.warmup_freeze_steps:
            enc_output = {
                'amplitude': enc_output['amplitude'],
                'tau': enc_output['tau'].detach(),
                'omega': enc_output['omega'].detach(),
                'sigma': enc_output['sigma'].detach(),
                'phi': enc_output['phi'],
                'gamma': enc_output['gamma'].detach(),
                'existence_mask': enc_output['existence_mask'],
                'existence_prob': enc_output['existence_prob'],
                'num_frames': enc_output['num_frames'],
                'atoms_per_frame': enc_output['atoms_per_frame'],
            }
        
        # Render in FP32
        fake_audio = self.render_batch(enc_output, num_samples)
        
        # ========== NaN Protection ==========
        if not torch.isfinite(fake_audio).all():
            if self.is_master:
                print(f"[Warning] Step {self.global_step}: NaN in fake_audio, skipping")
            self.accum_count += 1
            return self._nan_metrics()
        
        # ========== Target Ratio Sparsity (STRONG penalty) ==========
        active_ratio = (enc_output['existence_prob'] > 0.5).float().mean()
        
        # When too sparse: VERY strong penalty to push atoms to activate
        # When too dense: lighter penalty
        ratio_error = active_ratio - self.target_active_ratio
        if ratio_error < 0:  # Too sparse - VERY STRONG penalty
            # Make this loss ~20-50 when active_ratio is far below target
            # |ratio_error| = 0.17 → loss = 5000 * 0.17^2 = 14.5
            sparsity_loss = 5000.0 * (ratio_error ** 2)
        else:  # Too dense - normal penalty
            sparsity_loss = self.sparsity_weight * ratio_error
        
        # ========== Phase 1: Pure Reconstruction (no GAN) ==========
        if not use_gan:
            with torch.amp.autocast('cuda', enabled=False):
                recon_loss, loss_dict = self.recon_loss_fn(
                    fake_audio.float(), audio.float(),
                    model_amplitude=enc_output['amplitude'].float(),
                    model_sigma=enc_output['sigma'].float()
                )
            
            g_loss = (recon_loss + sparsity_loss) / self.accum_steps
            d_loss = torch.tensor(0.0, device=self.device)
            g_adv_loss = torch.tensor(0.0, device=self.device)
            fm_loss = torch.tensor(0.0, device=self.device)
            
        # ========== Phase 2: Full GAN Training ==========
        else:
            # Discriminator Step
            audio_clamped = audio.float().clamp(-1.0, 1.0)
            fake_clamped = fake_audio.detach().float().clamp(-1.0, 1.0)
            
            with torch.amp.autocast('cuda', enabled=False):
                real_d_out, real_d_feats = self.discriminator(audio_clamped)
                fake_d_out, fake_d_feats = self.discriminator(fake_clamped)
                d_loss = discriminator_loss(real_d_out, fake_d_out) / self.accum_steps
            
            if not torch.isfinite(d_loss):
                if self.is_master:
                    print(f"[Warning] Step {self.global_step}: NaN in d_loss")
                self.accum_count += 1
                return self._nan_metrics()
            
            d_loss.backward()
            
            if not is_accumulating:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()
            
            # Generator Step with GAN losses
            with torch.amp.autocast('cuda', enabled=False):
                fake_d_out_g, fake_d_feats_g = self.discriminator(fake_audio.float().clamp(-1.0, 1.0))
                _, real_d_feats_g = self.discriminator(audio_clamped)
                
                g_adv_loss = generator_loss(fake_d_out_g) * self.adv_weight
                fm_loss = feature_matching_loss(real_d_feats_g, fake_d_feats_g) * self.fm_weight
            
            # Reconstruction loss
            with torch.amp.autocast('cuda', enabled=False):
                recon_loss, loss_dict = self.recon_loss_fn(
                    fake_audio.float(), audio.float(),
                    model_amplitude=enc_output['amplitude'].float(),
                    model_sigma=enc_output['sigma'].float()
                )
            
            # sparsity_loss already computed above (Target Ratio)
            g_loss = (recon_loss + g_adv_loss + fm_loss + sparsity_loss) / self.accum_steps
        
        # Check G loss for NaN
        if not torch.isfinite(g_loss):
            if self.is_master:
                print(f"[Warning] Step {self.global_step}: NaN in g_loss")
            self.accum_count += 1
            return self._nan_metrics()
        
        # G backward (use scaler for potential future AMP)
        self.g_scaler.scale(g_loss).backward()
        
        if not is_accumulating:
            self.g_scaler.unscale_(self.g_optimizer)
            torch.nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=0.5)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
            self.g_scaler.step(self.g_optimizer)
            self.g_scaler.update()
            self.g_optimizer.zero_grad()
            
            self.g_scheduler.step()
            if use_gan:
                self.d_scheduler.step()
        
        self.global_step += 1
        self.accum_count += 1
        
        with torch.no_grad():
            active_ratio_val = (enc_output['existence_prob'] > 0.5).float().mean().item()
            si_sdr = compute_si_sdr(fake_audio, audio).item()
        
        return {
            'g_loss': g_loss.item() * self.accum_steps,
            'd_loss': d_loss.item() * self.accum_steps if use_gan else 0.0,
            'recon_loss': recon_loss.item(),
            'mel_loss': loss_dict.get('mel', 0.0),
            'si_sdr': si_sdr,
            'adv_loss': g_adv_loss.item() if use_gan else 0.0,
            'fm_loss': fm_loss.item() if use_gan else 0.0,
            'sparsity_loss': sparsity_loss.item(),
            'active_ratio': active_ratio_val,
            'g_lr': self.g_optimizer.param_groups[0]['lr'],
            'phase': 'GAN' if use_gan else 'Recon',
        }
    
    def _nan_metrics(self):
        """Return placeholder metrics when NaN is detected."""
        return {
            'g_loss': float('nan'), 'd_loss': float('nan'),
            'recon_loss': float('nan'), 'mel_loss': 0.0, 'si_sdr': -100.0,
            'adv_loss': float('nan'), 'fm_loss': float('nan'),
            'sparsity_loss': float('nan'), 'active_ratio': 0.0,
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
        
        # Compute PESQ on rank 0 only (slow, CPU-based)
        pesq_score = 0.0
        if PESQ_AVAILABLE and self.is_master and cnt > 0:
            try:
                # Use cached audio from last batch for PESQ
                ref_np = audio[0].cpu().numpy()
                deg_np = fake[0].cpu().numpy()
                if self.sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(self.sample_rate, 16000)
                    ref_np = resampler(torch.from_numpy(ref_np)).numpy()
                    deg_np = resampler(torch.from_numpy(deg_np)).numpy()
                pesq_score = pesq(16000, ref_np, deg_np, 'wb')
            except Exception:
                pass
        
        results = {
            'val_loss': (total_loss / cnt).item() if cnt > 0 else 0.0,
            'val_si_sdr': (total_si_sdr / cnt).item() if cnt > 0 else 0.0,
            'val_active_ratio': (total_active / cnt).item() if cnt > 0 else 0.0,
            'val_pesq': pesq_score,
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
        
        # Handle None discriminator (Phase 1)
        d_state = None
        if self.discriminator is not None:
            d_state = self.discriminator.module.state_dict() if isinstance(self.discriminator, DDP) else self.discriminator.state_dict()
        
        ckpt = {
            'global_step': self.global_step,
            'generator_state': g_state,
            'g_optimizer': self.g_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'd_initialized': self._d_initialized,
            'config': self.config,
        }
        
        # Only save D state if initialized
        if d_state is not None:
            ckpt['discriminator_state'] = d_state
        if self.d_optimizer is not None:
            ckpt['d_optimizer'] = self.d_optimizer.state_dict()
        if self.d_scheduler is not None:
            ckpt['d_scheduler'] = self.d_scheduler.state_dict()
        
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
        
        # Handle discriminator loading (may be None in Phase 1 checkpoint)
        if self.discriminator is not None and 'discriminator_state' in ckpt:
            if isinstance(self.discriminator, DDP):
                self.discriminator.module.load_state_dict(ckpt['discriminator_state'])
            else:
                self.discriminator.load_state_dict(ckpt['discriminator_state'])
        
        self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        if self.d_optimizer is not None and 'd_optimizer' in ckpt:
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
        if 'g_scheduler' in ckpt:
            self.g_scheduler.load_state_dict(ckpt['g_scheduler'])
        if self.d_scheduler is not None and 'd_scheduler' in ckpt:
            self.d_scheduler.load_state_dict(ckpt['d_scheduler'])
        self.global_step = ckpt.get('global_step', 0)
        self._d_initialized = ckpt.get('d_initialized', False)
        
        # Update current batch size based on phase
        if self._d_initialized:
            self.current_batch_size = self.batch_size_phase2
        
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
    
    # Samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    
    # Helper function to create DataLoader
    def create_train_loader(batch_size):
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                          sampler=train_sampler, num_workers=4, collate_fn=collate_fn,
                          pin_memory=True, drop_last=True)
    
    # Start with Phase 1 batch size
    current_batch_size = trainer.batch_size_phase1
    train_loader = create_train_loader(current_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=max(1, current_batch_size // 2),
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
        
        # Check if batch size changed (Phase 2 transition)
        if trainer.current_batch_size != current_batch_size:
            current_batch_size = trainer.current_batch_size
            train_loader = create_train_loader(current_batch_size)
            train_iter = iter(train_loader)
            if is_rank_zero():
                tqdm.write(f"\n[Phase Transition] Recreated DataLoader with batch_size={current_batch_size}")
        
        if is_rank_zero():
            pbar.update(1)
            pbar.set_postfix({
                'G': f"{metrics['g_loss']:.1f}",
                'D': f"{metrics['d_loss']:.1f}",
                'Mel': f"{metrics['mel_loss']:.1f}",
                'SDR': f"{metrics['si_sdr']:.1f}",
                'act': f"{metrics['active_ratio']:.0%}",
            })
            
            if args.use_wandb and trainer.global_step % log_interval == 0:
                wandb.log({
                    'train/g_loss': metrics['g_loss'],
                    'train/d_loss': metrics['d_loss'],
                    'train/recon_loss': metrics['recon_loss'],
                    'train/mel_loss': metrics['mel_loss'],
                    'train/si_sdr': metrics['si_sdr'],
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
                pesq_str = f", PESQ={val_metrics['val_pesq']:.2f}" if val_metrics.get('val_pesq', 0) > 0 else ""
                tqdm.write(f"\n[Val] Step {trainer.global_step}: "
                          f"Loss={val_metrics['val_loss']:.2f}, "
                          f"SI-SDR={val_metrics['val_si_sdr']:.1f}dB, "
                          f"Active={val_metrics['val_active_ratio']:.0%}{pesq_str}")
                
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