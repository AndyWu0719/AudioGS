#!/usr/bin/env python
"""
Encoder Training Script for GaborGridEncoder (DDP Enabled)

Features:
- Step-based training loop (not epoch-based)
- PESQ/SI-SDR metrics during validation
- Visualization: spectrogram + audio comparison for fixed reference sample
- dist.all_reduce for global metric averaging

Usage:
    torchrun --nproc_per_node=4 scripts/01_encoder_training/run_encoder_train.py \
        --config configs/AudioGS_config.yaml --use_wandb
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

# Optional quality metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

# CUDA renderer
try:
    from cuda_gabor import get_cuda_gabor_renderer
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    print("[Error] CUDA renderer not available!")


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0


class AudioDataset(Dataset):
    """Dataset for loading audio files from one or more directories."""
    
    def __init__(self, data_paths, sample_rate: int = 24000, 
                 max_length_sec: float = 5.0, min_length_sec: float = 1.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_sec * sample_rate)
        self.min_samples = int(min_length_sec * sample_rate)
        
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
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
        try:
            waveform, sr = torchaudio.load(str(audio_path))
        except Exception:
            return torch.zeros(self.min_samples), self.min_samples
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
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


def compute_si_sdr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        min_len = min(pred.shape[-1], target.shape[-1])
        pred, target = pred[..., :min_len], target[..., :min_len]
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (pred * target).sum(dim=-1, keepdim=True)
    s_target = dot * target / (target.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
    e_noise = pred - s_target
    return 10 * torch.log10(s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + 1e-8) + 1e-8).mean()


def compute_pesq_batch(pred: torch.Tensor, target: torch.Tensor, sr: int) -> float:
    """Compute PESQ for first sample in batch."""
    if not PESQ_AVAILABLE:
        return float('nan')
    try:
        ref = torchaudio.functional.resample(target[0].cpu(), sr, 16000).numpy()
        deg = torchaudio.functional.resample(pred[0].cpu(), sr, 16000).numpy()
        min_len = min(len(ref), len(deg))
        return pesq(16000, ref[:min_len], deg[:min_len], 'wb')
    except:
        return float('nan')


class EncoderTrainer:
    """Training manager for GaborGridEncoder (Step-Based with Visualization)."""
    
    def __init__(self, config: dict, device: torch.device, rank: int, output_dir: Path):
        self.config = config
        self.device = device
        self.rank = rank
        self.is_master = (rank == 0)
        self.output_dir = output_dir
        
        self.data_config = config.get('data', {})
        self.train_config = config.get('training', {})
        self.loss_config = config.get('loss', {})
        self.sample_rate = self.data_config.get('sample_rate', 24000)
        
        # Build encoder
        self.encoder = build_encoder(config).to(device)
        if self.is_master:
            print(f"[Encoder] Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        
        # Wrap DDP
        if dist.is_initialized():
            self.encoder = DDP(self.encoder, device_ids=[device.index],
                               find_unused_parameters=config.get('distributed', {}).get('find_unused_parameters', False))
        
        # Renderer
        if not RENDERER_AVAILABLE:
            raise RuntimeError("CUDA renderer required!")
        self.renderer = get_cuda_gabor_renderer(sample_rate=self.sample_rate)
        
        # Loss
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
        
        self.sparsity_weight = self.loss_config.get('sparsity_weight', 0.001)
        
        # Optimizer
        lr = float(self.train_config.get('learning_rate', 5e-4))
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=1e-4)
        
        # Schedulers
        warmup_steps = self.train_config.get('warmup_steps', 1000)
        max_steps = self.train_config.get('max_steps', 100000)
        warmup_sched = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=max_steps - warmup_steps)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
        
        self.global_step = 0
        
        # Visualizer (Rank 0 only)
        if self.is_master:
            self.visualizer = Visualizer(str(output_dir / "visualizations"), sample_rate=self.sample_rate)
        else:
            self.visualizer = None
        
        # Fixed reference sample for consistent visualization (set later)
        self.ref_audio = None
        
    def set_reference_sample(self, audio: torch.Tensor):
        """Set a fixed reference sample for visualization during validation."""
        self.ref_audio = audio.to(self.device)
        if self.is_master:
            print(f"[Trainer] Reference sample set: {audio.shape[-1] / self.sample_rate:.2f}s")
    
    def render_batch(self, enc_output: dict, num_samples: int) -> torch.Tensor:
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
    
    def train_step(self, batch: torch.Tensor) -> dict:
        audio, lengths = batch
        audio = audio.to(self.device, non_blocking=True)
        num_samples = audio.shape[-1]
        
        enc_output = self.encoder(audio)
        reconstructed = self.render_batch(enc_output, num_samples)
        
        loss, loss_dict = self.loss_fn(reconstructed, audio,
                                       model_amplitude=enc_output['amplitude'],
                                       model_sigma=enc_output['sigma'])
        
        # Sparsity loss
        sparsity_loss = self.sparsity_weight * enc_output['existence_prob'].mean()
        loss = loss + sparsity_loss
        loss_dict['sparsity'] = sparsity_loss.item()
        loss_dict['total'] = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        
        with torch.no_grad():
            active_ratio = (enc_output['existence_prob'] > 0.5).float().mean().item()
        
        return {'loss': loss.item(), 'active_ratio': active_ratio, 'lr': self.optimizer.param_groups[0]['lr'],
                **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, max_batches: int = 20) -> dict:
        """Run validation with metrics and visualization."""
        self.encoder.eval()
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_si_sdr = torch.tensor(0.0, device=self.device)
        total_active = torch.tensor(0.0, device=self.device)
        total_pesq = 0.0
        pesq_count = 0
        count = torch.tensor(0, device=self.device)
        
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            audio, lengths = batch
            audio = audio.to(self.device)
            num_samples = audio.shape[-1]
            
            enc_output = self.encoder(audio)
            reconstructed = self.render_batch(enc_output, num_samples)
            
            loss, _ = self.loss_fn(reconstructed, audio)
            total_loss += loss
            total_si_sdr += compute_si_sdr(reconstructed, audio)
            total_active += (enc_output['existence_prob'] > 0.5).float().mean()
            count += 1
            
            # PESQ (expensive, only first 3 batches)
            if self.is_master and i < 3:
                p = compute_pesq_batch(reconstructed, audio, self.sample_rate)
                if not np.isnan(p):
                    total_pesq += p
                    pesq_count += 1
        
        # All-reduce
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
        if pesq_count > 0:
            results['val_pesq'] = total_pesq / pesq_count
        
        # Visualization on fixed reference sample (Rank 0 only)
        if self.is_master and self.ref_audio is not None and self.visualizer is not None:
            ref_input = self.ref_audio.unsqueeze(0) if self.ref_audio.dim() == 1 else self.ref_audio
            enc_out = self.encoder(ref_input)
            ref_recon = self.render_batch(enc_out, ref_input.shape[-1])[0]
            
            step_str = f"step_{self.global_step:06d}"
            self.visualizer.save_audio(ref_recon, f"{step_str}_recon")
            self.visualizer.plot_spectrogram_comparison(
                self.ref_audio, ref_recon, f"{step_str}_spectrogram",
                title=f"Step {self.global_step}"
            )
        
        self.encoder.train()
        return results
    
    def save_checkpoint(self, path: str, extra: dict = None):
        if not self.is_master:
            return
        model_state = self.encoder.module.state_dict() if isinstance(self.encoder, DDP) else self.encoder.state_dict()
        ckpt = {'global_step': self.global_step, 'encoder_state': model_state,
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(), 'config': self.config}
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f"[Checkpoint] Saved to {path}")
        
    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(self.encoder, DDP):
            self.encoder.module.load_state_dict(ckpt['encoder_state'])
        else:
            self.encoder.load_state_dict(ckpt['encoder_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.global_step = ckpt.get('global_step', 0)
        if self.is_master:
            print(f"[Checkpoint] Loaded from step {self.global_step}")


def train(args, config):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    output_dir = Path(f"logs/encoder_{args.exp_name}")
    if is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sync before creating trainer to ensure dir exists
    if dist.is_initialized():
        dist.barrier()
    
    if is_rank_zero() and args.use_wandb:
        wandb.init(project="AudioGS-Encoder", config=config, name=args.exp_name)
    
    trainer = EncoderTrainer(config, device, rank, output_dir)
    
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
    val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size // 2), shuffle=False,
                             sampler=val_sampler, num_workers=2, collate_fn=collate_fn)
    
    # Set fixed reference sample for visualization (from first val batch)
    if is_rank_zero() and len(val_dataset) > 0:
        ref_audio, _ = val_dataset[0]  # First validation sample
        trainer.set_reference_sample(ref_audio)
        trainer.visualizer.save_audio(ref_audio, "reference_gt")
    
    max_steps = config['training'].get('max_steps', 100000)
    val_interval = config['training'].get('val_interval', 2000)
    save_interval = config['training'].get('save_interval', 10000)
    log_interval = config['training'].get('log_interval', 10)
    
    if is_rank_zero():
        print(f"\n{'='*60}")
        print(f"GaborGridEncoder Training (Step-Based)")
        print(f"{'='*60}")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, GPUs: {world_size}")
        print(f"Max steps: {max_steps}, Output: {output_dir}")
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
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 'active': f"{metrics['active_ratio']:.1%}",
                              'lr': f"{metrics['lr']:.2e}"})
            
            if args.use_wandb and trainer.global_step % log_interval == 0:
                wandb.log({'train/loss': metrics['loss'], 'train/active_ratio': metrics['active_ratio'],
                           'train/sparsity_loss': metrics.get('sparsity', 0.0), 'train/lr': metrics['lr'],
                           'epoch': epoch, 'step': trainer.global_step})
        
        if trainer.global_step % val_interval == 0 and trainer.global_step > 0:
            val_metrics = trainer.validate(val_loader)
            if is_rank_zero():
                msg = f"\n[Val] Step {trainer.global_step}: Loss={val_metrics['val_loss']:.4f}, SI-SDR={val_metrics['val_si_sdr']:.2f}, Active={val_metrics['val_active_ratio']:.1%}"
                if 'val_pesq' in val_metrics:
                    msg += f", PESQ={val_metrics['val_pesq']:.3f}"
                tqdm.write(msg)
                
                if args.use_wandb:
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()}, step=trainer.global_step)
                
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    trainer.save_checkpoint(str(output_dir / "best_model.pt"), extra={'val_metrics': val_metrics})
        
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
    parser.add_argument("--config", type=str, default="configs/AudioGS_config.yaml")
    parser.add_argument("--exp_name", type=str, default="v1")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train(args, config)


if __name__ == "__main__":
    main()