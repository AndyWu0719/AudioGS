"""
Multi-GPU Flow Matching Training Script using DistributedDataParallel.

Trains Flow Matching DiT on multiple GPUs for 4x faster training.

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 scripts/train_flow_ddp.py --config configs/flow_config.yaml
    
    # Or with specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_flow_ddp.py
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

from data.dataset import LibriTTSAtomDataset, collate_atoms, AtomNormalizer
from data.text_encoder import CharacterTokenizer, TextEncoder
from models.flow_dit import get_flow_model
from models.flow_matching import ConditionalFlowMatching, FlowODESolver

# Optional: CUDA renderer
try:
    from cuda_gabor import get_cuda_gabor_renderer
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU Flow Matching Training")
    parser.add_argument("--config", type=str, default="configs/flow_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


class DDPFlowTrainer:
    """Multi-GPU Flow Matching Trainer using DDP."""
    
    def __init__(
        self,
        config: Dict,
        rank: int,
        local_rank: int,
        world_size: int,
        output_dir: Path,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.output_dir = output_dir
        self.is_main = is_main_process()
        
        # Create output directories (only main process)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "checkpoints").mkdir(exist_ok=True)
            (self.output_dir / "samples").mkdir(exist_ok=True)
            
            with open(self.output_dir / "config.yaml", "w") as f:
                yaml.dump(config, f)
        
        # Synchronize before proceeding
        if world_size > 1:
            dist.barrier()
        
        # Initialize components
        self._init_data()
        self._init_models()
        self._init_training()
        self._init_logging()
        
        # State
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _init_logging(self):
        """Initialize logging (only main process)."""
        if self.is_main:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
            
            log_cfg = self.config.get('logging', {})
            self.use_wandb = log_cfg.get('use_wandb', False) and WANDB_AVAILABLE
            
            if self.use_wandb:
                wandb.init(
                    project=log_cfg.get('wandb_project', 'AudioGS-Flow'),
                    entity=log_cfg.get('wandb_entity'),
                    name=log_cfg.get('wandb_run_name') or f"{self.output_dir.name}_ddp{self.world_size}",
                    config={**self.config, 'world_size': self.world_size},
                    dir=str(self.output_dir),
                )
                print(f"[WandB] Initialized: {wandb.run.name}")
        else:
            self.writer = None
            self.use_wandb = False
    
    def _init_data(self):
        """Initialize distributed data loaders."""
        data_cfg = self.config['data']
        total_atoms = data_cfg.get('total_atoms', 16384)
        
        # Create datasets
        train_dataset = LibriTTSAtomDataset(
            data_dir=data_cfg['atom_dir'],
            max_atoms=total_atoms,
            split='train',
            val_ratio=data_cfg['val_ratio'],
        )
        
        val_dataset = LibriTTSAtomDataset(
            data_dir=data_cfg['atom_dir'],
            max_atoms=total_atoms,
            split='val',
            val_ratio=data_cfg['val_ratio'],
            speaker_to_id=train_dataset.speaker_to_id,
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Distributed sampler
        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        
        self.val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        
        # Create dataloaders
        # Per-GPU batch size
        per_gpu_batch = data_cfg['batch_size']
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch,
            sampler=self.train_sampler,
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            collate_fn=lambda b: collate_atoms(b, max_atoms=total_atoms),
            drop_last=True,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_batch,
            sampler=self.val_sampler,
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            collate_fn=lambda b: collate_atoms(b, max_atoms=total_atoms),
        )
        
        # Tokenizer and normalizer
        self.tokenizer = CharacterTokenizer()
        self.normalizer = AtomNormalizer(
            sample_rate=self.config.get('data', {}).get('sample_rate', 24000)
        )
        
        if self.is_main:
            effective_batch = per_gpu_batch * self.world_size
            print(f"[Data] Train samples: {len(train_dataset)}")
            print(f"[Data] Val samples: {len(val_dataset)}")
            print(f"[Data] Speakers: {train_dataset.num_speakers}")
            print(f"[Data] Per-GPU batch: {per_gpu_batch}")
            print(f"[Data] Effective batch: {effective_batch} (x{self.world_size} GPUs)")
    
    def _init_models(self):
        """Initialize models with DDP."""
        model_cfg = self.config['model']
        text_cfg = self.config['text_encoder']
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=text_cfg['embed_dim'],
            hidden_dim=text_cfg['hidden_dim'],
            num_layers=text_cfg['num_layers'],
            num_heads=text_cfg['num_heads'],
        ).to(self.device)
        
        # Flow model
        num_anchors = model_cfg.get('num_anchors', 2048)
        split_factor = model_cfg.get('split_factor', 8)
        
        self.model = get_flow_model(
            size=model_cfg['size'],
            num_speakers=self.train_dataset.num_speakers + 10,
            text_dim=text_cfg['hidden_dim'],
            num_anchors=num_anchors,
            split_factor=split_factor,
        ).to(self.device)
        
        # Wrap with DDP
        if self.world_size > 1:
            self.text_encoder = DDP(
                self.text_encoder,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        
        # CFM loss with OT
        flow_cfg = self.config.get('flow', {})
        self.cfm = ConditionalFlowMatching(
            sigma_min=flow_cfg.get('sigma_min', 1e-4),
            use_ot=flow_cfg.get('use_ot', True),
        )
        
        # Solver (uses unwrapped model)
        model_for_solver = self.model.module if hasattr(self.model, 'module') else self.model
        self.solver = FlowODESolver(model_for_solver, sigma_min=flow_cfg.get('sigma_min', 1e-4))
        
        if self.is_main:
            print(f"[Model] Anchors: {num_anchors} × {split_factor} = {num_anchors * split_factor} atoms")
            print(f"[Flow] OT coupling: {self.cfm.use_ot}")
            
            # Count parameters
            model_for_count = self.model.module if hasattr(self.model, 'module') else self.model
            text_for_count = self.text_encoder.module if hasattr(self.text_encoder, 'module') else self.text_encoder
            text_params = sum(p.numel() for p in text_for_count.parameters())
            model_params = sum(p.numel() for p in model_for_count.parameters())
            print(f"[Model] Text encoder: {text_params:,} params")
            print(f"[Model] Flow DiT: {model_params:,} params")
            print(f"[Model] Total: {text_params + model_params:,} params")
    
    def _init_training(self):
        """Initialize optimizer and scheduler."""
        train_cfg = self.config['training']
        opt_cfg = self.config['optimizer']
        
        all_params = list(self.text_encoder.parameters()) + list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
            betas=tuple(opt_cfg['betas']),
            eps=opt_cfg['eps'],
        )
        
        # Linear warmup + cosine decay
        def lr_lambda(step):
            warmup = train_cfg['warmup_steps']
            max_steps = train_cfg['max_steps']
            
            if step < warmup:
                return step / warmup
            else:
                progress = (step - warmup) / (max_steps - warmup)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict) -> Dict:
        """Single training step."""
        self.model.train()
        self.text_encoder.train()
        
        atoms = batch['atoms'].to(self.device)
        speaker_ids = batch['speaker_ids'].to(self.device)
        transcripts = batch['transcripts']
        
        atoms_norm = self.normalizer.normalize(atoms)
        
        text_batch = self.tokenizer.batch_encode(transcripts, max_length=256, return_tensors=True)
        input_ids = text_batch['input_ids'].to(self.device)
        attention_mask = text_batch['attention_mask'].to(self.device)
        
        text_emb, _ = self.text_encoder(input_ids, attention_mask)
        
        def model_fn(xt, t, **kwargs):
            return self.model(
                xt, t,
                speaker_ids=speaker_ids,
                text_embeddings=text_emb,
                text_mask=attention_mask.bool(),
                atom_mask=None,
            )
        
        loss, info = self.cfm.compute_loss(model_fn, atoms_norm, mask=None)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.text_encoder.parameters()),
            self.config['training']['grad_clip'],
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        info['grad_norm'] = grad_norm.item()
        info['lr'] = self.scheduler.get_last_lr()[0]
        
        return info
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation (all GPUs participate, main aggregates)."""
        self.model.eval()
        self.text_encoder.eval()
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        
        for batch in self.val_loader:
            atoms = batch['atoms'].to(self.device)
            speaker_ids = batch['speaker_ids'].to(self.device)
            transcripts = batch['transcripts']
            
            atoms_norm = self.normalizer.normalize(atoms)
            
            text_batch = self.tokenizer.batch_encode(transcripts, max_length=256, return_tensors=True)
            input_ids = text_batch['input_ids'].to(self.device)
            attention_mask = text_batch['attention_mask'].to(self.device)
            
            text_emb, _ = self.text_encoder(input_ids, attention_mask)
            
            def model_fn(xt, t, **kwargs):
                return self.model(
                    xt, t,
                    speaker_ids=speaker_ids,
                    text_embeddings=text_emb,
                    text_mask=attention_mask.bool(),
                    atom_mask=None,
                )
            
            loss, _ = self.cfm.compute_loss(model_fn, atoms_norm, mask=None)
            total_loss += loss
            num_batches += 1
            
            if num_batches >= 20:  # Limit validation batches
                break
        
        # All-reduce loss across GPUs
        if self.world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss / self.world_size
        
        return (total_loss / max(num_batches, 1)).item()
    
    def save_checkpoint(self, name: str = "latest"):
        """Save checkpoint (main process only)."""
        if not self.is_main:
            return
        
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        text_state = self.text_encoder.module.state_dict() if hasattr(self.text_encoder, 'module') else self.text_encoder.state_dict()
        
        ckpt = {
            'step': self.global_step,
            'model': model_state,
            'text_encoder': text_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'speaker_to_id': self.train_dataset.speaker_to_id,
        }
        torch.save(ckpt, self.output_dir / "checkpoints" / f"{name}.pt")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        text_to_load = self.text_encoder.module if hasattr(self.text_encoder, 'module') else self.text_encoder
        
        model_to_load.load_state_dict(ckpt['model'])
        text_to_load.load_state_dict(ckpt['text_encoder'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.global_step = ckpt['step']
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        
        if self.is_main:
            print(f"[Checkpoint] Loaded from {path} at step {self.global_step}")
    
    def train(self):
        """Full training loop with DDP."""
        train_cfg = self.config['training']
        max_steps = train_cfg['max_steps']
        
        steps_per_epoch = len(self.train_loader)
        total_epochs = max_steps / steps_per_epoch
        
        if self.is_main:
            print(f"\n{'='*60}")
            print(f"Starting DDP Flow Matching Training ({self.world_size} GPUs)")
            print(f"{'='*60}")
            print(f"Output: {self.output_dir}")
            print(f"Max steps: {max_steps}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Total epochs: {total_epochs:.1f}")
            print(f"{'='*60}\n")
        
        # Progress bar (main process only)
        pbar = tqdm(
            total=max_steps,
            initial=self.global_step,
            desc=f"Training (GPU {self.rank})" if self.is_main else None,
            disable=not self.is_main,
            unit="step",
            dynamic_ncols=True,
        )
        
        epoch = 0
        while self.global_step < max_steps:
            epoch += 1
            self.train_sampler.set_epoch(epoch)  # Important for shuffling
            
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break
                
                info = self.train_step(batch)
                self.global_step += 1
                
                # Logging (main process only)
                if self.is_main and self.global_step % train_cfg['log_interval'] == 0:
                    self.writer.add_scalar('train/loss', info['cfm_loss'], self.global_step)
                    self.writer.add_scalar('train/grad_norm', info['grad_norm'], self.global_step)
                    self.writer.add_scalar('train/lr', info['lr'], self.global_step)
                    
                    if self.use_wandb:
                        wandb.log({
                            'train/loss': info['cfm_loss'],
                            'train/grad_norm': info['grad_norm'],
                            'train/lr': info['lr'],
                            'epoch': epoch,
                        }, step=self.global_step)
                    
                    pbar.set_postfix({
                        'epoch': epoch,
                        'loss': f"{info['cfm_loss']:.4f}",
                        'lr': f"{info['lr']:.2e}",
                    })
                
                # Validation
                if self.global_step % train_cfg['val_interval'] == 0:
                    val_loss = self.validate()
                    
                    if self.is_main:
                        self.writer.add_scalar('val/loss', val_loss, self.global_step)
                        
                        if self.use_wandb:
                            wandb.log({'val/loss': val_loss}, step=self.global_step)
                        
                        tqdm.write(f"[Step {self.global_step}] Epoch {epoch} | Val Loss: {val_loss:.4f}")
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best")
                            tqdm.write(f"[Step {self.global_step}] New best! Loss: {val_loss:.4f}")
                
                # Save checkpoint
                if self.global_step % train_cfg['save_interval'] == 0:
                    self.save_checkpoint("latest")
                    self.save_checkpoint(f"step_{self.global_step}")
                
                if self.is_main:
                    pbar.update(1)
        
        pbar.close()
        self.save_checkpoint("final")
        
        if self.is_main:
            if self.use_wandb:
                wandb.finish()
            print("\n[Training] Complete!")


def main():
    args = parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    # Load config
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(str(config_path))
    
    # Output directory
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"flow_ddp{world_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = PROJECT_ROOT / config['output']['dir'] / exp_name
    
    if is_main_process():
        print("=" * 60)
        print(f"Multi-GPU Flow Matching Training ({world_size} GPUs)")
        print("=" * 60)
        print(f"Config: {config_path}")
        print(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
        print(f"Output: {output_dir}")
        print("=" * 60)
    
    # Create trainer
    trainer = DDPFlowTrainer(config, rank, local_rank, world_size, output_dir)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
