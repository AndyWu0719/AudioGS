"""
Flow Matching Training Script for Audio Gaussian Splatting.

Trains a Flow Matching DiT to generate Gabor atom parameters from text.

Usage:
    conda activate qwen2_CALM
    python scripts/train_flow.py --config configs/flow_config.yaml
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not available, install with: pip install wandb")

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

from data.dataset import LibriTTSAtomDataset, collate_atoms, AtomNormalizer, get_flow_dataloader
from data.text_encoder import CharacterTokenizer, TextEncoder, SpeakerEncoder
from models.flow_dit import FlowDiT, get_flow_model
from models.flow_matching import ConditionalFlowMatching, FlowODESolver

# Optional: CUDA renderer for audio generation
try:
    from cuda_gabor import get_cuda_gabor_renderer
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    print("[Warning] CUDA renderer not available, audio generation disabled")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flow Matching Model")
    parser.add_argument("--config", type=str, default="configs/flow_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


class FlowTrainer:
    """Flow Matching Trainer."""
    
    def __init__(
        self,
        config: Dict,
        device: torch.device,
        output_dir: Path,
    ):
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Initialize components
        self._init_data()
        self._init_models()
        self._init_training()
        self._init_logging()
        
        # State
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _init_logging(self):
        """Initialize logging (TensorBoard and WandB)."""
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # WandB
        log_cfg = self.config.get('logging', {})
        self.use_wandb = log_cfg.get('use_wandb', False) and WANDB_AVAILABLE
        
        if self.use_wandb:
            wandb.init(
                project=log_cfg.get('wandb_project', 'AudioGS-Flow'),
                entity=log_cfg.get('wandb_entity'),
                name=log_cfg.get('wandb_run_name') or self.output_dir.name,
                config=self.config,
                dir=str(self.output_dir),
            )
            print(f"[WandB] Initialized: {wandb.run.name}")
        else:
            if log_cfg.get('use_wandb', False):
                print("[Warning] WandB requested but not available")
    
    def _init_data(self):
        """Initialize data loaders."""
        data_cfg = self.config['data']
        
        # Build dataset (will scan for speaker IDs)
        print("[Data] Loading training dataset...")
        total_atoms = data_cfg.get('total_atoms', 20480)
        
        self.train_loader, self.train_dataset = get_flow_dataloader(
            data_dir=data_cfg['atom_dir'],
            batch_size=data_cfg['batch_size'],
            max_atoms=total_atoms,  # Fixed size for anchor-based
            split='train',
            num_workers=data_cfg['num_workers'],
            val_ratio=data_cfg['val_ratio'],
        )
        
        print("[Data] Loading validation dataset...")
        self.val_loader, self.val_dataset = get_flow_dataloader(
            data_dir=data_cfg['atom_dir'],
            batch_size=data_cfg['batch_size'],
            max_atoms=total_atoms,  # Fixed size for anchor-based
            split='val',
            num_workers=data_cfg['num_workers'],
            val_ratio=data_cfg['val_ratio'],
            speaker_to_id=self.train_dataset.speaker_to_id,  # Share speaker mapping
        )
        
        # Tokenizer
        self.tokenizer = CharacterTokenizer()
        
        # Atom normalizer
        self.normalizer = AtomNormalizer(sample_rate=self.config['output']['sample_rate'])
        
        print(f"[Data] Train samples: {len(self.train_dataset)}")
        print(f"[Data] Val samples: {len(self.val_dataset)}")
        print(f"[Data] Speakers: {self.train_dataset.num_speakers}")
    
    def _init_models(self):
        """Initialize models."""
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
        
        # Flow model (DiT) - anchor-based architecture
        num_anchors = model_cfg.get('num_anchors', 2560)
        split_factor = model_cfg.get('split_factor', 8)
        
        self.model = get_flow_model(
            size=model_cfg['size'],
            num_speakers=self.train_dataset.num_speakers + 10,  # +10 for safety
            text_dim=text_cfg['hidden_dim'],
            num_anchors=num_anchors,
            split_factor=split_factor,
        ).to(self.device)
        
        print(f"[Model] Anchors: {num_anchors} × {split_factor} = {num_anchors * split_factor} atoms")
        
        # CFM loss with OT coupling for straighter flows
        flow_cfg = self.config.get('flow', {})
        use_ot = flow_cfg.get('use_ot', True)
        self.cfm = ConditionalFlowMatching(
            sigma_min=flow_cfg.get('sigma_min', 1e-4),
            use_ot=use_ot,
        )
        print(f"[Flow] OT coupling: {self.cfm.use_ot}")
        
        # ODE solver for sampling
        self.solver = FlowODESolver(self.model, sigma_min=flow_cfg.get('sigma_min', 1e-4))
        
        # Count parameters
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        model_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] Text encoder: {text_params:,} params")
        print(f"[Model] Flow DiT: {model_params:,} params")
        print(f"[Model] Total: {text_params + model_params:,} params")
        
        # Optional: renderer for audio generation
        if RENDERER_AVAILABLE:
            self.renderer = get_cuda_gabor_renderer(
                sample_rate=self.config['output']['sample_rate']
            )
        else:
            self.renderer = None
    
    def _init_training(self):
        """Initialize optimizer and scheduler."""
        train_cfg = self.config['training']
        opt_cfg = self.config['optimizer']
        
        # Combine all trainable parameters
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
        
        # Move to device
        atoms = batch['atoms'].to(self.device)  # [B, N, 6] - fixed size
        atom_mask = batch['atom_mask']  # None for fixed-size
        speaker_ids = batch['speaker_ids'].to(self.device)  # [B]
        transcripts = batch['transcripts']  # List[str]
        
        # Normalize atoms for flow matching
        atoms_norm = self.normalizer.normalize(atoms)
        
        # Encode text
        text_batch = self.tokenizer.batch_encode(
            transcripts, 
            max_length=256,
            return_tensors=True,
        )
        input_ids = text_batch['input_ids'].to(self.device)
        attention_mask = text_batch['attention_mask'].to(self.device)
        
        text_emb, _ = self.text_encoder(input_ids, attention_mask)
        
        # Forward model wrapper for CFM
        def model_fn(xt, t, **kwargs):
            return self.model(
                xt, t,
                speaker_ids=speaker_ids,
                text_embeddings=text_emb,
                text_mask=attention_mask.bool(),
                atom_mask=None,  # Not needed for fixed-size anchor-based
            )
        
        # Compute CFM loss (no mask needed for fixed size)
        loss, info = self.cfm.compute_loss(model_fn, atoms_norm, mask=None)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
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
        """Run validation."""
        self.model.eval()
        self.text_encoder.eval()
        
        total_loss = 0
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
                    atom_mask=None,  # Fixed-size, no mask needed
                )
            
            loss, _ = self.cfm.compute_loss(model_fn, atoms_norm, mask=None)
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 50:  # Limit validation batches
                break
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 4):
        """Generate samples for visualization."""
        self.model.eval()
        self.text_encoder.eval()
        
        # Get a validation batch
        batch = next(iter(self.val_loader))
        
        transcripts = batch['transcripts'][:num_samples]
        speaker_ids = batch['speaker_ids'][:num_samples].to(self.device)
        durations = batch['durations'][:num_samples]
        
        # Encode text
        text_batch = self.tokenizer.batch_encode(transcripts, max_length=256, return_tensors=True)
        input_ids = text_batch['input_ids'].to(self.device)
        attention_mask = text_batch['attention_mask'].to(self.device)
        
        text_emb, _ = self.text_encoder(input_ids, attention_mask)
        
        # Generate using RK4 solver (faster with OT coupling)
        total_atoms = self.model.total_atoms
        num_steps = self.config['training'].get('num_sampling_steps', 25)
        solver_method = self.config.get('flow', {}).get('solver_method', 'rk4')
        
        generated = self.solver.sample(
            shape=(num_samples, total_atoms, 6),
            num_steps=num_steps,
            method=solver_method,
            device=self.device,
            speaker_ids=speaker_ids,
            text_embeddings=text_emb,
            text_mask=attention_mask.bool(),
        )
        
        # Denormalize
        generated = self.normalizer.denormalize(generated)
        
        return generated, transcripts, durations
    
    def save_checkpoint(self, name: str = "latest"):
        """Save checkpoint."""
        ckpt = {
            'step': self.global_step,
            'model': self.model.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'speaker_to_id': self.train_dataset.speaker_to_id,
        }
        torch.save(ckpt, self.output_dir / "checkpoints" / f"{name}.pt")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.text_encoder.load_state_dict(ckpt['text_encoder'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.global_step = ckpt['step']
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"[Checkpoint] Loaded from {path} at step {self.global_step}")
    
    def train(self):
        """Full training loop with wandb and improved progress bar."""
        train_cfg = self.config['training']
        max_steps = train_cfg['max_steps']
        
        # Calculate epoch info
        steps_per_epoch = len(self.train_loader)
        total_epochs = max_steps / steps_per_epoch
        
        print(f"\n{'='*60}")
        print("Starting Flow Matching Training")
        print(f"{'='*60}")
        print(f"Output: {self.output_dir}")
        print(f"Max steps: {max_steps}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total epochs: {total_epochs:.1f}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.name}")
        print(f"{'='*60}\n")
        
        # Progress bar with epoch tracking
        pbar = tqdm(
            total=max_steps, 
            initial=self.global_step, 
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )
        
        epoch = 0
        while self.global_step < max_steps:
            epoch += 1
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break
                
                # Train step
                info = self.train_step(batch)
                self.global_step += 1
                
                # Logging
                if self.global_step % train_cfg['log_interval'] == 0:
                    # TensorBoard
                    self.writer.add_scalar('train/loss', info['cfm_loss'], self.global_step)
                    self.writer.add_scalar('train/grad_norm', info['grad_norm'], self.global_step)
                    self.writer.add_scalar('train/lr', info['lr'], self.global_step)
                    
                    # WandB
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
                    
                    # TensorBoard
                    self.writer.add_scalar('val/loss', val_loss, self.global_step)
                    
                    # WandB
                    if self.use_wandb:
                        wandb.log({'val/loss': val_loss}, step=self.global_step)
                    
                    tqdm.write(f"[Step {self.global_step}] Epoch {epoch} | Val Loss: {val_loss:.4f}")
                    
                    # Save best
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best")
                        tqdm.write(f"[Step {self.global_step}] New best! Loss: {val_loss:.4f}")
                
                # Save checkpoint
                if self.global_step % train_cfg['save_interval'] == 0:
                    self.save_checkpoint("latest")
                    self.save_checkpoint(f"step_{self.global_step}")
                
                pbar.update(1)
        
        pbar.close()
        self.save_checkpoint("final")
        
        # Finish wandb
        if self.use_wandb:
            wandb.finish()
        print("\n[Training] Complete!")


def main():
    args = parse_args()
    
    # Load config
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(str(config_path))
    
    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Output directory
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = PROJECT_ROOT / config['output']['dir'] / exp_name
    
    print("=" * 60)
    print("Flow Matching TTS Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Create trainer
    trainer = FlowTrainer(config, device, output_dir)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
