"""
Training Script for Audio Gaussian Splatting (AGS).
Revised Version: Enforces Single-Instance Optimization & Scheduler.

This script optimizes a set of Gabor atoms to reconstruct a SPECIFIC target audio file.
Even if a dataset path is provided, it will pick the first sample and overfit to it.

Features:
- Distributed Data Parallel (DDP) support
- STFT-guided Initialization (Phase & Frequency)
- Adaptive Density control (Split/Clone/Prune)
- MultiStep Learning Rate Scheduler (Stability)
- Periodic visualization and checkpointing
"""

import os
import sys
import argparse
import yaml
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models and utils
# Ensure these files are present in your project structure
from models.atom import AudioGSModel
from models.renderer import GaborRenderer
from losses.spectral_loss import CombinedAudioLoss
from utils.data_loader import get_dataloader
from utils.density_control import AdaptiveDensityController, rebuild_optimizer_from_model
from utils.visualization import Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audio Gaussian Splatting Training")
    
    # Data arguments
    parser.add_argument(
        "--target_file",
        type=str,
        default=None,
        help="Path to single wav file for overfitting (Recommended mode)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data0/determined/users/andywu/workplace/data/raw/LibriTTS_R",
        help="Path to LibriTTS_R dataset"
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/AudioGS_config.yaml",
        help="Path to config YAML file"
    )
    
    # Training arguments
    parser.add_argument("--max_iters", type=int, default=None, help="Override max iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Force batch size (default 1)")
    parser.add_argument("--lr", type=float, default=None, help="Override base learning rate")
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs",
        help="Output directory for checkpoints and visualizations"
    )
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    
    # Distributed
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs (reference)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """Setup distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return True, rank, world_size, local_rank
    
    return False, 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args, config: Dict):
    """Main training function."""
    
    # --- 1. Setup Environment ---
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
    else:
        device = torch.device("cpu")
    
    # Create experiment directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.target_file:
            filename = Path(args.target_file).stem
            exp_name = f"single_{filename}_{timestamp}"
        else:
            exp_name = f"dataset_sample_{timestamp}"
    else:
        exp_name = args.exp_name
    
    output_dir = Path(args.output_dir) / exp_name
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Train] Output directory: {output_dir}")
        print(f"[Train] Using device: {device}")

    # Config shortcuts
    sample_rate = config["data"]["sample_rate"]
    max_iters = args.max_iters or config["training"]["max_iters"]
    
    # --- 2. Data Loading (Single Instance Lock) ---
    # We acquire a SINGLE target waveform and stick to it.
    dataloader, sampler = get_dataloader(
        root_path=args.data_path,
        target_file=args.target_file,
        batch_size=1, # Strict 1
        sample_rate=sample_rate,
        max_audio_length=config["data"]["max_audio_length"],
        subsets=config["data"].get("subsets"),
        shuffle=False,
    )
    
    # Fetch the FIRST batch and lock it
    try:
        fixed_batch = next(iter(dataloader))
    except StopIteration:
        print("[Error] Dataset is empty!")
        return

    # Move Ground Truth to Device
    gt_waveform = fixed_batch["waveforms"][0].to(device) # [T]
    num_samples = int(fixed_batch["lengths"][0].item())
    gt_waveform = gt_waveform[:num_samples] # Trim padding
    audio_duration = num_samples / sample_rate
    
    if is_main_process:
        print(f"\n[Train] TARGET LOCKED: {fixed_batch['file_paths'][0]}")
        print(f"[Train] Duration: {audio_duration:.2f}s ({num_samples} samples)")

    # --- 3. Model Initialization ---
    model = AudioGSModel(
        num_atoms=config["model"]["initial_num_atoms"],
        sample_rate=sample_rate,
        audio_duration=audio_duration,
        device=device,
    )
    
    # STFT Guided Initialization
    if is_main_process:
        print(f"[Train] Initializing atoms using STFT guidance...")
    model.initialize_from_audio(gt_waveform)
    
    renderer = GaborRenderer(sample_rate=sample_rate)
    
    # --- 4. Loss & Optimizer ---
    loss_config = config["loss"]
    loss_fn = CombinedAudioLoss(
        sample_rate=sample_rate,
        fft_sizes=loss_config["fft_sizes"],
        hop_sizes=loss_config["hop_sizes"],
        win_lengths=loss_config["win_lengths"],
        stft_weight=loss_config.get("spectral_weight", 1.0),
        mel_weight=loss_config.get("mel_weight", 0.5),
        time_weight=loss_config.get("time_domain_weight", 0.1),
        amp_reg_weight=loss_config.get("amp_reg_weight", 0.01),
    )
    
    # Check Learning Rates (Warning for user)
    lr_config = config["training"]
    if is_main_process and lr_config.get("lr_frequency", 0) < 0.002:
        print("\n[WARNING] lr_frequency is very low (< 0.002). High frequency details may be lost.")
        print("[WARNING] Please update AudioGS_config.yaml with recommended values (freq=0.005, phase=0.02).")

    optimizer = rebuild_optimizer_from_model(
        model,
        optimizer_class=torch.optim.Adam,
        lr_config=lr_config,
    )
    
    # --- 5. Scheduler & Density Control ---
    # Scheduler: Reduces LR at 40%, 70%, 90% of training to stabilize fine details
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[4000, 7000, 9000], 
        gamma=0.5
    )
    
    density_config = config["density_control"]
    density_controller = AdaptiveDensityController(
        grad_threshold=density_config["grad_threshold"],
        sigma_split_threshold=density_config["sigma_split_threshold"],
        prune_amplitude_threshold=density_config["prune_amplitude_threshold"],
        max_num_atoms=config["model"]["max_num_atoms"],
        warmup_iters=500,
        decay_factor=0.995
    )
    
    # Visualizer
    if is_main_process:
        visualizer = Visualizer(
            output_dir=str(output_dir),
            sample_rate=sample_rate,
        )
        visualizer.save_audio(gt_waveform, "target_ground_truth")
        pbar = tqdm(range(max_iters), desc="Training")
    else:
        pbar = range(max_iters)
    
    # --- 6. Training Loop ---
    for iteration in pbar:
        # Forward pass
        amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
        
        pred_waveform = renderer(
            amplitude, tau, omega, sigma, phi, gamma,
            num_samples=num_samples,
        )
        
        # Compute loss
        sigma_div_weight = density_config.get("sigma_diversity_weight", 0.001)
        loss, loss_dict = loss_fn(
            pred_waveform, gt_waveform, 
            model_amplitude=amplitude,
            model_sigma=sigma,
            sigma_diversity_weight=sigma_div_weight,
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Density gradients
        model.accumulate_gradients()
        
        # Clip & Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Scheduler Step (Crucial for convergence)
        scheduler.step()
        
        # Update adaptive thresholds
        density_controller.update_thresholds(loss.item())
        
        # --- Density Control Logic ---
        # Only perform topology changes (split/prune) within the allowed window.
        # Stopping changes late in training allows the optimizer to fine-tune.
        
        densify_from = density_config["densify_from_iter"]
        densify_until = density_config["densify_until_iter"] # Recommended: 4000
        densify_interval = density_config["densification_interval"]
        
        # We assume structure stabilizes after densify_until_iter, so we stop pruning too.
        allow_topology_changes = (densify_from <= iteration < densify_until)
        
        if allow_topology_changes:
            # Densify (Split/Clone)
            if iteration % densify_interval == 0 and iteration > 0:
                stats = density_controller.densify_and_prune(model, optimizer)
                
                if is_main_process and (stats["split"] > 0 or stats["cloned"] > 0):
                    tqdm.write(
                        f"[Density] iter={iteration}: "
                        f"split={stats['split']}, cloned={stats['cloned']}, "
                        f"total={model.num_atoms}, thresh={density_controller.grad_threshold:.2e}"
                    )
            
            # Prune
            prune_interval = density_config.get("prune_interval", 500)
            if iteration > 0 and iteration % prune_interval == 0:
                stats = density_controller.densify_and_prune(
                    model, optimizer,
                    do_split=False, do_clone=False, do_prune=True
                )
                if is_main_process and stats["pruned"] > 0:
                    tqdm.write(f"[Prune] iter={iteration}: pruned={stats['pruned']}, total={model.num_atoms}")
        
        # --- Logging & Viz ---
        if is_main_process:
            visualizer.log_loss(iteration, loss_dict["total"])
            
            if iteration % config["training"]["log_interval"] == 0:
                current_lr = scheduler.get_last_lr()[0] # Log main LR
                pbar.set_postfix({
                    "loss": f"{loss_dict['total']:.4f}",
                    "atoms": model.num_atoms,
                    "lr": f"{current_lr:.1e}"
                })
            
            # Visualization
            if iteration % config["training"]["vis_interval"] == 0 and iteration > 0:
                with torch.no_grad():
                    visualizer.generate_all_visualizations(
                        gt_waveform=gt_waveform,
                        pred_waveform=pred_waveform.detach(),
                        model=model,
                        iteration=iteration,
                    )
            
            # Checkpoint
            if iteration % config["training"]["checkpoint_interval"] == 0 and iteration > 0:
                checkpoint_path = output_dir / f"checkpoint_{iteration:06d}.pt"
                torch.save({
                    "iteration": iteration,
                    "model_state": model.state_dict_full(),
                    "config": config,
                }, str(checkpoint_path))
    
    # --- 7. Final Output ---
    if is_main_process:
        print("\n[Train] Training complete!")
        
        with torch.no_grad():
            amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
            pred_waveform = renderer(
                amplitude, tau, omega, sigma, phi, gamma,
                num_samples=num_samples,
            )
            
            outputs = visualizer.generate_all_visualizations(
                gt_waveform=gt_waveform,
                pred_waveform=pred_waveform,
                model=model,
                iteration=max_iters,
                prefix="final",
            )
            
            print(f"[Train] Final outputs saved to: {output_dir}")
            for key, path in outputs.items():
                print(f"  - {key}: {path}")
        
        final_checkpoint = output_dir / "checkpoint_final.pt"
        torch.save({
            "iteration": max_iters,
            "model_state": model.state_dict_full(),
            "config": config,
        }, str(final_checkpoint))
        print(f"[Train] Final checkpoint: {final_checkpoint}")
    
    cleanup_distributed()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config_path = args.config
    if not Path(config_path).is_absolute():
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / config_path
    
    if not Path(config_path).exists():
        print(f"[Error] Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    print("=" * 60)
    print("Audio Gaussian Splatting (AGS)")
    print("=" * 60)
    
    if args.target_file:
        print(f"Mode: Single-file optimization")
        print(f"Target: {args.target_file}")
    else:
        print(f"Mode: Dataset first-sample optimization")
        print(f"Dataset: {args.data_path}")
    
    print("=" * 60)
    
    train(args, config)


if __name__ == "__main__":
    main()