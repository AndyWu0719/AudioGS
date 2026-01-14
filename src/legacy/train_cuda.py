"""
CUDA-Accelerated Training Script for Audio Gaussian Splatting.

Uses the Triton-based CUDA renderer for significantly faster training.
All other functionality is identical to train.py.

Usage:
    python src/train_cuda.py --target_file path/to/audio.wav --max_iters 10000
"""

import os
import sys
import argparse
import yaml
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

from models.atom import AudioGSModel
from models.renderer_cuda import get_cuda_renderer, GaborRendererCUDA
from losses.spectral_loss import CombinedAudioLoss
from utils.data_loader import get_dataloader
from utils.density_control import AdaptiveDensityController, rebuild_optimizer_from_model
from utils.visualization import Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AudioGS Training (CUDA Accelerated)")
    
    parser.add_argument("--target_file", type=str, default=None, help="Target audio file")
    parser.add_argument("--data_path", type=str, 
                        default="/data0/determined/users/andywu/workplace/data/raw/LibriTTS_R")
    parser.add_argument("--config", type=str, default="configs/AudioGS_config.yaml")
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args, config: Dict):
    """Main training function with CUDA acceleration."""
    
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Experiment directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"cuda_{timestamp}"
    else:
        exp_name = args.exp_name
    
    output_dir = Path(args.output_dir) / exp_name
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Train-CUDA] Output: {output_dir}")
        print(f"[Train-CUDA] Device: {device}")
    
    # Config
    sample_rate = config["data"]["sample_rate"]
    max_iters = args.max_iters or config["training"]["max_iters"]
    
    # Data
    dataloader, sampler = get_dataloader(
        root_path=args.data_path,
        target_file=args.target_file,
        batch_size=1,
        sample_rate=sample_rate,
        max_audio_length=config["data"]["max_audio_length"],
        subsets=config["data"].get("subsets"),
        shuffle=False,
    )
    
    try:
        fixed_batch = next(iter(dataloader))
    except StopIteration:
        print("[Error] Dataset empty!")
        return
    
    gt_waveform = fixed_batch["waveforms"][0].to(device)
    num_samples = int(fixed_batch["lengths"][0].item())
    audio_duration = num_samples / sample_rate
    
    if is_main_process:
        print(f"[Train-CUDA] Target: {fixed_batch['file_paths'][0]}")
        print(f"[Train-CUDA] Duration: {audio_duration:.2f}s")
    
    # Model
    model = AudioGSModel(
        num_atoms=config["model"]["initial_num_atoms"],
        sample_rate=sample_rate,
        audio_duration=audio_duration,
        device=device,
    )
    model.initialize_from_audio(gt_waveform)
    
    # CUDA Renderer - KEY DIFFERENCE
    renderer = get_cuda_renderer(sample_rate=sample_rate)
    if is_main_process:
        print(f"[Train-CUDA] Renderer: {type(renderer).__name__}")
    
    # Loss
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
        pre_emp_weight=loss_config.get("pre_emp_weight", 20.0),
    )
    
    # Optimizer
    optimizer = rebuild_optimizer_from_model(
        model, torch.optim.Adam, config["training"]
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[4000, 7000, 9000], gamma=0.5
    )
    
    # Density Controller
    density_config = config["density_control"]
    density_controller = AdaptiveDensityController(
        grad_threshold=density_config["grad_threshold"],
        sigma_split_threshold=density_config["sigma_split_threshold"],
        prune_amplitude_threshold=density_config["prune_amplitude_threshold"],
        max_num_atoms=config["model"]["max_num_atoms"],
    )
    
    # Visualizer
    if is_main_process:
        visualizer = Visualizer(str(output_dir), sample_rate)
        visualizer.save_audio(gt_waveform, "target_gt")
        pbar = tqdm(range(max_iters), desc="Training (CUDA)")
    else:
        pbar = range(max_iters)
    
    # Training loop
    for iteration in pbar:
        amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
        
        # CUDA-accelerated rendering
        pred_waveform = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
        
        # Loss
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
        model.accumulate_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Density control
        density_controller.update_thresholds(loss.item())
        densify_from = density_config["densify_from_iter"]
        densify_until = density_config["densify_until_iter"]
        densify_interval = density_config["densification_interval"]
        
        if densify_from <= iteration < densify_until:
            if iteration % densify_interval == 0 and iteration > 0:
                density_controller.densify_and_prune(model, optimizer)
        
        # Logging
        if is_main_process:
            visualizer.log_loss(iteration, loss_dict["total"])
            
            if iteration % config["training"]["log_interval"] == 0:
                pbar.set_postfix({
                    "loss": f"{loss_dict['total']:.4f}",
                    "atoms": model.num_atoms,
                })
            
            if iteration % config["training"]["vis_interval"] == 0 and iteration > 0:
                with torch.no_grad():
                    visualizer.generate_all_visualizations(
                        gt_waveform, pred_waveform.detach(), model, iteration
                    )
    
    # Final output
    if is_main_process:
        print("\n[Train-CUDA] Complete!")
        with torch.no_grad():
            amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
            pred_waveform = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
            visualizer.generate_all_visualizations(
                gt_waveform, pred_waveform, model, max_iters, prefix="final"
            )
        
        torch.save({
            "iteration": max_iters,
            "model_state": model.state_dict_full(),
            "config": config,
        }, output_dir / "checkpoint_final.pt")
    
    cleanup_distributed()


def main():
    args = parse_args()
    
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    
    if not Path(config_path).exists():
        print(f"[Error] Config not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    print("=" * 60)
    print("Audio Gaussian Splatting (CUDA Accelerated)")
    print("=" * 60)
    
    if args.target_file:
        print(f"Target: {args.target_file}")
    
    print("=" * 60)
    
    train(args, config)


if __name__ == "__main__":
    main()
