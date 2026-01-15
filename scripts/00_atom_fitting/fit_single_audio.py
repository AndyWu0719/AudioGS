"""
CUDA Extension Training Script for Audio Gaussian Splatting.

Uses custom C++/CUDA kernels for maximum performance.
Expected speedup: 2-3x compared to standard PyTorch version.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict

import torch
import torch.distributed as dist
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cuda_gabor"))

from models.atom import AudioGSModel
from cuda_gabor import get_cuda_gabor_renderer
from losses.spectral_loss import CombinedAudioLoss
from utils.data_loader import get_dataloader
from utils.density_control import AdaptiveDensityController, rebuild_optimizer_from_model
from utils.visualization import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="AudioGS Training (CUDA Extension)")
    parser.add_argument("--target_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, 
                        default="/data0/determined/users/andywu/workplace/data/raw/LibriTTS_R")
    parser.add_argument("--config", type=str, default="configs/AudioGS_config.yaml")
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path) as f:
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
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Output dir
    if args.exp_name is None:
        exp_name = f"cuda_ext_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        exp_name = args.exp_name
    
    output_dir = Path(args.output_dir) / exp_name
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CUDA-Ext] Output: {output_dir}")
        print(f"[CUDA-Ext] Device: {device}")
    
    # Config
    sample_rate = config["data"]["sample_rate"]
    max_iters = args.max_iters or config["training"]["max_iters"]
    
    # Data
    dataloader, _ = get_dataloader(
        root_path=args.data_path,
        target_file=args.target_file,
        batch_size=1,
        sample_rate=sample_rate,
        max_audio_length=config["data"]["max_audio_length"],
        subsets=config["data"].get("subsets"),
        shuffle=False,
    )
    
    fixed_batch = next(iter(dataloader))
    gt_waveform = fixed_batch["waveforms"][0].to(device)
    num_samples = int(fixed_batch["lengths"][0].item())
    audio_duration = num_samples / sample_rate
    
    if is_main:
        print(f"[CUDA-Ext] Target: {fixed_batch['file_paths'][0]}")
        print(f"[CUDA-Ext] Duration: {audio_duration:.2f}s")
    
    # Model
    model = AudioGSModel(
        num_atoms=config["model"]["initial_num_atoms"],
        sample_rate=sample_rate,
        audio_duration=audio_duration,
        device=device,
    )
    model.initialize_from_audio(gt_waveform)
    
    # CUDA Extension Renderer
    renderer = get_cuda_gabor_renderer(sample_rate=sample_rate)
    
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
    optimizer = rebuild_optimizer_from_model(model, torch.optim.Adam, config["training"])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[4000, 7000, 9000], gamma=0.5
    )
    
    # Density Controller
    dc = config["density_control"]
    density_controller = AdaptiveDensityController(
        grad_threshold=dc["grad_threshold"],
        sigma_split_threshold=dc["sigma_split_threshold"],
        prune_amplitude_threshold=dc["prune_amplitude_threshold"],
        max_num_atoms=config["model"]["max_num_atoms"],
    )
    
    # Visualizer
    if is_main:
        visualizer = Visualizer(str(output_dir), sample_rate)
        visualizer.save_audio(gt_waveform, "target_gt")
        pbar = tqdm(range(max_iters), desc="Training (CUDA-Ext)")
    else:
        pbar = range(max_iters)
    
    # Training loop
    for iteration in pbar:
        amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
        
        # CUDA extension forward
        pred_waveform = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
        
        # Loss
        loss, loss_dict = loss_fn(
            pred_waveform, gt_waveform,
            model_amplitude=amplitude,
            model_sigma=sigma,
            sigma_diversity_weight=dc.get("sigma_diversity_weight", 0.001),
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
        if dc["densify_from_iter"] <= iteration < dc["densify_until_iter"]:
            if iteration % dc["densification_interval"] == 0 and iteration > 0:
                stats = density_controller.densify_and_prune(model, optimizer)
                if is_main and (stats["split"] > 0 or stats["cloned"] > 0):
                    tqdm.write(f"[Density] iter={iteration}: split={stats['split']}, cloned={stats['cloned']}, total={model.num_atoms}")
        
        # Logging
        if is_main:
            visualizer.log_loss(iteration, loss_dict["total"])
            if iteration % config["training"]["log_interval"] == 0:
                pbar.set_postfix({"loss": f"{loss_dict['total']:.4f}", "atoms": model.num_atoms})
            
            if iteration % config["training"]["vis_interval"] == 0 and iteration > 0:
                with torch.no_grad():
                    visualizer.generate_all_visualizations(gt_waveform, pred_waveform.detach(), model, iteration)
    
    # Final
    if is_main:
        print("\n[CUDA-Ext] Complete!")
        with torch.no_grad():
            amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
            pred = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
            visualizer.generate_all_visualizations(gt_waveform, pred, model, max_iters, prefix="final")
        
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
    
    config = load_config(str(config_path))
    
    print("=" * 60)
    print("Audio Gaussian Splatting (C++/CUDA EXTENSION)")
    print("=" * 60)
    if args.target_file:
        print(f"Target: {args.target_file}")
    print("=" * 60)
    
    train(args, config)


if __name__ == "__main__":
    main()