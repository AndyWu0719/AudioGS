#!/usr/bin/env python
"""
Dataset Preprocessing: Atom Dumping

Runs the trained GaborGridEncoder on the entire dataset and saves
predicted atom parameters to disk. These serve as ground truth for Stage 2 (Flow Matching).

Usage:
    torchrun --nproc_per_node=4 scripts/03_dataset_preprocessing/dump_atoms.py \
        --config configs/AudioGS_config.yaml \
        --checkpoint logs/encoder_v1/best_model.pt \
        --output_dir data/processed/atoms_v1
"""

import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import yaml
import torch
import torchaudio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
import numpy as np

from models.encoder import build_encoder


def setup_ddp():
    """Initialize Distributed Data Parallel."""
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


class InferenceDataset(Dataset):
    """Dataset that returns filename (ID) along with audio."""
    
    def __init__(self, data_path, sample_rate=24000):
        self.data_path = Path(data_path)
        self.sample_rate = sample_rate
        self.files = list(self.data_path.rglob("*.wav"))
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[Dataset] Found {len(self.files)} files")
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = torchaudio.load(str(path))
        
        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            
        # Identifier (relative path as ID)
        rel_path = str(path.relative_to(self.data_path)).replace("/", "_").replace(".wav", "")
        
        return wav.squeeze(0), rel_path


def collate_fn(batch):
    # Single sample batching is easiest for variable length inference without padding madness
    # But for speed, we can pad.
    
    wavs, ids = zip(*batch)
    lengths = torch.tensor([w.shape[-1] for w in wavs])
    max_len = lengths.max().item()
    
    padded = torch.stack([
        torch.nn.functional.pad(w, (0, max_len - w.shape[-1])) 
        for w in wavs
    ])
    
    return padded, lengths, ids


@torch.no_grad()
def dump_dataset(args, config):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # Load Encoder
    encoder = build_encoder(config).to(device)
    
    # Load Checkpoint
    print(f"[Rank {rank}] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['encoder_state']
    
    # Handle DDP prefix if present in checkpoint
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    encoder.load_state_dict(state_dict)
    encoder.eval()
    
    # Dataset
    dataset = InferenceDataset(
        data_path=config['data']['dataset_path'],
        sample_rate=config['data']['sample_rate']
    )
    
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    
    # Output Dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        pbar = tqdm(loader, total=len(loader), desc="Dumping Atoms")
    else:
        pbar = loader
        
    for batch_wavs, lengths, ids in pbar:
        batch_wavs = batch_wavs.to(device)
        
        # Inference
        enc_out = encoder(batch_wavs)
        
        # Save each sample
        # enc_out keys: 'amplitude', 'tau', 'omega', 'sigma', 'phi', 'gamma', 'existence_prob'
        # shape: [B, N_atoms]
        
        for i, file_id in enumerate(ids):
            valid_len = lengths[i]
            # Since atoms are time-aligned, valid atoms correspond to valid time.
            # However, the encoder output is flat [B, N_atoms].
            # We need to know how many atoms correspond to valid_len.
            
            # Re-calculate frames from length
            # num_frames = ceil(len / hop)
            frames = int(np.ceil(valid_len.item() / encoder.time_downsample_factor))
            valid_atoms = frames * encoder.grid_freq_bins * encoder.atoms_per_cell
            
            # Slice flat arrays
            sample_data = {}
            for k in ['amplitude', 'tau', 'omega', 'sigma', 'phi', 'gamma', 'existence_prob']:
                # Ensure we don't index out of bounds if padding was huge
                # Actually encoder output size depends on input size.
                # Since we padded input, encoder output is also "padded" (has extra atoms).
                # We slice to valid_atoms.
                sample_data[k] = enc_out[k][i, :valid_atoms].cpu()
            
            # Save
            torch.save(sample_data, output_dir / f"{file_id}.pt")
            
    cleanup_ddp()
    if rank == 0:
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    dump_dataset(args, config)
