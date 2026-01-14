"""
Data Loader for Audio Gaussian Splatting.

Supports two modes:
1. Single-file mode: Load only the specified target file (for debugging)
2. Full dataset mode: Recursively scan LibriTTS_R directory

In DDP mode, uses DistributedSampler to ensure different GPUs
process different data slices.
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import random


class LibriTTSDataset(Dataset):
    """
    PyTorch Dataset for LibriTTS-R audio files.
    
    Supports both single-file mode (for debugging/overfitting)
    and full dataset mode (for training).
    """
    
    def __init__(
        self,
        root_path: str,
        target_file: Optional[str] = None,
        subsets: Optional[List[str]] = None,
        sample_rate: int = 24000,
        max_audio_length: float = 10.0,
        min_audio_length: float = 0.5,
    ):
        """
        Initialize dataset.
        
        Args:
            root_path: Root path to LibriTTS_R dataset
            target_file: If provided, dataset contains only this file (len=1)
            subsets: List of subsets to include (e.g., ["train-clean-100"])
            sample_rate: Target sample rate
            max_audio_length: Maximum audio length in seconds (truncate longer)
            min_audio_length: Minimum audio length in seconds (skip shorter)
        """
        super().__init__()
        
        self.root_path = Path(root_path)
        self.target_file = target_file
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.max_samples = int(max_audio_length * sample_rate)
        self.min_samples = int(min_audio_length * sample_rate)
        
        if subsets is None:
            subsets = ["train-clean-100", "train-clean-360"]
        self.subsets = subsets
        
        # Build file list
        self.file_list = self._build_file_list()
        
        print(f"[Dataset] Loaded {len(self.file_list)} audio files")
        
    def _build_file_list(self) -> List[Path]:
        """Build list of audio file paths."""
        
        # Single-file mode
        if self.target_file is not None:
            target_path = Path(self.target_file)
            if not target_path.exists():
                raise FileNotFoundError(f"Target file not found: {self.target_file}")
            print(f"[Dataset] Single-file mode: {self.target_file}")
            return [target_path]
        
        # Full dataset mode - recursive scan
        file_list = []
        
        for subset in self.subsets:
            subset_path = self.root_path / subset
            if not subset_path.exists():
                print(f"[Dataset] Warning: Subset path not found: {subset_path}")
                continue
            
            # Recursively find all .wav files
            wav_files = list(subset_path.rglob("*.wav"))
            file_list.extend(wav_files)
            print(f"[Dataset] Found {len(wav_files)} files in {subset}")
        
        if len(file_list) == 0:
            raise RuntimeError(f"No audio files found in {self.root_path}")
        
        # Sort for reproducibility
        file_list.sort()
        
        return file_list
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Load and preprocess audio file.
        
        Returns:
            Dict with:
                - waveform: [T] tensor
                - sample_rate: int
                - file_path: str
                - original_length: int (samples before truncation)
        """
        file_path = self.file_list[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(str(file_path))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # [T]
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        original_length = waveform.shape[0]
        
        # Truncate if too long
        if waveform.shape[0] > self.max_samples:
            # Random crop for training, or start from beginning for debug
            if len(self.file_list) > 1:
                start = random.randint(0, waveform.shape[0] - self.max_samples)
            else:
                start = 0
            waveform = waveform[start:start + self.max_samples]
        
        # Normalize to [-1, 1]
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        
        return {
            "waveform": waveform,
            "sample_rate": self.sample_rate,
            "file_path": str(file_path),
            "original_length": original_length,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function that pads audio to the longest in the batch.
    
    Args:
        batch: List of sample dicts from dataset
        
    Returns:
        Batched dict with:
            - waveforms: [B, T] tensor (padded)
            - lengths: [B] tensor of original lengths
            - sample_rates: List of sample rates
            - file_paths: List of file paths
    """
    # Find max length in batch
    max_len = max(sample["waveform"].shape[0] for sample in batch)
    
    # Pad waveforms
    waveforms = []
    lengths = []
    
    for sample in batch:
        waveform = sample["waveform"]
        length = waveform.shape[0]
        
        # Pad to max length
        if length < max_len:
            padding = torch.zeros(max_len - length)
            waveform = torch.cat([waveform, padding])
        
        waveforms.append(waveform)
        lengths.append(length)
    
    return {
        "waveforms": torch.stack(waveforms),  # [B, T]
        "lengths": torch.tensor(lengths),  # [B]
        "sample_rates": [s["sample_rate"] for s in batch],
        "file_paths": [s["file_path"] for s in batch],
    }


def get_dataloader(
    root_path: str,
    target_file: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    sample_rate: int = 24000,
    max_audio_length: float = 10.0,
    subsets: Optional[List[str]] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Create DataLoader with optional DDP support.
    
    CRITICAL: When dist.is_initialized() is True, this function
    MUST use DistributedSampler to ensure different GPUs process
    different data slices. Otherwise, all GPUs would process the
    same data, wasting compute and causing incorrect gradients.
    
    Args:
        root_path: Path to LibriTTS_R dataset
        target_file: Optional single file path (debug mode)
        batch_size: Batch size (per GPU in DDP mode)
        num_workers: Number of data loading workers
        sample_rate: Target sample rate
        max_audio_length: Max audio length in seconds
        subsets: Dataset subsets to include
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (DataLoader, sampler or None)
    """
    # Create dataset
    dataset = LibriTTSDataset(
        root_path=root_path,
        target_file=target_file,
        subsets=subsets,
        sample_rate=sample_rate,
        max_audio_length=max_audio_length,
    )
    
    # Check if running in distributed mode
    sampler = None
    if dist.is_initialized():
        # CRITICAL: Use DistributedSampler for DDP
        # This ensures each GPU processes a different subset of data
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
        print(f"[DataLoader] Using DistributedSampler: "
              f"rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        sampler=sampler,
        drop_last=True if sampler is not None else False,
    )
    
    return dataloader, sampler


class InfiniteDataLoader:
    """
    Wrapper for infinite iteration over a DataLoader.
    
    Useful for iteration-based training rather than epoch-based.
    """
    
    def __init__(self, dataloader: DataLoader, sampler: Optional[DistributedSampler] = None):
        self.dataloader = dataloader
        self.sampler = sampler
        self.iterator = None
        self.epoch = 0
        
    def __iter__(self):
        return self
    
    def __next__(self) -> Dict:
        if self.iterator is None:
            self._reset_iterator()
        
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self._reset_iterator()
            batch = next(self.iterator)
        
        return batch
    
    def _reset_iterator(self):
        """Reset iterator and update sampler epoch for proper shuffling."""
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        self.iterator = iter(self.dataloader)
