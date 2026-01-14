"""
Visualization utilities for AGS inference and evaluation.

Matches the style of AudioGS training visualizations:
- Dark theme
- Side-by-side comparisons
- Mel-spectrogram with dB scale
- Waveform overlay for detailed comparison
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set dark theme
plt.style.use('dark_background')
matplotlib.rcParams['figure.facecolor'] = '#1e1e1e'
matplotlib.rcParams['axes.facecolor'] = '#1e1e1e'
matplotlib.rcParams['axes.edgecolor'] = '#555555'
matplotlib.rcParams['axes.labelcolor'] = '#cccccc'
matplotlib.rcParams['xtick.color'] = '#cccccc'
matplotlib.rcParams['ytick.color'] = '#cccccc'
matplotlib.rcParams['text.color'] = '#cccccc'
matplotlib.rcParams['grid.color'] = '#333333'


def compute_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> torch.Tensor:
    """Compute mel-spectrogram in dB scale."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = mel_transform(audio)
    mel_db = torchaudio.functional.amplitude_to_DB(mel, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)
    return mel_db


def plot_spectrogram_comparison(
    gt_audio: torch.Tensor,
    gen_audio: torch.Tensor,
    sample_rate: int = 24000,
    title: str = "Spectrogram Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot side-by-side mel-spectrogram comparison.
    
    Style matches AudioGS training visualizations.
    """
    # Compute mel spectrograms
    gt_mel = compute_mel_spectrogram(gt_audio, sample_rate).numpy()
    gen_mel = compute_mel_spectrogram(gen_audio, sample_rate).numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Ground Truth
    ax = axes[0]
    im1 = ax.imshow(gt_mel, aspect='auto', origin='lower', cmap='magma',
                    vmin=-80, vmax=0)
    ax.set_title('Ground Truth', fontsize=11)
    ax.set_ylabel('Mel Frequency Bins', fontsize=10)
    ax.set_xlabel('Time Frames', fontsize=10)
    cbar1 = fig.colorbar(im1, ax=ax, pad=0.02)
    cbar1.set_label('dB', fontsize=9)
    
    # Generated
    ax = axes[1]
    im2 = ax.imshow(gen_mel, aspect='auto', origin='lower', cmap='magma',
                    vmin=-80, vmax=0)
    ax.set_title('Generated', fontsize=11)
    ax.set_ylabel('Mel Frequency Bins', fontsize=10)
    ax.set_xlabel('Time Frames', fontsize=10)
    cbar2 = fig.colorbar(im2, ax=ax, pad=0.02)
    cbar2.set_label('dB', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[Visualization] Saved spectrogram: {save_path}")
    
    return fig


def plot_waveform_overlay(
    gt_audio: torch.Tensor,
    gen_audio: torch.Tensor,
    sample_rate: int = 24000,
    window_ms: int = 50,
    start_ms: Optional[int] = None,
    title: str = "Waveform Overlay",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot waveform overlay for detailed comparison.
    
    Shows a small window for phase/frequency comparison.
    """
    # Convert to numpy
    gt = gt_audio.numpy() if isinstance(gt_audio, torch.Tensor) else gt_audio
    gen = gen_audio.numpy() if isinstance(gen_audio, torch.Tensor) else gen_audio
    
    # Normalize to same scale
    gt = gt / (np.abs(gt).max() + 1e-8)
    gen = gen / (np.abs(gen).max() + 1e-8)
    
    # Select window
    window_samples = int(window_ms * sample_rate / 1000)
    if start_ms is None:
        # Find an interesting region (max energy)
        energy = np.convolve(np.abs(gt), np.ones(window_samples), 'valid')
        start_sample = np.argmax(energy)
    else:
        start_sample = int(start_ms * sample_rate / 1000)
    
    end_sample = min(start_sample + window_samples, len(gt), len(gen))
    start_sample = max(0, end_sample - window_samples)
    
    gt_window = gt[start_sample:end_sample]
    gen_window = gen[start_sample:end_sample] if len(gen) > start_sample else np.zeros_like(gt_window)
    
    time_ms = np.arange(len(gt_window)) / sample_rate * 1000 + start_sample / sample_rate * 1000
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    fig.suptitle(f"{title} ({window_ms}ms window)", fontsize=12, fontweight='bold')
    
    ax.plot(time_ms, gt_window, color='#ff6b9d', linewidth=1.2, label='Ground Truth', alpha=0.9)
    ax.plot(time_ms, gen_window, color='#00d4aa', linewidth=1.2, label='Generated', alpha=0.8)
    
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[Visualization] Saved waveform: {save_path}")
    
    return fig


def plot_atom_distribution(
    atoms: torch.Tensor,
    title: str = "Atom Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot atom parameter distributions.
    
    Args:
        atoms: [N, 6] tensor (tau, omega, sigma, amplitude, phi, gamma)
    """
    atoms = atoms.numpy() if isinstance(atoms, torch.Tensor) else atoms
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    param_names = ['tau (time)', 'omega (Hz)', 'sigma (width)', 
                   'amplitude', 'phi (phase)', 'gamma (chirp)']
    
    for i, (ax, name) in enumerate(zip(axes.flat, param_names)):
        data = atoms[:, i]
        ax.hist(data, bins=100, color='#00d4aa', alpha=0.7, edgecolor='none')
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.axvline(data.mean(), color='#ff6b9d', linestyle='--', linewidth=1.5, label=f'mean={data.mean():.3f}')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[Visualization] Saved atom distribution: {save_path}")
    
    return fig


def plot_inference_results(
    gen_audio: torch.Tensor,
    gt_audio: Optional[torch.Tensor] = None,
    gen_atoms: Optional[torch.Tensor] = None,
    sample_rate: int = 24000,
    text: Optional[str] = None,
    output_dir: str = "outputs/viz",
    prefix: str = "inference",
) -> List[str]:
    """
    Generate complete visualization for inference results.
    
    Returns list of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    
    title_suffix = f": '{text[:50]}...'" if text and len(text) > 50 else (f": '{text}'" if text else "")
    
    # 1. Spectrogram comparison (if GT available)
    if gt_audio is not None:
        spec_path = str(output_dir / f"{prefix}_spectrogram.png")
        plot_spectrogram_comparison(
            gt_audio, gen_audio, sample_rate,
            title=f"Spectrogram Comparison{title_suffix}",
            save_path=spec_path,
        )
        saved_files.append(spec_path)
        plt.close()
        
        # 2. Waveform overlay
        wave_path = str(output_dir / f"{prefix}_waveform.png")
        plot_waveform_overlay(
            gt_audio, gen_audio, sample_rate,
            title=f"Waveform Overlay{title_suffix}",
            save_path=wave_path,
        )
        saved_files.append(wave_path)
        plt.close()
    else:
        # Just plot generated spectrogram
        gen_mel = compute_mel_spectrogram(gen_audio, sample_rate).numpy()
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        im = ax.imshow(gen_mel, aspect='auto', origin='lower', cmap='magma', vmin=-80, vmax=0)
        ax.set_title(f"Generated Spectrogram{title_suffix}", fontsize=12)
        ax.set_ylabel('Mel Frequency Bins')
        ax.set_xlabel('Time Frames')
        fig.colorbar(im, ax=ax, label='dB')
        spec_path = str(output_dir / f"{prefix}_spectrogram.png")
        fig.savefig(spec_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        saved_files.append(spec_path)
        plt.close(fig)
    
    # 3. Atom distribution (if available)
    if gen_atoms is not None:
        atom_path = str(output_dir / f"{prefix}_atoms.png")
        plot_atom_distribution(gen_atoms, title="Generated Atom Distribution", save_path=atom_path)
        saved_files.append(atom_path)
        plt.close()
    
    return saved_files
