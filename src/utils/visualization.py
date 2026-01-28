"""
Visualization Tools for Audio Gaussian Splatting.

Generates rich outputs for analysis and debugging:
1. Reconstructed audio (.wav files)
2. Spectrogram comparisons (GT vs Reconstructed)
3. Waveform overlays (zoomed sections)
4. Atom distribution maps (time-frequency scatter)
5. Loss curves
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import soundfile as sf


class Visualizer:
    """
    Visualization and output generation for AudioGS.
    """
    
    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save outputs
            sample_rate: Audio sample rate
            n_fft: FFT size for spectrograms
            hop_length: Hop size for spectrograms
            n_mels: Number of mel bands
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Style settings
        plt.style.use('dark_background')
        self.cmap = 'magma'
        
        # Loss history for plotting
        self.loss_history: List[float] = []
        self.iteration_history: List[int] = []
        
    def save_audio(
        self,
        waveform: torch.Tensor,
        filename: str,
        normalize: bool = True,
    ) -> str:
        """
        Save waveform as .wav file.
        
        Args:
            waveform: Audio tensor [T] or [B, T]
            filename: Output filename (without extension)
            normalize: Whether to normalize to [-1, 1]
            
        Returns:
            Path to saved file
        """
        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        
        # Handle batched input
        if waveform.ndim > 1:
            waveform = waveform[0]

        # Sanitize non-finite values (can happen if upstream diverges).
        if not np.isfinite(waveform).all():
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize
        if normalize:
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val * 0.95
        
        # Save
        output_path = self.output_dir / f"{filename}.wav"
        sf.write(str(output_path), waveform, self.sample_rate)
        
        return str(output_path)
    
    def compute_mel_spectrogram(
        self,
        waveform: Union[torch.Tensor, np.ndarray],
        return_db: bool = True,
        ref_power: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute log mel-spectrogram.
        
        Args:
            waveform: Audio tensor or array
            return_db: If True, return dB-scaled mel; otherwise return power mel
            ref_power: Reference power for dB conversion (shared ref across plots)
            
        Returns:
            Mel-spectrogram [n_mels, T'] (dB if return_db else power)
        """
        import librosa
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        
        if waveform.ndim > 1:
            waveform = waveform[0]

        # librosa.util.valid_audio rejects NaN/inf. Prefer a best-effort plot.
        if not np.isfinite(waveform).all():
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        
        if return_db:
            ref = ref_power if ref_power is not None else np.max
            mel_spec = librosa.power_to_db(mel_spec, ref=ref)
        
        return mel_spec
    
    def plot_spectrogram_comparison(
        self,
        gt_waveform: torch.Tensor,
        pred_waveform: torch.Tensor,
        filename: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Plot side-by-side spectrogram comparison.
        
        Args:
            gt_waveform: Ground truth waveform
            pred_waveform: Reconstructed waveform
            filename: Output filename (without extension)
            title: Optional plot title
            
        Returns:
            Path to saved figure
        """
        import librosa

        # Compute spectrograms
        gt_mel_power = self.compute_mel_spectrogram(gt_waveform, return_db=False)
        pred_mel_power = self.compute_mel_spectrogram(pred_waveform, return_db=False)
        ref_power = float(np.max(gt_mel_power))
        if (not np.isfinite(ref_power)) or ref_power <= 0:
            ref_power = 1e-8
        gt_mel = librosa.power_to_db(gt_mel_power, ref=ref_power)
        pred_mel = librosa.power_to_db(pred_mel_power, ref=ref_power)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Ground truth
        im1 = axes[0].imshow(
            gt_mel, aspect='auto', origin='lower', cmap=self.cmap
        )
        axes[0].set_title('Ground Truth', fontsize=12)
        axes[0].set_xlabel('Time Frames')
        axes[0].set_ylabel('Mel Frequency Bins')
        plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')
        
        # Reconstructed
        im2 = axes[1].imshow(
            pred_mel, aspect='auto', origin='lower', cmap=self.cmap
        )
        axes[1].set_title('Reconstructed', fontsize=12)
        axes[1].set_xlabel('Time Frames')
        axes[1].set_ylabel('Mel Frequency Bins')
        plt.colorbar(im2, ax=axes[1], format='%+2.0f dB')
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"{filename}.png"
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(output_path)
    
    def plot_waveform_overlay(
        self,
        gt_waveform: torch.Tensor,
        pred_waveform: torch.Tensor,
        filename: str,
        zoom_ms: float = 50.0,
        start_ms: Optional[float] = None,
    ) -> str:
        """
        Plot zoomed waveform overlay for phase inspection.
        
        Args:
            gt_waveform: Ground truth waveform
            pred_waveform: Reconstructed waveform
            filename: Output filename (without extension)
            zoom_ms: Window size in milliseconds
            start_ms: Start time in milliseconds (None = auto)
            
        Returns:
            Path to saved figure
        """
        # Convert to numpy
        if isinstance(gt_waveform, torch.Tensor):
            gt = gt_waveform.detach().cpu().numpy()
        else:
            gt = gt_waveform
        if isinstance(pred_waveform, torch.Tensor):
            pred = pred_waveform.detach().cpu().numpy()
        else:
            pred = pred_waveform
        
        if gt.ndim > 1:
            gt = gt[0]
        if pred.ndim > 1:
            pred = pred[0]
        
        # Calculate zoom window
        zoom_samples = int(zoom_ms * self.sample_rate / 1000)
        
        if start_ms is None:
            # Find an interesting region (high energy)
            energy = np.convolve(gt ** 2, np.ones(zoom_samples), mode='valid')
            start_sample = max(0, np.argmax(energy))
        else:
            start_sample = int(start_ms * self.sample_rate / 1000)
        
        end_sample = min(start_sample + zoom_samples, len(gt), len(pred))
        
        # Extract windows
        gt_window = gt[start_sample:end_sample]
        pred_window = pred[start_sample:end_sample]
        
        # Time axis in milliseconds
        time_ms = np.arange(len(gt_window)) / self.sample_rate * 1000
        time_ms += start_sample / self.sample_rate * 1000
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.plot(time_ms, gt_window, 'c-', alpha=0.8, linewidth=1.0, label='Ground Truth')
        ax.plot(time_ms, pred_window, 'm-', alpha=0.8, linewidth=1.0, label='Reconstructed')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveform Overlay ({zoom_ms:.0f}ms window)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"{filename}.png"
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(output_path)
    
    def plot_atom_distribution(
        self,
        tau: torch.Tensor,
        omega: torch.Tensor,
        amplitude: torch.Tensor,
        sigma: torch.Tensor,
        filename: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Plot atom distribution in time-frequency space.
        
        X-axis: Time (τ)
        Y-axis: Frequency (ω)
        Point size: Proportional to amplitude
        Point color: Sigma (envelope width)
        
        Args:
            tau: Time centers [N]
            omega: Frequencies [N]
            amplitude: Amplitudes [N]
            sigma: Envelope widths [N]
            filename: Output filename
            title: Optional title
            
        Returns:
            Path to saved figure
        """
        # Convert to numpy
        if isinstance(tau, torch.Tensor):
            tau = tau.detach().cpu().numpy()
        if isinstance(omega, torch.Tensor):
            omega = omega.detach().cpu().numpy()
        if isinstance(amplitude, torch.Tensor):
            amplitude = amplitude.detach().cpu().numpy()
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize sizes for visibility
        size_scale = amplitude / (amplitude.max() + 1e-8) * 100 + 5
        
        # Color by sigma (log scale for better visibility)
        log_sigma = np.log10(sigma + 1e-8)
        
        scatter = ax.scatter(
            tau * 1000,  # Convert to ms
            omega,
            s=size_scale,
            c=log_sigma,
            cmap='viridis',
            alpha=0.6,
            edgecolors='white',
            linewidths=0.3,
        )
        
        ax.set_xlabel('Time τ (ms)')
        ax.set_ylabel('Frequency ω (Hz)')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Atom Distribution (N={len(tau)})')
        
        # Colorbar for sigma
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(σ) [seconds]')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"{filename}.png"
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(output_path)
    
    def log_loss(self, iteration: int, loss: float):
        """Record loss for plotting."""
        self.iteration_history.append(iteration)
        self.loss_history.append(loss)
    
    def plot_loss_curve(
        self,
        filename: str = "loss_curve",
        title: Optional[str] = None,
    ) -> str:
        """
        Plot training loss curve.
        
        Args:
            filename: Output filename
            title: Optional title
            
        Returns:
            Path to saved figure
        """
        if len(self.loss_history) == 0:
            print("[Visualizer] Warning: No loss history to plot")
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(self.iteration_history, self.loss_history, 'c-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title or 'Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add smoothed curve
        if len(self.loss_history) > 20:
            window = min(50, len(self.loss_history) // 5)
            smoothed = np.convolve(
                self.loss_history,
                np.ones(window) / window,
                mode='valid'
            )
            # Adjust x-axis for smoothed curve
            offset = window // 2
            smooth_iters = self.iteration_history[offset:offset + len(smoothed)]
            ax.plot(smooth_iters, smoothed, 'm-', linewidth=2, alpha=0.8, label='Smoothed')
            ax.legend()
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"{filename}.png"
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(output_path)
