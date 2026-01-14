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
    ) -> np.ndarray:
        """
        Compute log mel-spectrogram.
        
        Args:
            waveform: Audio tensor or array
            
        Returns:
            Log mel-spectrogram [n_mels, T']
        """
        import librosa
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        
        if waveform.ndim > 1:
            waveform = waveform[0]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
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
        # Compute spectrograms
        gt_mel = self.compute_mel_spectrogram(gt_waveform)
        pred_mel = self.compute_mel_spectrogram(pred_waveform)
        
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
    
    def generate_all_visualizations(
        self,
        gt_waveform: torch.Tensor,
        pred_waveform: torch.Tensor,
        model,
        iteration: int,
        prefix: str = "",
    ) -> Dict[str, str]:
        """
        Generate all visualization outputs.
        
        Args:
            gt_waveform: Ground truth waveform
            pred_waveform: Reconstructed waveform
            model: AudioGSModel instance
            iteration: Current iteration number
            prefix: Filename prefix
            
        Returns:
            Dict mapping output type to file path
        """
        outputs = {}
        
        iter_str = f"iter_{iteration:06d}"
        if prefix:
            iter_str = f"{prefix}_{iter_str}"
        
        # Save audio
        outputs["audio_pred"] = self.save_audio(
            pred_waveform, f"{iter_str}_pred"
        )
        outputs["audio_gt"] = self.save_audio(
            gt_waveform, f"{iter_str}_gt"
        )
        
        # Spectrogram comparison
        outputs["spectrogram"] = self.plot_spectrogram_comparison(
            gt_waveform, pred_waveform,
            f"{iter_str}_spectrogram",
            title=f"Spectrogram Comparison (Iteration {iteration})"
        )
        
        # Waveform overlay
        outputs["waveform"] = self.plot_waveform_overlay(
            gt_waveform, pred_waveform,
            f"{iter_str}_waveform"
        )
        
        # Atom distribution
        outputs["atoms"] = self.plot_atom_distribution(
            model.tau,
            model.omega,
            model.amplitude,
            model.sigma,
            f"{iter_str}_atoms",
            title=f"Atom Distribution (N={model.num_atoms}, Iter={iteration})"
        )
        
        # Loss curve
        outputs["loss"] = self.plot_loss_curve(
            f"{iter_str}_loss",
            title=f"Training Loss (Iteration {iteration})"
        )
        
        return outputs


def plot_single_atom_contribution(
    renderer,
    atom_params: Dict[str, float],
    num_samples: int,
    output_path: str,
    sample_rate: int = 24000,
):
    """
    Visualize a single atom's contribution.
    
    Args:
        renderer: GaborRenderer instance
        atom_params: Dict with A, tau, omega, sigma, phi, gamma
        num_samples: Number of samples to render
        output_path: Path to save figure
        sample_rate: Sample rate
    """
    # Render single atom
    waveform = renderer.render_single_atom(
        amplitude=atom_params["A"],
        tau=atom_params["tau"],
        omega=atom_params["omega"],
        sigma=atom_params["sigma"],
        phi=atom_params["phi"],
        gamma=atom_params["gamma"],
        num_samples=num_samples,
    )
    
    waveform = waveform.cpu().numpy()
    time = np.arange(num_samples) / sample_rate * 1000  # ms
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Waveform
    axes[0].plot(time, waveform, 'c-', linewidth=0.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Gabor Atom Waveform')
    axes[0].axvline(x=atom_params["tau"] * 1000, color='r', linestyle='--', alpha=0.5, label='τ')
    axes[0].legend()
    
    # Spectrogram
    from scipy import signal
    f, t, Sxx = signal.spectrogram(waveform, sample_rate, nperseg=256)
    axes[1].pcolormesh(t * 1000, f, 10 * np.log10(Sxx + 1e-10), shading='auto', cmap='magma')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Spectrogram')
    axes[1].axhline(y=atom_params["omega"], color='c', linestyle='--', alpha=0.5, label='ω')
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
