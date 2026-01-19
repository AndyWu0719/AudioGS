"""
TTS Evaluation Metrics.

Provides functions for evaluating TTS quality:
- WER: Word Error Rate (ASR-based)
- FAD: Fréchet Audio Distance
- SIM: Speaker Similarity (cosine distance)

Usage:
    from src.tools.metrics import compute_wer, compute_fad, compute_speaker_sim
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio

# Optional dependencies
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False


# ============================================================
# Word Error Rate (WER)
# ============================================================

def compute_wer(
    reference_texts: List[str],
    audio_paths: List[str],
    whisper_model: str = "base",
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute Word Error Rate using Whisper ASR.
    
    Args:
        reference_texts: Ground truth transcripts
        audio_paths: Paths to generated audio files
        whisper_model: Whisper model size ("tiny", "base", "small", "medium", "large")
        device: Device to run ASR on
        
    Returns:
        dict with 'wer', 'insertions', 'deletions', 'substitutions'
    """
    if not JIWER_AVAILABLE:
        raise ImportError("jiwer not installed. Run: pip install jiwer")
    if not WHISPER_AVAILABLE:
        raise ImportError("whisper not installed. Run: pip install openai-whisper")
    
    # Load Whisper model
    model = whisper.load_model(whisper_model, device=device)
    
    hypotheses = []
    for audio_path in audio_paths:
        result = model.transcribe(audio_path)
        hypotheses.append(result["text"].strip().lower())
    
    # Normalize references
    references = [t.strip().lower() for t in reference_texts]
    
    # Compute WER
    measures = jiwer.compute_measures(references, hypotheses)
    
    return {
        "wer": measures["wer"],
        "mer": measures["mer"],  # Match Error Rate
        "wil": measures["wil"],  # Word Information Lost
        "insertions": measures["insertions"],
        "deletions": measures["deletions"],
        "substitutions": measures["substitutions"],
    }


def compute_wer_single(
    reference: str,
    audio_path: str,
    whisper_model: str = "base",
    device: str = "cuda",
) -> float:
    """Compute WER for a single sample."""
    result = compute_wer([reference], [audio_path], whisper_model, device)
    return result["wer"]


# ============================================================
# Speaker Similarity
# ============================================================

def compute_speaker_similarity(
    audio1_path: str,
    audio2_path: str,
    sample_rate: int = 16000,
    device: str = "cuda",
) -> float:
    """
    Compute speaker similarity using cosine distance of embeddings.
    
    Uses a simple MFCC-based embedding (can be replaced with speaker encoder).
    
    Args:
        audio1_path: Path to first audio
        audio2_path: Path to second audio
        sample_rate: Target sample rate
        device: Computation device
        
    Returns:
        similarity: Cosine similarity score [0, 1]
    """
    def load_and_embed(path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        
        # Compute MFCC embedding
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
        )
        mfcc = mfcc_transform(waveform)  # [1, n_mfcc, T]
        
        # Mean pooling over time
        embedding = mfcc.mean(dim=-1).squeeze(0)  # [n_mfcc]
        return embedding
    
    emb1 = load_and_embed(audio1_path)
    emb2 = load_and_embed(audio2_path)
    
    # Cosine similarity
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
    
    # Convert to [0, 1] range (cosine sim is [-1, 1])
    similarity = (similarity + 1) / 2
    
    return similarity


def compute_speaker_similarity_batch(
    generated_paths: List[str],
    reference_paths: List[str],
    sample_rate: int = 16000,
) -> Dict[str, float]:
    """
    Compute speaker similarity for a batch of samples.
    
    Returns:
        dict with 'mean', 'std', 'min', 'max'
    """
    similarities = []
    for gen_path, ref_path in zip(generated_paths, reference_paths):
        sim = compute_speaker_similarity(gen_path, ref_path, sample_rate)
        similarities.append(sim)
    
    return {
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "samples": len(similarities),
    }


# ============================================================
# Audio Quality Metrics
# ============================================================

def compute_mss_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    sample_rate: int = 24000,
    fft_sizes: List[int] = [512, 1024, 2048],
) -> float:
    """
    Compute Multi-Scale Spectral Loss.
    
    Args:
        original: Original waveform [T]
        reconstructed: Reconstructed waveform [T]
        sample_rate: Sample rate (unused, for API consistency)
        fft_sizes: List of FFT sizes for multi-scale
        
    Returns:
        mss_loss: Average spectral distance
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    total_loss = 0
    for fft_size in fft_sizes:
        hop_size = fft_size // 4
        window = torch.hann_window(fft_size, device=original.device)
        
        # Compute STFT
        spec_orig = torch.stft(original, fft_size, hop_size, window=window, return_complex=True)
        spec_recon = torch.stft(reconstructed, fft_size, hop_size, window=window, return_complex=True)
        
        # Magnitude
        mag_orig = spec_orig.abs()
        mag_recon = spec_recon.abs()
        
        # L1 + log loss
        l1_loss = (mag_orig - mag_recon).abs().mean()
        log_loss = (torch.log(mag_orig + 1e-7) - torch.log(mag_recon + 1e-7)).abs().mean()
        
        total_loss += l1_loss + log_loss
    
    return (total_loss / len(fft_sizes)).item()


def compute_f0_error(
    original_path: str,
    modified_path: str,
    expected_shift: float = 0,
    sample_rate: int = 24000,
) -> Dict[str, float]:
    """
    Compute F0 tracking error for pitch editing evaluation.
    
    Args:
        original_path: Path to original audio
        modified_path: Path to pitch-shifted audio
        expected_shift: Expected shift in semitones
        sample_rate: Sample rate
        
    Returns:
        dict with 'mean_error', 'std_error' in semitones
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa not installed. Run: pip install librosa")
    
    # Load audio
    y_orig, _ = librosa.load(original_path, sr=sample_rate)
    y_mod, _ = librosa.load(modified_path, sr=sample_rate)
    
    # Extract F0
    f0_orig, _, _ = librosa.pyin(y_orig, fmin=50, fmax=500, sr=sample_rate)
    f0_mod, _, _ = librosa.pyin(y_mod, fmin=50, fmax=500, sr=sample_rate)
    
    # Remove NaN
    valid = ~(np.isnan(f0_orig) | np.isnan(f0_mod))
    f0_orig = f0_orig[valid]
    f0_mod = f0_mod[valid]
    
    if len(f0_orig) == 0:
        return {"mean_error": float('nan'), "std_error": float('nan')}
    
    # Convert to semitones
    ratio = f0_mod / (f0_orig + 1e-8)
    measured_shift = 12 * np.log2(ratio + 1e-8)
    
    # Error from expected
    error = measured_shift - expected_shift
    
    return {
        "mean_error": float(np.mean(np.abs(error))),
        "std_error": float(np.std(error)),
        "measured_shift_mean": float(np.mean(measured_shift)),
    }


def compute_duration_error(
    original_path: str,
    modified_path: str,
    expected_rate: float = 1.0,
) -> Dict[str, float]:
    """
    Compute duration error for time-stretching evaluation.
    
    Args:
        original_path: Path to original audio
        modified_path: Path to time-stretched audio
        expected_rate: Expected speed rate (>1 = faster)
        
    Returns:
        dict with 'duration_original', 'duration_modified', 'error'
    """
    # Get durations
    wf_orig, sr_orig = torchaudio.load(original_path)
    wf_mod, sr_mod = torchaudio.load(modified_path)
    
    dur_orig = wf_orig.shape[1] / sr_orig
    dur_mod = wf_mod.shape[1] / sr_mod
    
    # Expected duration
    expected_dur = dur_orig / expected_rate
    error = abs(dur_mod - expected_dur)
    error_percent = error / expected_dur * 100
    
    return {
        "duration_original": dur_orig,
        "duration_modified": dur_mod,
        "expected_duration": expected_dur,
        "error_seconds": error,
        "error_percent": error_percent,
    }


# ============================================================
# Aggregate Evaluation
# ============================================================

# ============================================================
# Advanced Audio Metrics (PESQ, SI-SDR)
# ============================================================

def compute_pesq(
    ref_wav: torch.Tensor,
    deg_wav: torch.Tensor,
    sample_rate: int = 16000,
    mode: str = "wb",
) -> float:
    """
    Compute PESQ score.
    
    Args:
        ref_wav: Reference waveform [T]
        deg_wav: Degraded waveform [T]
        sample_rate: Sample rate (must be 16000 or 8000)
        mode: 'wb' (wideband) or 'nb' (narrowband)
        
    Returns:
        PESQ score [-0.5, 4.5]
    """

    if not PESQ_AVAILABLE:
        return float('nan')
    
    # Resample if needed (PESQ requires 16k or 8k)
    target_sr = 16000 if mode == 'wb' else 8000
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr).to(ref_wav.device)
        ref_wav = resampler(ref_wav.view(1, -1)).squeeze()
        deg_wav = resampler(deg_wav.view(1, -1)).squeeze()
        sample_rate = target_sr

    # Determine max length to avoid errors
    min_len = min(ref_wav.shape[-1], deg_wav.shape[-1])
    ref_wav = ref_wav[..., :min_len]
    deg_wav = deg_wav[..., :min_len]

    # Normalize to prevent PESQ error
    ref_wav = ref_wav / (torch.max(torch.abs(ref_wav)) + 1e-8)
    deg_wav = deg_wav / (torch.max(torch.abs(deg_wav)) + 1e-8)

    try:
        score = pesq(
            sample_rate, 
            ref_wav.detach().cpu().numpy(), 
            deg_wav.detach().cpu().numpy(), 
            mode
        )
        return score
    except Exception as e:
        print(f"PESQ Error: {e}")
        return float('nan')


def compute_sisdr(
    ref_wav: torch.Tensor,
    est_wav: torch.Tensor,
) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        ref_wav: Reference waveform [T]
        est_wav: Estimated waveform [T]
        
    Returns:
        SI-SDR score in dB
    """

    eps = 1e-8
    
    # Ensure 1D
    if ref_wav.dim() > 1: ref_wav = ref_wav.squeeze()
    if est_wav.dim() > 1: est_wav = est_wav.squeeze()
        
    # Match length
    min_len = min(len(ref_wav), len(est_wav))
    ref = ref_wav[:min_len]
    est = est_wav[:min_len]
    
    # Zero mean
    ref = ref - torch.mean(ref)
    est = est - torch.mean(est)
    
    # Optimal scaling
    # alpha = <ref, est> / <ref, ref>
    ref_energy = torch.sum(ref ** 2) + eps
    alpha = torch.sum(ref * est) / ref_energy
    
    target = alpha * ref
    noise = est - target
    
    # SI-SDR
    target_pow = torch.sum(target ** 2) + eps
    noise_pow = torch.sum(noise ** 2) + eps
    
    sisdr = 10 * torch.log10(target_pow / noise_pow)
    return sisdr.item()


# ============================================================
# Aggregate Evaluation
# ============================================================

def evaluate_tts_batch(
    generated_paths: List[str],
    reference_texts: List[str],
    reference_audio_paths: Optional[List[str]] = None,
    whisper_model: str = "base",
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Run full TTS evaluation on a batch of samples.
    
    Args:
        generated_paths: Paths to generated audio files
        reference_texts: Ground truth transcripts
        reference_audio_paths: Paths to reference audio (for speaker sim)
        whisper_model: Whisper model size
        device: Computation device
        
    Returns:
        dict with all metrics
    """
    results = {}
    
    # WER
    try:
        wer_results = compute_wer(reference_texts, generated_paths, whisper_model, device)
        results["wer"] = wer_results
    except Exception as e:
        results["wer"] = {"error": str(e)}
    
    # Speaker similarity
    if reference_audio_paths:
        try:
            sim_results = compute_speaker_similarity_batch(
                generated_paths, reference_audio_paths
            )
            results["speaker_similarity"] = sim_results
        except Exception as e:
            results["speaker_similarity"] = {"error": str(e)}
    
    return results
