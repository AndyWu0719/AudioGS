"""
Stage 00: Gabor Frame Reconstruction (STFT/ISTFT)
======================================================

Goal:
  Demonstrate that a windowed Gabor dictionary can reconstruct audio near-perfectly
  with a clean, deterministic analysis/synthesis pipeline.

Approach:
  - Compute complex STFT coefficients (analysis)
  - Reconstruct via ISTFT overlap-add (synthesis)

This is a true Gabor-atom representation (windowed complex exponentials) and provides
an unambiguous Stage00 "existence proof" baseline with minimal code and no heuristics.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import yaml

from src.tools.metrics import compute_mss_loss, compute_pesq, compute_sisdr
from src.utils.visualization import Visualizer
from src.utils.gabor_frame import GaborFrameConfig, istft as g_istft, stft as g_stft


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_audio(path: str, sample_rate: int) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0), sample_rate


def find_test_files_by_duration(
    root_path: str,
    target_durations: List[float],
    sample_rate: int,
    scan_limit: int = 500,
) -> List[Optional[str]]:
    root = Path(root_path)
    all_files = list(root.rglob("*.wav"))
    if not all_files:
        raise ValueError(f"No WAV files found in {root_path}")

    file_durations: List[Tuple[Path, float]] = []
    for fpath in all_files[:scan_limit]:
        try:
            info = torchaudio.info(str(fpath))
            duration = info.num_frames / info.sample_rate
            file_durations.append((fpath, duration))
        except Exception:
            continue

    results: List[Optional[str]] = []
    for target in target_durations:
        best: Optional[Tuple[Path, float]] = None
        best_diff = float("inf")
        for fpath, dur in file_durations:
            diff = abs(dur - target)
            if diff < best_diff:
                best = (fpath, dur)
                best_diff = diff
        if best is None:
            results.append(None)
        else:
            results.append(str(best[0]))
    return results


def compute_metrics(gt: torch.Tensor, pred: torch.Tensor, sample_rate: int) -> Dict[str, float]:
    min_len = min(gt.numel(), pred.numel())
    gt = gt[:min_len]
    pred = pred[:min_len]

    l1 = F.l1_loss(pred, gt).item()
    mse = F.mse_loss(pred, gt).item()

    noise = pred - gt
    signal_power = (gt ** 2).mean()
    noise_power = (noise ** 2).mean()
    snr = 10.0 * torch.log10(signal_power / (noise_power + 1e-12)).item()

    mss = compute_mss_loss(gt, pred, sample_rate)
    pesq_score = compute_pesq(gt, pred, sample_rate)
    sisdr = compute_sisdr(gt, pred)

    # Mel-L1 (torchaudio, power mel)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        power=2.0,
    ).to(gt.device)
    mel_gt = mel_transform(gt.unsqueeze(0))
    mel_pred = mel_transform(pred.unsqueeze(0))
    mel_l1 = F.l1_loss(torch.log(mel_pred + 1e-8), torch.log(mel_gt + 1e-8)).item()

    return {
        "snr_db": snr,
        "sisdr": sisdr,
        "pesq": pesq_score,
        "mel_l1": mel_l1,
        "mss_loss": mss,
        "l1_loss": l1,
        "mse_loss": mse,
    }

def _sparsify_stft(
    stft: torch.Tensor,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return stft, {"enabled": False, "kept": int(stft.numel()), "total": int(stft.numel())}

    mode = str(cfg.get("mode", "topk_per_frame")).lower()
    mag = stft.abs()
    total = int(mag.numel())

    if mode == "topk":
        k = int(cfg.get("topk", 0))
        k = max(0, min(k, total))
        if k == 0:
            return torch.zeros_like(stft), {"enabled": True, "mode": mode, "kept": 0, "total": total}
        flat = mag.reshape(-1)
        thresh = flat.topk(k).values.min()
        mask = mag >= thresh
        return stft * mask, {"enabled": True, "mode": mode, "kept": int(mask.sum().item()), "total": total}

    if mode == "topk_per_frame":
        k = int(cfg.get("topk", 0))
        k = max(0, min(k, mag.shape[-2]))  # freq bins
        if k == 0:
            return torch.zeros_like(stft), {"enabled": True, "mode": mode, "kept": 0, "total": total}
        _, idx = mag.topk(k, dim=-2)
        mask = torch.zeros_like(mag, dtype=torch.bool)
        mask.scatter_(-2, idx, True)
        return stft * mask, {"enabled": True, "mode": mode, "kept": int(mask.sum().item()), "total": total}

    if mode == "threshold_db":
        threshold_db = float(cfg.get("threshold_db", -80.0))
        max_mag = mag.max()
        if max_mag.item() <= 0:
            return torch.zeros_like(stft), {"enabled": True, "mode": mode, "kept": 0, "total": total}
        threshold = max_mag * (10.0 ** (threshold_db / 20.0))
        mask = mag >= threshold
        return stft * mask, {"enabled": True, "mode": mode, "kept": int(mask.sum().item()), "total": total}

    raise ValueError(f"Unsupported sparsify mode: {mode}")


def gabor_frame_reconstruct(
    gt: torch.Tensor,
    sample_rate: int,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    cfg = config["gabor_frame"]
    gcfg = GaborFrameConfig(
        sample_rate=sample_rate,
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        win_length=int(cfg.get("win_length", int(cfg["n_fft"]))),
        window=str(cfg.get("window", "gaussian")).lower(),  # type: ignore[arg-type]
        gaussian_std_frac=float(cfg.get("gaussian_std_frac", 0.125)),
        center=bool(cfg.get("center", True)),
        pad_mode=str(cfg.get("pad_mode", "reflect")),
        periodic=bool(cfg.get("periodic", True)),
    )

    stft = g_stft(gt, gcfg)[0]  # [F, TT]
    stft_s, sparsify_info = _sparsify_stft(stft, cfg.get("sparsify", {}))
    pred = g_istft(stft_s, gcfg, length=gt.numel()).squeeze(0)

    atoms_info = {
        "method": "gabor_frame",
        "sample_rate": sample_rate,
        "n_fft": gcfg.n_fft,
        "hop_length": gcfg.hop_length,
        "win_length": gcfg.win_length,
        "window": str(gcfg.window),
        "gaussian_std_frac": float(gcfg.gaussian_std_frac),
        "center": bool(gcfg.center),
        "pad_mode": str(gcfg.pad_mode),
        "sparsify": sparsify_info,
        "stft_shape": tuple(stft_s.shape),
        "atom_count": int(sparsify_info.get("kept", stft.numel())),
    }
    return pred, {"atoms_info": atoms_info, "stft": stft_s}


def run_single(
    audio_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    sample_rate = int(config["data"]["sample_rate"])
    gt, _ = load_audio(audio_path, sample_rate)
    gt = gt.to(device)

    filename = Path(audio_path).stem
    output_root = Path(config["output"]["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    vis_dir = output_root / "visualization"
    vis_dir.mkdir(exist_ok=True)
    atoms_dir = output_root / "atoms"
    if bool(config["output"].get("save_atoms", True)):
        atoms_dir.mkdir(exist_ok=True)

    method = str(config.get("method", "gabor_frame")).lower()
    if method != "gabor_frame":
        raise ValueError(f"Unsupported method: {method} (supported: gabor_frame)")

    with torch.no_grad():
        pred, atoms_payload = gabor_frame_reconstruct(gt, sample_rate, config, device)

    # Save audio
    recon_path = output_root / f"{filename}_recon.wav"
    torchaudio.save(str(recon_path), pred.unsqueeze(0).cpu(), sample_rate)

    # Visualization
    visualizer = Visualizer(str(vis_dir), sample_rate)
    visualizer.plot_spectrogram_comparison(gt, pred, f"{filename}_vis")

    # Save atoms
    if bool(config["output"].get("save_atoms", True)):
        torch.save(
            {
                **atoms_payload["atoms_info"],
                "stft": atoms_payload["stft"].detach().cpu(),
            },
            atoms_dir / f"{filename}_atoms_stft.pt",
        )

    metrics = compute_metrics(gt, pred, sample_rate)
    metrics["file"] = Path(audio_path).name
    metrics["duration_s"] = gt.numel() / sample_rate
    metrics["method"] = method
    metrics["final_atoms"] = int(atoms_payload["atoms_info"]["atom_count"])
    metrics["atoms_info"] = atoms_payload["atoms_info"]

    return metrics


def write_metrics(metrics: List[Dict[str, Any]], output_root: Path):
    out = output_root / "metrics.txt"
    with open(out, "w") as f:
        f.write("Stage00 (Gabor Frame STFT/ISTFT) Results\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        for m in metrics:
            f.write(f"File: {m['file']}\n")
            f.write(f"  Duration: {m['duration_s']:.2f}s\n")
            f.write(f"  Method: {m.get('method', 'gabor_frame')}\n")
            f.write(f"  SNR: {m['snr_db']:.2f} dB\n")
            f.write(f"  Mel-L1: {m['mel_l1']:.4f}\n")
            f.write(f"  PESQ: {m['pesq']:.2f}\n")
            f.write(f"  SI-SDR: {m['sisdr']:.2f} dB\n")
            f.write(f"  MSS Loss: {m['mss_loss']:.4f}\n")
            f.write(f"  L1 Loss: {m['l1_loss']:.6f}\n")
            f.write(f"  MSE Loss: {m['mse_loss']:.6e}\n")
            f.write(f"  Final Atoms: {m['final_atoms']}\n")
            f.write("\n")
