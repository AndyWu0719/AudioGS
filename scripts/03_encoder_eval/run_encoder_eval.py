#!/usr/bin/env python
"""
Stage03: Evaluate a trained codec (AE/VAE) on held-out audio.

Example:
  python scripts/03_encoder_eval/run_encoder_eval.py \
    --checkpoint logs/codec/checkpoints/final.pt \
    --codec_config configs/codec_config.yaml \
    --data_dir data/raw/LibriTTS_R/dev/dev-clean \
    --num_samples 20 \
    --out_dir logs/codec_eval
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gabor_codec import GaborFrameCodec  # noqa: E402
from src.tools.metrics import compute_mss_loss, compute_pesq, compute_sisdr  # noqa: E402
from src.utils.gabor_frame import GaborFrameConfig  # noqa: E402
from src.utils.visualization import Visualizer  # noqa: E402


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_codec(ckpt_path: Path, cfg: Dict[str, Any], device: torch.device) -> Tuple[GaborFrameCodec, GaborFrameConfig]:
    g = cfg["gabor_frame"]
    gcfg = GaborFrameConfig(
        sample_rate=int(cfg["data"]["sample_rate"]),
        n_fft=int(g["n_fft"]),
        hop_length=int(g["hop_length"]),
        win_length=int(g.get("win_length", g["n_fft"])),
        window=str(g.get("window", "gaussian")).lower(),  # type: ignore[arg-type]
        gaussian_std_frac=float(g.get("gaussian_std_frac", 0.125)),
        center=bool(g.get("center", True)),
        pad_mode=str(g.get("pad_mode", "reflect")),
    )
    m = cfg["model"]
    model = GaborFrameCodec(
        gabor_cfg=gcfg,
        latent_dim=int(m["latent_dim"]),
        hidden_dim=int(m["hidden_dim"]),
        num_layers=int(m.get("num_layers", 6)),
        time_downsample=int(m.get("time_downsample", 4)),
        use_vae=bool(m.get("use_vae", False)),
        dropout=float(m.get("dropout", 0.0)),
        dilation_schedule=tuple(m.get("dilation_schedule", [])) or None,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, gcfg


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return wav


def compute_metrics(gt: torch.Tensor, pred: torch.Tensor, sample_rate: int) -> Dict[str, float]:
    min_len = min(gt.numel(), pred.numel())
    gt = gt[:min_len]
    pred = pred[:min_len]

    l1 = F.l1_loss(pred, gt).item()
    mse = F.mse_loss(pred, gt).item()

    noise = pred - gt
    snr = 10.0 * torch.log10((gt**2).mean() / ((noise**2).mean() + 1e-12)).item()

    mss = compute_mss_loss(gt, pred, sample_rate)
    pesq_score = compute_pesq(gt, pred, sample_rate)
    sisdr = compute_sisdr(gt, pred)

    return {"snr_db": snr, "sisdr": sisdr, "pesq": pesq_score, "mss_loss": mss, "l1_loss": l1, "mse_loss": mse}


def main():
    parser = argparse.ArgumentParser(description="Stage03: codec evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--codec_config", type=str, default="configs/codec_config.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="logs/codec_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment_seconds", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(PROJECT_ROOT / args.codec_config)
    codec, gcfg = build_codec(PROJECT_ROOT / args.checkpoint, cfg, device=device)

    data_dir = PROJECT_ROOT / args.data_dir
    wavs = list(Path(data_dir).rglob("*.wav"))
    wavs.sort()
    if not wavs:
        raise ValueError(f"No wav files found under {data_dir}")

    rng = random.Random(args.seed)
    picks = rng.sample(wavs, k=min(len(wavs), args.num_samples))

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualization"
    vis_dir.mkdir(exist_ok=True)

    visualizer = Visualizer(str(vis_dir), sample_rate=gcfg.sample_rate, n_fft=gcfg.n_fft, hop_length=gcfg.hop_length)

    rows: List[Dict[str, Any]] = []
    for wav_path in tqdm(picks, desc="Eval"):
        gt = load_audio(wav_path, sample_rate=gcfg.sample_rate).to(device)
        if args.segment_seconds and args.segment_seconds > 0:
            seg_len = int(args.segment_seconds * gcfg.sample_rate)
            if gt.numel() > seg_len:
                start = rng.randint(0, gt.numel() - seg_len)
                gt = gt[start : start + seg_len]
        with torch.no_grad():
            out = codec(gt, sample_latent=False)
            pred = out["recon"].squeeze(0)

        metrics = compute_metrics(gt, pred, sample_rate=gcfg.sample_rate)
        metrics["file"] = wav_path.name
        metrics["duration_s"] = gt.numel() / gcfg.sample_rate
        metrics["latent_len"] = int(out["z"].shape[1])
        rows.append(metrics)

        stem = wav_path.stem
        torchaudio.save(str(out_dir / f"{stem}_gt.wav"), gt.unsqueeze(0).cpu(), gcfg.sample_rate)
        torchaudio.save(str(out_dir / f"{stem}_recon.wav"), pred.unsqueeze(0).cpu(), gcfg.sample_rate)
        visualizer.plot_spectrogram_comparison(gt, pred, f"{stem}_vis")

    # Write summary
    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Stage03 Codec Evaluation\n")
        f.write("=" * 60 + "\n\n")
        for m in rows:
            f.write(f"File: {m['file']}\n")
            f.write(f"  Duration: {m['duration_s']:.2f}s\n")
            f.write(f"  SNR: {m['snr_db']:.2f} dB\n")
            f.write(f"  SI-SDR: {m['sisdr']:.2f} dB\n")
            f.write(f"  PESQ: {m['pesq']:.2f}\n")
            f.write(f"  MSS Loss: {m['mss_loss']:.4f}\n")
            f.write(f"  L1 Loss: {m['l1_loss']:.6f}\n")
            f.write(f"  Latent Len: {m['latent_len']}\n")
            f.write("\n")

    print(f"[Stage03] Wrote: {metrics_path}")


if __name__ == "__main__":
    main()
