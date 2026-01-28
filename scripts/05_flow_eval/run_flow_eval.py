#!/usr/bin/env python
"""
Stage05: Evaluate Flow-on-latents by generating audio from text+speaker.

This script:
  - loads a trained Flow checkpoint + TextEncoder
  - samples latent sequences with ODE solver
  - denormalizes latents (if stats provided)
  - decodes to waveform with the Stage01 codec
  - optionally compares to ground-truth audio (paired eval)

Example (paired eval on val latents):
  python scripts/05_flow_eval/run_flow_eval.py \
    --flow_ckpt logs/flow_latent/checkpoints/final.pt \
    --flow_config configs/flow_config.yaml \
    --num_samples 10 \
    --out_dir logs/flow_eval
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import LatentNormalizer, LibriTTSLatentDataset  # noqa: E402
from src.data.text_encoder import CharacterTokenizer, TextEncoder  # noqa: E402
from src.models.flow_dit import FlowDiT  # noqa: E402
from src.models.flow_matching import FlowODESolver  # noqa: E402
from src.models.gabor_codec import GaborFrameCodec  # noqa: E402
from src.tools.metrics import compute_mss_loss, compute_pesq, compute_sisdr  # noqa: E402
from src.utils.gabor_frame import GaborFrameConfig  # noqa: E402
from src.utils.visualization import Visualizer  # noqa: E402


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_codec(ckpt_path: Path, cfg_path: Path, device: torch.device) -> Tuple[GaborFrameCodec, GaborFrameConfig]:
    cfg = load_yaml(cfg_path)
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
    codec = GaborFrameCodec(
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
    codec.load_state_dict(ckpt["model"])
    codec.eval()
    return codec, gcfg


def build_flow(flow_ckpt: Path, flow_cfg: Dict[str, Any], device: torch.device) -> Tuple[FlowDiT, TextEncoder, Dict[str, int]]:
    ckpt = torch.load(flow_ckpt, map_location=device, weights_only=False)
    speaker_to_id: Dict[str, int] = ckpt.get("speaker_to_id", {})

    text_cfg = flow_cfg["text_encoder"]
    model_cfg = flow_cfg["model"]

    tokenizer = CharacterTokenizer()
    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=int(text_cfg["embed_dim"]),
        hidden_dim=int(text_cfg["hidden_dim"]),
        num_layers=int(text_cfg["num_layers"]),
        num_heads=int(text_cfg["num_heads"]),
    ).to(device)
    flow_model = FlowDiT(
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        cond_dim=int(model_cfg.get("hidden_dim", 512)),
        text_dim=int(text_cfg["hidden_dim"]),
        num_layers=int(model_cfg.get("num_layers", 8)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_speakers=int(model_cfg.get("num_speakers", max(1, len(speaker_to_id)))),
        use_cross_attention=bool(model_cfg.get("use_cross_attention", True)),
    ).to(device)

    flow_model.load_state_dict(ckpt["flow_model"])
    text_encoder.load_state_dict(ckpt["text_encoder"])
    flow_model.eval()
    text_encoder.eval()
    return flow_model, text_encoder, speaker_to_id


def compute_audio_metrics(gt: torch.Tensor, pred: torch.Tensor, sample_rate: int) -> Dict[str, float]:
    min_len = min(gt.numel(), pred.numel())
    gt = gt[:min_len]
    pred = pred[:min_len]
    l1 = F.l1_loss(pred, gt).item()
    mse = F.mse_loss(pred, gt).item()
    snr = 10.0 * torch.log10((gt**2).mean() / ((pred - gt) ** 2).mean().clamp(min=1e-12)).item()
    mss = compute_mss_loss(gt, pred, sample_rate)
    pesq = compute_pesq(gt, pred, sample_rate)
    sisdr = compute_sisdr(gt, pred)
    return {"snr_db": snr, "sisdr": sisdr, "pesq": pesq, "mss_loss": mss, "l1_loss": l1, "mse_loss": mse}


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return wav


def main():
    parser = argparse.ArgumentParser(description="Stage05: Flow evaluation")
    parser.add_argument("--flow_ckpt", type=str, required=True)
    parser.add_argument("--flow_config", type=str, default="configs/flow_config.yaml")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="logs/flow_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    flow_cfg = load_yaml(PROJECT_ROOT / args.flow_config)

    flow_model, text_encoder, speaker_to_id = build_flow(PROJECT_ROOT / args.flow_ckpt, flow_cfg, device=device)

    codec_cfg = flow_cfg["codec"]
    codec, gcfg = build_codec(PROJECT_ROOT / codec_cfg["checkpoint"], PROJECT_ROOT / codec_cfg["config"], device=device)

    normalizer = None
    stats_path = flow_cfg["data"].get("normalizer_path", None)
    if stats_path:
        sp = PROJECT_ROOT / stats_path
        if sp.exists():
            normalizer = LatentNormalizer.load(sp)

    # Build paired eval dataset (val split).
    latent_dir = PROJECT_ROOT / flow_cfg["data"]["latent_dir"]
    val_ratio = float(flow_cfg["data"].get("val_ratio", 0.01))
    ds = LibriTTSLatentDataset(
        data_dir=str(latent_dir),
        split="val",
        val_ratio=val_ratio,
        seed=42,
        normalizer_path=str(PROJECT_ROOT / flow_cfg["data"].get("normalizer_path", "")) if stats_path else None,
        normalize_latent=True,
    )

    random.seed(args.seed)
    idxs = random.sample(range(len(ds)), k=min(args.num_samples, len(ds)))

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visualization").mkdir(exist_ok=True)
    vis = Visualizer(str(out_dir / "visualization"), sample_rate=gcfg.sample_rate, n_fft=gcfg.n_fft, hop_length=gcfg.hop_length)

    solver = FlowODESolver(flow_model, sigma_min=float(flow_cfg.get("flow", {}).get("sigma_min", 1e-4)))
    method = str(flow_cfg.get("flow", {}).get("solver_method", "rk4"))
    steps = int(flow_cfg.get("training", {}).get("num_sampling_steps", 25))

    tokenizer = CharacterTokenizer()

    rows: List[Dict[str, Any]] = []
    for j, idx in enumerate(tqdm(idxs, desc="Sample")):
        sample = ds[idx]
        transcript = sample["transcript"]
        speaker_id = int(sample["speaker_id"])
        duration_s = float(sample["duration_s"])
        audio_path = sample.get("audio_path", "")
        Tz = int(sample["latent_len"])
        Dz = int(sample["latent"].shape[-1])

        tb = tokenizer.batch_encode([transcript], max_length=256, return_tensors=True)
        input_ids = tb["input_ids"].to(device)
        text_mask = tb["attention_mask"].to(device).bool()
        with torch.no_grad():
            text_h, _, _ = text_encoder(input_ids, text_mask.long())

            latent_mask = torch.ones(1, Tz, device=device, dtype=torch.bool)
            z_norm = solver.sample(
                shape=(1, Tz, Dz),
                num_steps=steps,
                method=method,  # type: ignore[arg-type]
                device=device,
                speaker_ids=torch.tensor([speaker_id], device=device, dtype=torch.long),
                text_embeddings=text_h,
                text_mask=text_mask,
                latent_mask=latent_mask,
            )

        # Denormalize for codec decoding.
        z = z_norm
        if normalizer is not None:
            z = normalizer.denormalize(z)

        num_samples = int(math.ceil(duration_s * gcfg.sample_rate))
        with torch.no_grad():
            audio = codec.decode(z, num_samples=num_samples).squeeze(0)

        stem = f"sample_{j:03d}"
        torchaudio.save(str(out_dir / f"{stem}_gen.wav"), audio.unsqueeze(0).cpu(), gcfg.sample_rate)

        metrics: Dict[str, Any] = {"sample": stem, "latent_len": Tz, "duration_s": duration_s, "speaker_id": speaker_id}

        if audio_path:
            gt = load_audio(str(audio_path), sample_rate=gcfg.sample_rate).to(device)
            torchaudio.save(str(out_dir / f"{stem}_gt.wav"), gt.unsqueeze(0).cpu(), gcfg.sample_rate)
            metrics.update(compute_audio_metrics(gt, audio, gcfg.sample_rate))
            vis.plot_spectrogram_comparison(gt, audio, f"{stem}_vis")

        rows.append(metrics)

    out_txt = out_dir / "metrics.txt"
    with open(out_txt, "w") as f:
        f.write("Stage05 Flow Eval (paired)\n")
        f.write("=" * 60 + "\n\n")
        for m in rows:
            f.write(f"Sample: {m['sample']}\n")
            f.write(f"  Latent Len: {m['latent_len']}\n")
            f.write(f"  Duration: {m['duration_s']:.2f}s\n")
            if "snr_db" in m:
                f.write(f"  SNR: {m['snr_db']:.2f} dB\n")
                f.write(f"  SI-SDR: {m['sisdr']:.2f} dB\n")
                f.write(f"  PESQ: {m['pesq']:.2f}\n")
                f.write(f"  MSS Loss: {m['mss_loss']:.4f}\n")
            f.write("\n")

    print(f"[Stage05] Wrote: {out_txt}")


if __name__ == "__main__":
    main()
