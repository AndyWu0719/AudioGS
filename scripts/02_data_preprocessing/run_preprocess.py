#!/usr/bin/env python
"""
Stage02: Preprocess dataset into codec latents for Flow training.

Outputs one `.pt` per utterance with:
  - latent: [Tz, Dz] (float16)
  - transcript, speaker_id, duration_s, audio_path
Plus:
  - speaker_to_id.json
  - latent_stats.pt (mean/std for normalization)

Example:
  python scripts/02_data_preprocessing/run_preprocess.py \
    --codec_ckpt logs/codec/checkpoints/final.pt \
    --codec_config configs/codec_config.yaml \
    --subset_dir data/raw/LibriTTS_R/train/train-clean-100 \
    --out_dir data/latents/LibriTTS_R/train-clean-100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torchaudio
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gabor_codec import GaborFrameCodec  # noqa: E402
from src.utils.gabor_frame import GaborFrameConfig  # noqa: E402


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_codec(codec_ckpt: Path, codec_cfg: dict, device: torch.device) -> Tuple[GaborFrameCodec, dict]:
    g = codec_cfg["gabor_frame"]
    gabor_cfg = GaborFrameConfig(
        sample_rate=int(codec_cfg["data"]["sample_rate"]),
        n_fft=int(g["n_fft"]),
        hop_length=int(g["hop_length"]),
        win_length=int(g.get("win_length", g["n_fft"])),
        window=str(g.get("window", "gaussian")).lower(),  # type: ignore[arg-type]
        gaussian_std_frac=float(g.get("gaussian_std_frac", 0.125)),
        center=bool(g.get("center", True)),
        pad_mode=str(g.get("pad_mode", "reflect")),
    )
    m = codec_cfg["model"]
    model = GaborFrameCodec(
        gabor_cfg=gabor_cfg,
        latent_dim=int(m["latent_dim"]),
        hidden_dim=int(m["hidden_dim"]),
        num_layers=int(m.get("num_layers", 6)),
        time_downsample=int(m.get("time_downsample", 4)),
        use_vae=bool(m.get("use_vae", False)),
        dropout=float(m.get("dropout", 0.0)),
        dilation_schedule=tuple(m.get("dilation_schedule", [])) or None,
    ).to(device)

    ckpt = torch.load(codec_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    payload = {"gabor_cfg": gabor_cfg, "codec_cfg": codec_cfg}
    return model, payload


def read_transcript(wav_path: Path) -> str:
    txt = wav_path.with_suffix(".normalized.txt")
    if txt.exists():
        try:
            return txt.read_text().strip()
        except Exception:
            return ""
    # fallback: original text
    txt2 = wav_path.with_suffix(".original.txt")
    if txt2.exists():
        try:
            return txt2.read_text().strip()
        except Exception:
            return ""
    return ""


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


def speaker_from_path(wav_path: Path) -> str:
    # LibriTTS_R layout: subset/speaker/chapter/utterance.wav
    parts = wav_path.parts
    # Find speaker as the directory just above chapter.
    # Example: .../train-clean-100/426/122821/xxx.wav -> speaker="426"
    if len(parts) >= 3:
        return parts[-3]
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Stage02: preprocess latents")
    parser.add_argument("--codec_ckpt", type=str, required=True)
    parser.add_argument("--codec_config", type=str, default="configs/codec_config.yaml")
    parser.add_argument("--subset_dir", type=str, required=True, help="Path to a LibriTTS subset folder (contains *.wav)")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_seconds", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    codec_cfg = load_yaml(PROJECT_ROOT / args.codec_config)
    sample_rate = int(codec_cfg["data"]["sample_rate"])
    codec, payload = build_codec(PROJECT_ROOT / args.codec_ckpt, codec_cfg, device=device)

    subset_dir = Path(args.subset_dir)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = list(subset_dir.rglob("*.wav"))
    wavs.sort()
    if args.limit is not None:
        wavs = wavs[: args.limit]
    if not wavs:
        raise ValueError(f"No wavs found under {subset_dir}")

    # Speaker mapping (deterministic).
    speakers = sorted({speaker_from_path(w) for w in wavs})
    speaker_to_id: Dict[str, int] = {s: i for i, s in enumerate(speakers)}
    (out_dir / "speaker_to_id.json").write_text(json.dumps(speaker_to_id, indent=2))

    # Streaming stats for normalization.
    latent_dim = int(codec_cfg["model"]["latent_dim"])
    sum_ = torch.zeros(latent_dim, dtype=torch.float64)
    sumsq = torch.zeros(latent_dim, dtype=torch.float64)
    count = 0

    max_samples = int(float(args.max_seconds) * sample_rate)

    pbar = tqdm(wavs, desc="Preprocess", total=len(wavs))
    for wav_path in pbar:
        rel = wav_path.relative_to(subset_dir)
        out_path = out_dir / rel.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        wav = load_audio(wav_path, sample_rate=sample_rate)
        if wav.numel() > max_samples:
            wav = wav[:max_samples]

        with torch.no_grad():
            z, mu, _ = codec.encode(wav.to(device), sample=False)  # deterministic
            z = mu.squeeze(0).detach().cpu()  # [Tz, Dz]

        # Update stats on mu (not sampled).
        sum_ += z.double().sum(dim=0)
        sumsq += (z.double() ** 2).sum(dim=0)
        count += int(z.shape[0])

        transcript = read_transcript(wav_path)
        speaker = speaker_from_path(wav_path)
        speaker_id = speaker_to_id.get(speaker, 0)
        duration_s = wav.numel() / sample_rate

        torch.save(
            {
                "latent": z.to(torch.float16),
                "transcript": transcript,
                "speaker": speaker,
                "speaker_id": speaker_id,
                "duration_s": float(duration_s),
                "audio_path": str(wav_path),
            },
            out_path,
        )

    if count <= 0:
        raise RuntimeError("No latents processed; stats invalid.")

    mean = (sum_ / count).float()
    var = (sumsq / count).float() - mean**2
    std = torch.sqrt(var.clamp(min=1e-8))
    torch.save({"mean": mean, "std": std, "count": count}, out_dir / "latent_stats.pt")

    # Store codec settings for downstream.
    with open(out_dir / "codec_config_used.yaml", "w") as f:
        yaml.safe_dump(payload["codec_cfg"], f)

    print(f"[Stage02] Wrote latents: {out_dir}")


if __name__ == "__main__":
    main()
