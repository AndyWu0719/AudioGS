#!/usr/bin/env python
"""
Stage06: Text-to-speech inference (Flow -> latents -> codec decode -> waveform).

Example:
  python scripts/06_tts_inference/run_tts_inference.py \
    --flow_ckpt logs/flow_latent/checkpoints/final.pt \
    --flow_config configs/flow_config.yaml \
    --text "hello world" \
    --speaker_id 0 \
    --out_wav logs/tts_out.wav
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import LatentNormalizer  # noqa: E402
from src.data.text_encoder import CharacterTokenizer, TextEncoder  # noqa: E402
from src.models.flow_dit import FlowDiT  # noqa: E402
from src.models.flow_matching import FlowODESolver  # noqa: E402
from src.models.gabor_codec import GaborFrameCodec  # noqa: E402
from src.utils.gabor_frame import GaborFrameConfig, num_frames  # noqa: E402


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
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    codec.load_state_dict(ckpt["model"])
    codec.eval()
    return codec, gcfg


def build_flow(flow_ckpt: Path, flow_cfg: Dict[str, Any], device: torch.device) -> Tuple[FlowDiT, TextEncoder]:
    ckpt = torch.load(flow_ckpt, map_location=device, weights_only=False)
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
        num_speakers=int(model_cfg.get("num_speakers", 500)),
        use_cross_attention=bool(model_cfg.get("use_cross_attention", True)),
    ).to(device)

    flow_model.load_state_dict(ckpt["flow_model"])
    text_encoder.load_state_dict(ckpt["text_encoder"])
    flow_model.eval()
    text_encoder.eval()
    return flow_model, text_encoder


def main():
    parser = argparse.ArgumentParser(description="Stage06: TTS inference")
    parser.add_argument("--flow_ckpt", type=str, required=True)
    parser.add_argument("--flow_config", type=str, default="configs/flow_config.yaml")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--duration_s", type=float, default=None, help="Override predicted duration (seconds)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--method", type=str, default=None, choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--out_wav", type=str, default="logs/tts_out.wav")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    flow_cfg = load_yaml(PROJECT_ROOT / args.flow_config)

    flow_model, text_encoder = build_flow(PROJECT_ROOT / args.flow_ckpt, flow_cfg, device=device)

    codec_cfg = flow_cfg["codec"]
    codec, gcfg = build_codec(PROJECT_ROOT / codec_cfg["checkpoint"], PROJECT_ROOT / codec_cfg["config"], device=device)

    normalizer: Optional[LatentNormalizer] = None
    stats_path = flow_cfg["data"].get("normalizer_path", None)
    if stats_path:
        sp = PROJECT_ROOT / stats_path
        if sp.exists():
            normalizer = LatentNormalizer.load(sp)

    tokenizer = CharacterTokenizer()
    tb = tokenizer.batch_encode([args.text], max_length=256, return_tensors=True)
    input_ids = tb["input_ids"].to(device)
    text_mask = tb["attention_mask"].to(device).bool()

    with torch.no_grad():
        text_h, _, log_dur = text_encoder(input_ids, text_mask.long())
        pred_dur = float(torch.exp(log_dur).item())

    duration_s = float(args.duration_s) if args.duration_s is not None else pred_dur
    duration_s = max(0.2, min(duration_s, 20.0))
    num_samples = int(math.ceil(duration_s * gcfg.sample_rate))

    # Latent length derived from desired waveform length.
    frames = num_frames(num_samples, gcfg)
    latent_len = int(math.ceil(frames / codec.time_downsample))
    latent_dim = codec.latent_dim

    solver = FlowODESolver(flow_model, sigma_min=float(flow_cfg.get("flow", {}).get("sigma_min", 1e-4)))
    steps = int(args.steps) if args.steps is not None else int(flow_cfg.get("training", {}).get("num_sampling_steps", 25))
    method = args.method or str(flow_cfg.get("flow", {}).get("solver_method", "rk4"))

    latent_mask = torch.ones(1, latent_len, device=device, dtype=torch.bool)
    with torch.no_grad():
        z_norm = solver.sample(
            shape=(1, latent_len, latent_dim),
            num_steps=steps,
            method=method,  # type: ignore[arg-type]
            device=device,
            speaker_ids=torch.tensor([args.speaker_id], device=device, dtype=torch.long),
            text_embeddings=text_h,
            text_mask=text_mask,
            latent_mask=latent_mask,
        )
        z = normalizer.denormalize(z_norm) if normalizer is not None else z_norm
        audio = codec.decode(z, num_samples=num_samples).squeeze(0)

    out_path = PROJECT_ROOT / args.out_wav
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), audio.unsqueeze(0).cpu(), gcfg.sample_rate)
    print(f"[Stage06] Wrote: {out_path} (duration={duration_s:.2f}s, latent_len={latent_len})")


if __name__ == "__main__":
    main()
