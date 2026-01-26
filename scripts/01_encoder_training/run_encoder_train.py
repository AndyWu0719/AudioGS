#!/usr/bin/env python
"""
Stage01: Train a Gabor-frame codec (AE/VAE).

The codec learns a compact latent sequence z on top of a deterministic STFT/ISTFT:
  waveform -> STFT (Gabor atoms) -> z -> STFT_hat -> ISTFT -> waveform_hat

Multi-GPU (DDP) usage (4x 4090D):
  torchrun --nproc_per_node=4 scripts/01_encoder_training/run_encoder_train.py \
    --config configs/codec_config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_dataset import LibriTTSSegmentDataset, collate_fixed_segments, scan_wavs  # noqa: E402
from src.losses.spectral_loss import MultiResolutionSTFTLoss  # noqa: E402
from src.models.gabor_codec import CodecLossWeights, GaborFrameCodec  # noqa: E402
from src.utils.gabor_frame import GaborFrameConfig  # noqa: E402


def setup_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_gabor_cfg(cfg: Dict[str, Any]) -> GaborFrameConfig:
    g = cfg["gabor_frame"]
    return GaborFrameConfig(
        sample_rate=int(cfg["data"]["sample_rate"]),
        n_fft=int(g["n_fft"]),
        hop_length=int(g["hop_length"]),
        win_length=int(g.get("win_length", g["n_fft"])),
        window=str(g.get("window", "gaussian")).lower(),  # type: ignore[arg-type]
        gaussian_std_frac=float(g.get("gaussian_std_frac", 0.125)),
        center=bool(g.get("center", True)),
        pad_mode=str(g.get("pad_mode", "reflect")),
    )


def build_loss_weights(cfg: Dict[str, Any]) -> CodecLossWeights:
    w = cfg.get("loss", {}).get("weights", {})
    return CodecLossWeights(
        time_l1=float(w.get("time_l1", 1.0)),
        stft_mss=float(w.get("stft_mss", 0.5)),
        kl=float(w.get("kl", 1e-4)),
        latent_l1=float(w.get("latent_l1", 0.0)),
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: Dict[str, Any],
):
    state = {
        "step": step,
        "config": config,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser(description="Stage01: Train Gabor-frame codec (AE/VAE)")
    parser.add_argument("--config", type=str, default="configs/codec_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg_path = PROJECT_ROOT / args.config
    cfg = load_config(str(cfg_path))

    out_dir = PROJECT_ROOT / cfg["output"]["dir"]
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)
        (out_dir / "samples").mkdir(exist_ok=True)

    # Dataset split (by file list, deterministic).
    data_cfg = cfg["data"]
    train_root = str(data_cfg["dataset_root"])
    train_subsets = list(data_cfg.get("subsets", ["train-clean-100"]))
    train_files = scan_wavs(train_root, subsets=train_subsets)

    # Prefer a separate validation dataset if provided.
    val_root = str(data_cfg.get("val_dataset_root", "") or "")
    val_subsets = list(data_cfg.get("val_subsets", []))
    val_files = []
    if val_root and val_subsets:
        try:
            val_files = scan_wavs(val_root, subsets=val_subsets)
        except Exception:
            val_files = []

    if not val_files:
        # Fallback: split from training files.
        val_size = max(1, int(len(train_files) * float(args.val_ratio)))
        val_files = train_files[:val_size]
        train_files = train_files[val_size:]
        if is_main():
            print(f"[Stage01] Validation: train split val_ratio={float(args.val_ratio):.4f} (val={len(val_files)})")
    else:
        if is_main():
            print(f"[Stage01] Validation: external set root={val_root} subsets={val_subsets} (val={len(val_files)})")

    train_ds = LibriTTSSegmentDataset(
        root_path=train_root,
        subsets=train_subsets,
        sample_rate=int(data_cfg["sample_rate"]),
        segment_seconds=float(data_cfg.get("segment_seconds", 3.0)),
        min_seconds=float(data_cfg.get("min_seconds", 1.0)),
        files=train_files,
    )
    val_ds = LibriTTSSegmentDataset(
        root_path=(val_root if val_files and val_root else train_root),
        subsets=(val_subsets if val_files and val_root else train_subsets),
        sample_rate=int(data_cfg["sample_rate"]),
        segment_seconds=float(data_cfg.get("segment_seconds", 3.0)),
        min_seconds=float(data_cfg.get("min_seconds", 1.0)),
        files=val_files,
        seed=123,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fixed_segments,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fixed_segments,
    )

    gabor_cfg = build_gabor_cfg(cfg)
    model_cfg = cfg["model"]
    codec = GaborFrameCodec(
        gabor_cfg=gabor_cfg,
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg.get("num_layers", 6)),
        time_downsample=int(model_cfg.get("time_downsample", 4)),
        use_vae=bool(model_cfg.get("use_vae", False)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    if world_size > 1:
        codec = DDP(codec, device_ids=[local_rank], find_unused_parameters=False)

    loss_weights = build_loss_weights(cfg)
    mss_cfg = cfg.get("loss", {}).get("mss", {})
    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=list(mss_cfg.get("fft_sizes", [2048, 1024, 512])),
        hop_sizes=[s // 4 for s in list(mss_cfg.get("fft_sizes", [2048, 1024, 512]))],
        win_lengths=list(mss_cfg.get("fft_sizes", [2048, 1024, 512])),
        spectral_weight=1.0,
        log_mag_weight=1.0,
        time_domain_weight=0.0,
    ).to(device)

    train_cfg = cfg["training"]
    lr = float(train_cfg.get("learning_rate", 2e-4))
    wd = float(train_cfg.get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(codec.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=wd)

    max_steps = int(train_cfg.get("max_steps", 200000))
    warmup_steps = int(train_cfg.get("warmup_steps", 2000))
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        (codec.module if isinstance(codec, DDP) else codec).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        step = int(ckpt.get("step", 0))

    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_interval = int(train_cfg.get("log_interval", 20))
    val_interval = int(train_cfg.get("val_interval", 2000))
    save_interval = int(train_cfg.get("save_interval", 10000))
    accum_steps = int(train_cfg.get("accumulation_steps", 1))

    codec.train()
    pbar = tqdm(total=max_steps, initial=step, disable=not is_main(), desc="Codec Train")

    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(step)

        for batch in train_loader:
            wave = batch["waveforms"].to(device)  # [B, T]

            out = codec(wave, sample_latent=bool(model_cfg.get("use_vae", False)))
            recon = out["recon"]
            z = out["z"]
            mu = out["mu"]
            logvar = out["logvar"]

            # Losses
            loss_time = F.l1_loss(recon, wave)
            loss_stft, _ = stft_loss(recon, wave)
            loss = loss_weights.time_l1 * loss_time + loss_weights.stft_mss * loss_stft

            if loss_weights.latent_l1 > 0:
                loss = loss + loss_weights.latent_l1 * z.abs().mean()

            if (codec.module.use_vae if isinstance(codec, DDP) else codec.use_vae) and logvar is not None:
                loss_kl = GaborFrameCodec.kl_loss(mu, logvar)
                loss = loss + loss_weights.kl * loss_kl

            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if is_main() and step % log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item()*accum_steps:.4f}",
                        "l_time": f"{loss_time.item():.4f}",
                        "l_stft": f"{loss_stft.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            if is_main() and step % val_interval == 0 and step > 0:
                codec.eval()
                with torch.no_grad():
                    v = next(iter(val_loader))
                    v_wave = v["waveforms"].to(device)
                    v_out = codec(v_wave, sample_latent=False)
                    v_recon = v_out["recon"]
                    v_l1 = F.l1_loss(v_recon, v_wave).item()
                    (out_dir / "samples").mkdir(exist_ok=True)
                    import torchaudio

                    torchaudio.save(
                        str(out_dir / "samples" / f"step_{step:08d}_gt.wav"),
                        v_wave.cpu(),
                        gabor_cfg.sample_rate,
                    )
                    torchaudio.save(
                        str(out_dir / "samples" / f"step_{step:08d}_recon.wav"),
                        v_recon.cpu(),
                        gabor_cfg.sample_rate,
                    )
                    print(f"[Val] step={step} L1={v_l1:.6f}")
                codec.train()

            if is_main() and step % save_interval == 0 and step > 0:
                save_checkpoint(
                    out_dir / "checkpoints" / f"step_{step:08d}.pt",
                    codec,
                    optimizer,
                    scheduler,
                    step,
                    cfg,
                )

            step += 1
            pbar.update(1)
            if step >= max_steps:
                break

    pbar.close()
    if is_main():
        save_checkpoint(out_dir / "checkpoints" / "final.pt", codec, optimizer, scheduler, step, cfg)
        # Store the final gabor config for downstream preprocessing/inference.
        with open(out_dir / "gabor_frame_config.yaml", "w") as f:
            yaml.safe_dump(asdict(gabor_cfg), f)

    cleanup_distributed()


if __name__ == "__main__":
    main()
