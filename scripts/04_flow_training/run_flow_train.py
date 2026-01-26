#!/usr/bin/env python
"""
Stage04: Train Flow Matching on codec latents (text-conditioned).

Multi-GPU (4 GPUs):
  torchrun --nproc_per_node=4 scripts/04_flow_training/run_flow_train.py --config configs/flow_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

from src.data.dataset import LibriTTSLatentDataset, collate_latents  # noqa: E402
from src.data.text_encoder import CharacterTokenizer, TextEncoder  # noqa: E402
from src.models.flow_dit import FlowDiT  # noqa: E402
from src.models.flow_matching import ConditionalFlowMatching  # noqa: E402


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


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(
    path: Path,
    flow_model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: Dict[str, Any],
    speaker_to_id: Dict[str, int],
):
    state = {
        "step": step,
        "config": config,
        "speaker_to_id": speaker_to_id,
        "flow_model": flow_model.module.state_dict() if isinstance(flow_model, DDP) else flow_model.state_dict(),
        "text_encoder": text_encoder.module.state_dict() if isinstance(text_encoder, DDP) else text_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser(description="Stage04: Flow Matching training on codec latents")
    parser.add_argument("--config", type=str, default="configs/flow_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = load_yaml(PROJECT_ROOT / args.config)
    out_dir = PROJECT_ROOT / cfg["output"]["dir"]
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)

    data_cfg = cfg["data"]
    latent_dir = PROJECT_ROOT / data_cfg["latent_dir"]
    normalizer_path = PROJECT_ROOT / data_cfg.get("normalizer_path", "")
    speaker_map_path = PROJECT_ROOT / data_cfg.get("speaker_map", "")

    speaker_to_id: Dict[str, int] = {}
    if speaker_map_path.exists():
        speaker_to_id = json.loads(speaker_map_path.read_text())

    val_ratio = float(data_cfg.get("val_ratio", 0.01))
    train_ds = LibriTTSLatentDataset(
        data_dir=str(latent_dir),
        split="train",
        val_ratio=val_ratio,
        seed=42,
        normalizer_path=str(normalizer_path) if normalizer_path.exists() else None,
        normalize_latent=True,
    )
    val_ds = LibriTTSLatentDataset(
        data_dir=str(latent_dir),
        split="val",
        val_ratio=val_ratio,
        seed=42,
        normalizer_path=str(normalizer_path) if normalizer_path.exists() else None,
        normalize_latent=True,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_latents,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_latents,
    )

    model_cfg = cfg["model"]
    text_cfg = cfg["text_encoder"]
    num_speakers = int(model_cfg.get("num_speakers", max(1, len(speaker_to_id))))

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
        num_speakers=num_speakers,
        use_cross_attention=bool(model_cfg.get("use_cross_attention", True)),
    ).to(device)

    if world_size > 1:
        text_encoder = DDP(text_encoder, device_ids=[local_rank], find_unused_parameters=False)
        flow_model = DDP(flow_model, device_ids=[local_rank], find_unused_parameters=False)

    flow_cfg = cfg.get("flow", {})
    cfm = ConditionalFlowMatching(sigma_min=float(flow_cfg.get("sigma_min", 1e-4)), use_ot=bool(flow_cfg.get("use_ot", False)))

    train_cfg = cfg["training"]
    lr = float(train_cfg.get("learning_rate", 2e-4))
    wd = float(train_cfg.get("weight_decay", 1e-2))
    params = list(flow_model.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=wd)

    max_steps = int(train_cfg.get("max_steps", 250000))
    warmup_steps = int(train_cfg.get("warmup_steps", 5000))
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], [warmup_steps])

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        (flow_model.module if isinstance(flow_model, DDP) else flow_model).load_state_dict(ckpt["flow_model"])
        (text_encoder.module if isinstance(text_encoder, DDP) else text_encoder).load_state_dict(ckpt["text_encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        step = int(ckpt.get("step", 0))

    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_interval = int(train_cfg.get("log_interval", 50))
    val_interval = int(train_cfg.get("val_interval", 2000))
    save_interval = int(train_cfg.get("save_interval", 10000))
    dur_w = float(train_cfg.get("duration_loss_weight", 0.1))

    flow_model.train()
    text_encoder.train()

    pbar = tqdm(total=max_steps, initial=step, disable=not is_main(), desc="Flow Train")

    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(step)

        for batch in train_loader:
            latents = batch["latents"].to(device)  # [B, T, Dz]
            latent_mask = batch["latent_mask"].to(device)  # [B, T]
            speaker_ids = batch["speaker_ids"].to(device)
            durations_s = batch["durations_s"].to(device)
            transcripts = batch["transcripts"]

            text_batch = tokenizer.batch_encode(transcripts, max_length=256, return_tensors=True)
            input_ids = text_batch["input_ids"].to(device)
            text_mask = text_batch["attention_mask"].to(device).bool()

            text_h, _, log_dur = text_encoder(input_ids, text_mask.long())

            flow_loss, flow_info = cfm.compute_loss(
                model=flow_model,
                x1=latents,
                mask=latent_mask,
                speaker_ids=speaker_ids,
                text_embeddings=text_h,
                text_mask=text_mask,
                latent_mask=latent_mask,
            )

            target_log_dur = torch.log(durations_s.clamp(min=1e-3)).view(-1, 1)
            dur_loss = F.mse_loss(log_dur, target_log_dur)

            loss = flow_loss + dur_w * dur_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            if is_main() and step % log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "cfm": f"{flow_info['cfm_loss']:.4f}",
                        "dur": f"{dur_loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            if is_main() and step % val_interval == 0 and step > 0:
                flow_model.eval()
                text_encoder.eval()
                with torch.no_grad():
                    v = next(iter(val_loader))
                    v_lat = v["latents"].to(device)
                    v_mask = v["latent_mask"].to(device)
                    v_spk = v["speaker_ids"].to(device)
                    v_dur = v["durations_s"].to(device)
                    v_txt = v["transcripts"]
                    tb = tokenizer.batch_encode(v_txt, max_length=256, return_tensors=True)
                    v_ids = tb["input_ids"].to(device)
                    v_tmask = tb["attention_mask"].to(device).bool()
                    v_text_h, _, v_log_dur = text_encoder(v_ids, v_tmask.long())
                    v_flow_loss, _ = cfm.compute_loss(
                        model=flow_model,
                        x1=v_lat,
                        mask=v_mask,
                        speaker_ids=v_spk,
                        text_embeddings=v_text_h,
                        text_mask=v_tmask,
                        latent_mask=v_mask,
                    )
                    v_target = torch.log(v_dur.clamp(min=1e-3)).view(-1, 1)
                    v_dur_loss = F.mse_loss(v_log_dur, v_target)
                    print(f"[Val] step={step} cfm={v_flow_loss.item():.4f} dur={v_dur_loss.item():.4f}")
                flow_model.train()
                text_encoder.train()

            if is_main() and step % save_interval == 0 and step > 0:
                save_checkpoint(
                    out_dir / "checkpoints" / f"step_{step:08d}.pt",
                    flow_model,
                    text_encoder,
                    optimizer,
                    scheduler,
                    step,
                    cfg,
                    speaker_to_id,
                )

            step += 1
            pbar.update(1)
            if step >= max_steps:
                break

    pbar.close()
    if is_main():
        save_checkpoint(out_dir / "checkpoints" / "final.pt", flow_model, text_encoder, optimizer, scheduler, step, cfg, speaker_to_id)

    cleanup_distributed()


if __name__ == "__main__":
    main()
