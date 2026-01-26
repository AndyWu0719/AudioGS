"""
Stage 00 entrypoint: Gabor frame (STFT/ISTFT) benchmark.

Runs on ~1s/3s/5s samples (auto-selected) and writes:
  - recon wavs
  - GT vs recon mel visualizations
  - metrics.txt
to logs/00_atom_fitting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (  # noqa: E402
    find_test_files_by_duration,
    load_yaml,
    run_single,
    write_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Stage00: Gabor frame (STFT/ISTFT) benchmark")
    parser.add_argument("--config", type=str, default="scripts/00_atom_fitting/config.yaml")
    parser.add_argument("--audio_file", type=str, default=None, help="Run a single file instead of benchmark")
    parser.add_argument("--durations", type=float, nargs="*", default=[1.0, 3.0, 5.0])
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    config: Dict[str, Any] = load_yaml(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage00] Device: {device}")

    metrics: List[Dict[str, Any]] = []

    if args.audio_file is not None:
        audio_path = args.audio_file
        metrics.append(run_single(audio_path, config, device))
    else:
        dataset_path = config["data"]["dataset_path"]
        sample_rate = int(config["data"]["sample_rate"])
        files: List[Optional[str]] = find_test_files_by_duration(
            dataset_path, target_durations=args.durations, sample_rate=sample_rate
        )
        files = [f for f in files if f is not None]
        if not files:
            raise ValueError("No test files found; check dataset_path.")
        for fpath in files:
            metrics.append(run_single(fpath, config, device))

    out_root = Path(config["output"]["root_dir"])
    write_metrics(metrics, out_root)
    print(f"[Stage00] Wrote metrics to: {out_root / 'metrics.txt'}")


if __name__ == "__main__":
    main()
