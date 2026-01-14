"""
TTS Inference CLI.

Generate audio from text using trained Flow Matching model.

Usage:
    python scripts/04_inference_eval/run_inference.py \
        --checkpoint logs/flow/best.pt \
        --text "Hello, this is a test." \
        --output output.wav
"""

import sys
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.inference import AGSInferenceEngine


def parse_args():
    parser = argparse.ArgumentParser(description="AGS TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config (optional)")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker ID")
    parser.add_argument("--output", type=str, default="output.wav", help="Output path")
    parser.add_argument("--steps", type=int, default=25, help="ODE solver steps")
    parser.add_argument("--method", type=str, default="rk4", choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("AGS TTS Inference")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Text: {args.text}")
    print(f"Speaker ID: {args.speaker_id}")
    print(f"Steps: {args.steps} ({args.method})")
    print("=" * 60)
    
    # Initialize engine
    engine = AGSInferenceEngine(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    
    # Generate
    audio = engine.generate(
        text=args.text,
        speaker_id=args.speaker_id,
        steps=args.steps,
        method=args.method,
        temperature=args.temperature,
    )
    
    # Save
    engine.save_audio(audio, args.output)
    
    print(f"\nGenerated audio saved to: {args.output}")
    print(f"Duration: {len(audio) / engine.sample_rate:.2f}s")


if __name__ == "__main__":
    main()
