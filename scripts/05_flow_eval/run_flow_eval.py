"""
TTS Evaluation Script.

Evaluate TTS quality using WER, FAD, and Speaker Similarity.

Usage:
    python scripts/04_inference_eval/run_eval_tts.py \
        --checkpoint logs/flow/best.pt \
        --data_dir data/atoms/LibriTTS_R/train/train-clean-100 \
        --num_samples 50
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

import torch
import torchaudio
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

from tools.inference import AGSInferenceEngine
from tools.metrics import compute_wer, compute_speaker_similarity_batch


def parse_args():
    parser = argparse.ArgumentParser(description="TTS Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config path")
    parser.add_argument("--data_dir", type=str, required=True, help="Validation atom data dir")
    parser.add_argument("--original_dir", type=str, default=None, help="Original audio dir")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--whisper_model", type=str, default="base")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TTS Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Samples: {args.num_samples}")
    print("=" * 60)
    
    # Initialize engine
    engine = AGSInferenceEngine(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    
    # Collect samples
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else PROJECT_ROOT / args.data_dir
    atom_files = list(data_dir.rglob("*.pt"))[:args.num_samples]
    
    print(f"Found {len(atom_files)} samples")
    
    # Output directory
    output_dir = PROJECT_ROOT / "outputs" / f"eval_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated").mkdir(exist_ok=True)
    
    # Generate and collect
    reference_texts = []
    generated_paths = []
    reference_audio_paths = []
    
    for i, atom_file in enumerate(tqdm(atom_files, desc="Generating")):
        # Load atom data
        data = torch.load(atom_file, weights_only=False)
        transcript = data.get('transcript', '')
        source_path = data.get('source_path', '')
        speaker_id = data.get('speaker_id', '0')
        duration = data.get('audio_duration', 5.0)
        
        if not transcript:
            continue
        
        # Map speaker to ID
        if isinstance(speaker_id, str):
            speaker_id = hash(speaker_id) % 100
        
        # Generate
        try:
            audio = engine.generate(
                text=transcript,
                speaker_id=int(speaker_id),
                steps=25,
                method='rk4',
            )
            
            # Save generated audio
            gen_path = output_dir / "generated" / f"sample_{i:04d}.wav"
            engine.save_audio(audio, str(gen_path))
            
            reference_texts.append(transcript)
            generated_paths.append(str(gen_path))
            
            if source_path and Path(source_path).exists():
                reference_audio_paths.append(source_path)
            
        except Exception as e:
            print(f"[Error] {atom_file}: {e}")
            continue
    
    print(f"\nGenerated {len(generated_paths)} samples")
    
    # Compute metrics
    results = {}
    
    # WER
    print("\nComputing WER...")
    try:
        wer_results = compute_wer(
            reference_texts, 
            generated_paths,
            whisper_model=args.whisper_model,
            device=args.device,
        )
        results["wer"] = wer_results
        print(f"  WER: {wer_results['wer']*100:.2f}%")
    except Exception as e:
        print(f"  [Error] WER: {e}")
        results["wer"] = {"error": str(e)}
    
    # Speaker Similarity
    if reference_audio_paths:
        print("Computing Speaker Similarity...")
        try:
            sim_results = compute_speaker_similarity_batch(
                generated_paths[:len(reference_audio_paths)],
                reference_audio_paths,
            )
            results["speaker_similarity"] = sim_results
            print(f"  SIM: {sim_results['mean']:.3f} ± {sim_results['std']:.3f}")
        except Exception as e:
            print(f"  [Error] SIM: {e}")
            results["speaker_similarity"] = {"error": str(e)}
    
    # Save results
    results["config"] = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "num_samples": len(generated_paths),
        "whisper_model": args.whisper_model,
    }
    
    output_path = args.output or str(output_dir / "results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if "wer" in results and "wer" in results["wer"]:
        print(f"WER: {results['wer']['wer']*100:.2f}%")
    if "speaker_similarity" in results and "mean" in results["speaker_similarity"]:
        print(f"Speaker Similarity: {results['speaker_similarity']['mean']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
