# AudioGS Run Scripts

Shell launchers for the AudioGS pipeline. All scripts include conda environment activation (`qwen2_CALM`).

## Execution Order

| Script | Description |
|--------|-------------|
| `01_encoder_debug.sh` | Single audio debug training for encoder |
| `03_encoder_eval.sh` | Evaluate encoder reconstruction quality |
| `04_flow_ddp.sh` | Multi-GPU Flow Matching training |
| `05_eval_tts.sh` | TTS quality evaluation (WER, Speaker Similarity) |
| `06_tts_inference.sh` | TTS inference (text to audio generation) |
