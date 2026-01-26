# AudioGS Run Scripts

Shell launchers for the AudioGS pipeline. All scripts include conda environment activation (`qwen2_CALM`).

## Execution Order

| Script | Description |
|--------|-------------|
| `00_atom_fitting.sh` | Stage00 Gabor-frame (STFT/ISTFT) benchmark |
| `01_encoder_ddp.sh` | Stage01 codec training (DDP, multi-GPU) |
| `01_encoder_debug.sh` | Stage01 codec training (debug, single GPU) |
| `02_preprocess_latents.sh` | Stage02 preprocess dataset into latents |
| `03_encoder_eval.sh` | Stage03 codec reconstruction eval |
| `04_flow_ddp.sh` | Stage04 Flow Matching training (DDP) |
| `05_eval_tts.sh` | Stage05 Flow eval (paired sampling) |
| `06_tts_inference.sh` | Stage06 TTS inference (text → audio) |
