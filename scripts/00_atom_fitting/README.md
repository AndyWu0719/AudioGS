Stage 00: Gabor Frame Reconstruction (STFT/ISTFT)
=================================================

Goal
----
Provide a clean, deterministic Stage00 pipeline that demonstrates **near-perfect audio reconstruction**
with a (windowed) Gabor dictionary, without densify/prune heuristics, MP hacks, or overlapping loss terms.

Approach
--------
Use a standard **Gabor frame** analysis/synthesis pair:
1) **STFT** (analysis): compute complex coefficients of windowed sinusoids.
2) **ISTFT** (synthesis): overlap-add reconstruction using the same window/hop.

This yields near-zero reconstruction error (floating-point noise) while still being a true
Gabor-atom representation (windowed complex exponentials).

Usage
-----
Run the benchmark (auto-picks ~1s/3s/5s samples):

`python scripts/00_atom_fitting/run_benchmark.py --config scripts/00_atom_fitting/config.yaml`

Outputs go to `logs/00_atom_fitting` (wav + visualization + metrics).
