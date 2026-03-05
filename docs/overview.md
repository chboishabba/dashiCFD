# dashiCFD Overview

## Quickstart

CPU-only (headless) sanity run:

```bash
MPLBACKEND=Agg python run_v4_snapshots.py --N 64 --steps 3000 --stride 300 --out-dir outputs
```

Kernel-only perf (needs kernel artifacts):

```bash
MPLBACKEND=Agg python perf_kernel.py \
  --z0-npz outputs/kernel_N128_z0.npz \
  --A-npz outputs/kernel_N128_A.npz \
  --steps 20000 \
  --decode-every 200
```

GPU LES-only (vkFFT + Vulkan):

```bash
MPLBACKEND=Agg python run_les_gpu.py \
  --N 512 \
  --steps 20000 \
  --dt 0.01 \
  --nu0 1e-4 \
  --Cs 0.17 \
  --stats-every 200 \
  --progress-every 2000 \
  --spectral-truncation exp \
  --trunc-alpha 36 \
  --trunc-power 8 \
  --out-dir outputs \
  --prefix les_gpu
```

If you need a single interactive entry point, use:

```bash
python dashi_cli.py les --interactive
python dashi_cli.py kernel --interactive
python dashi_cli.py plot --interactive
python dashi_cli.py compare --interactive
```

## Runner Map

Primary runners:
- `run_v4_snapshots.py` — canonical end-to-end DASHI CFD pipeline (LES → encode → learn → rollout → decode + plots/metrics).
- `perf_kernel.py` — kernel-only proxy rollout + decode; supports metrics-only mode and backend benchmarking.
- `run_les_gpu.py` — GPU LES only (vkFFT + Vulkan); saves enstrophy CSV and optional PNGs.
- `dashi_cli.py` — interactive wrapper around the runners above plus plotting/compare helpers.

Targeted utilities:
- `make_kernel_artifacts.py` — export kernel-only artifacts (`A` and `z0`) from a v4 run.
- `scripts/compare_les_gpu_cpu.py` — GPU vs CPU LES sanity check.
- `scripts/plot_enstrophy.py` — plot enstrophy from JSON/CSV logs.
- `scripts/validate_gpu_truth.py` — validate GPU LES against stored truth.
- `scripts/run_sweep.py` / `scripts/perf_sampler.py` — parameter sweeps and perf sampling.

## Backend Notes

Backends used across the repo:
- `cpu` — pure NumPy, always available.
- `accelerated` — dashiCORE accelerated CPU backend (if available).
- `vulkan` — Vulkan backend for dashiCORE ops; requires Python Vulkan bindings and compiled shaders.
- `vkfft-*` — FFT backends (NumPy or vkFFT over Vulkan/OpenCL).

Common tips:
- Use `MPLBACKEND=Agg` for headless runs to avoid GUI errors.
- Vulkan decode and vkFFT backends may fall back to CPU unless `--require-gpu` is set (or `--permissive-backends` is disabled).
- For explicit ICD selection, set `VK_ICD_FILENAMES` before launching.
- Large grids may overflow in float32; use `--dtype float64` or reduce `dt`/`Cs`.

## Outputs and Artifacts

Conventions:
- Use `outputs/` for generated PNGs, CSVs, and JSON metrics.
- Use timestamped prefixes to avoid overwriting prior runs.
- Keep large artifacts in compressed `.npz` when possible.
