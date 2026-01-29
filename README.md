# dashiCFD

CFD + DASHI experiments for 2D vorticity rollouts, ternary structural codecs, and SUSY gauge scans. Assets in the repo are mostly matplotlib outputs; simulations run in pure NumPy.

## How to Run
- Use Python 3.10+ with `numpy` and `matplotlib`; set `MPLBACKEND=Agg` for headless environments.
- Typical commands:
  - `MPLBACKEND=Agg python dashi_cfd_operator_v3.py`
  - `MPLBACKEND=Agg python dashi_cfd_operator_v4.py`
  - `MPLBACKEND=Agg python dashi_les_vorticity_codec_v2.py`
  - `MPLBACKEND=Agg python vortex_tester_mdl.py`
  - `MPLBACKEND=Agg CORE_BACKEND=cpu python CORE_cfd_operator.py`

## Vulkan GPU commands

Compile SPIR-V (preferred path `dashiCORE/spv/comp` -> `dashiCORE/spv`):

```bash
python dashiCORE/scripts/compile_spv.py
```

Kernel-only perf (GPU rollout + Vulkan decode, metrics-only readback):

```bash
python perf_kernel.py \
  --z0-npz outputs/kernel_N128_z0.npz \
  --A-npz outputs/kernel_N128_A.npz \
  --steps 20000 \
  --decode-every 200 \
  --decode-backend vulkan \
  --observer metrics \
  --backend vulkan \
  --fft-backend vkfft-vulkan \
  --op-backend vulkan \
  --require-gpu \
  --metrics-json outputs/perf_metrics_gpu.json
```

Kernel-only perf (full ω̂ readback for energy/enstrophy):

```bash
python perf_kernel.py \
  --z0-npz outputs/kernel_N128_z0.npz \
  --A-npz outputs/kernel_N128_A.npz \
  --steps 20000 \
  --decode-every 200 \
  --decode-backend vulkan \
  --observer snapshots \
  --backend vulkan \
  --fft-backend vkfft-vulkan \
  --op-backend vulkan \
  --require-gpu \
  --metrics-json outputs/perf_snapshots_gpu.json
```

Long kernel-only GPU run with visuals:

```bash
MPLBACKEND=Agg python run_v4_snapshots.py \
  --kernel-only \
  --z0-npz outputs/kernel_N128_z0.npz \
  --A-npz outputs/kernel_N128_A.npz \
  --steps 20000 \
  --stride 200 \
  --no-ground-truth \
  --out-dir outputs \
  --prefix kernel_N128 \
  --backend vulkan \
  --op-backend vulkan \
  --decode-backend vulkan \
  --fft-backend vkfft-vulkan \
  --timing \
  --progress-every 200
```

Enstrophy graph from the snapshots metrics JSON:

```bash
python scripts/plot_enstrophy.py \
  --input outputs/perf_snapshots_gpu.json \
  --output outputs/enstrophy_kernel_only.png
```

GPU-only LES run (vkFFT + Vulkan, enstrophy CSV + optional PNGs):

```bash
MPLBACKEND=Agg python run_les_gpu.py \
  --N 512 \
  --steps 20000 \
  --dt 0.01 \
  --nu0 1e-4 \
  --Cs 0.17 \
  --stats-every 200 \
  --progress-every 2000 \
  --viz-every 2000 \
  --spectral-truncation exp \
  --trunc-alpha 36 \
  --trunc-power 8 \
  --out-dir outputs \
  --prefix les_gpu
```

Enstrophy plot from the LES CSV:

```bash
python scripts/plot_enstrophy.py \
  --input outputs/les_gpu_enstrophy.csv \
  --output outputs/enstrophy_les_gpu.png \
  --format csv \
  --title "LES GPU enstrophy"
```

Interactive CLI (recommended for day-to-day runs):

```bash
python dashi_cli.py les --interactive
python dashi_cli.py kernel --interactive
python dashi_cli.py plot --interactive
python dashi_cli.py compare --interactive
```

## Latest Run Results (2026-01-24, headless)
- `dashi_cfd_operator_v3.py` — success; baseline 300 steps in 0.619s (2.06 ms/step). Final relL2 0.473, corr 0.881, ΔE −1.221e-03, ΔZ −1.077e-01.
- `dashi_cfd_operator_v4.py` — success; baseline 300 steps in 0.627s (2.09 ms/step). Final relL2 0.648, corr 0.787, ΔE −2.38e-04, ΔZ −1.64e-02. Now preserves top-128 mid-band phases (indices fixed) and only synthesizes the remaining mid/high energy.
- `dashi_les_vorticity_codec.py` — success; codec stats: compression_ratio 0.714, relL2 0.03997, corr 0.9992, support_cells 4078.
- `dashi_les_vorticity_codec_v2.py` — success; sim 1.114s, codec 0.016s. q sweep: ratios 4.10→6.95, relL2 0.091→0.106, corr 0.996→0.994.
- `vortex_tester_mdl.py` — ran; only FigureCanvasAgg warnings (plots suppressed).
- `naw.py` — failed at `plot_3d_isosurface_voxels`: ValueError broadcasting (10,60,60) vs (11,61,61) after completing 10 g3 slices; also log10 divide-by-zero warnings.
- `naw2.py` — failed: `ModuleNotFoundError: No module named 'skimage'` (needs `scikit-image` for marching_cubes).
- `CORE_cfd_operator.py` — compares legacy numpy gating vs dashiCORE Carrier path. N=64, steps=120: legacy 3.05 ms/step; core (accelerated) 3.35 ms/step; core+fused mask 3.10 ms/step. Larger grids with fused mask: N=128 → legacy 8.21 ms/step vs core 8.18 (speedup 1.004×); N=256 → legacy 32.42 vs core 32.65 ms/step (speedup 0.993×). Enstrophy matches across runs; mask_ops≈0 on accelerated paths.
- `CORE_cfd_operator.py` @ 1024×1024, steps=120 (accelerated, fused mask) — legacy path not rerun; core fused path: 702.2 ms/step, enstrophy 0.00629, mask_mean 0.916. Final vorticity snapshot saved to `outputs/core_1024_final.png`.
- `outputs/v4_t300_compare.png` — side-by-side ω true / decoded+residual / error at t=300 (N=64) generated from `dashi_cfd_operator_v4.py` pipeline.
- `run_v4_snapshots.py` — CLI runner to save triptychs every stride: e.g., `MPLBACKEND=Agg python run_v4_snapshots.py --N 64 --steps 3000 --stride 300 --out-dir outputs --dpi 150 --figsize 14,5 --progress-every 100`. Supports `--pix-width/--pix-height` for exact pixels, `--traj-npz` to reuse a stored trajectory, `--save-traj` to write one, `--no-ground-truth` to skip ω_true/error panels (must pair with `--traj-npz`), `--timing` to print stage timings, `--dtype {auto,float32,float64}` (auto → float64 when N>1024), `--backend {cpu,accelerated,vulkan}` (best-effort; falls back to CPU if unavailable), and `--fft-backend {numpy,vkfft,vkfft-opencl,vkfft-vulkan}`. Vulkan path now tries vkFFT for FFTs (and falls back to NumPy if bindings/ICD missing). To force a specific ICD: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json MPLBACKEND=Agg python run_v4_snapshots.py ... --backend vulkan --fft-backend vkfft-vulkan`. Kernel-only start: `--kernel-only --z0-npz path.npz` where the file contains `z` (proxy vector), `mask_low` (bool mask), `anchor_idx` (int indices for preserved mid-band coeffs).
- `run_v4_snapshots.py` also accepts `--les-backend {cpu,gpu}` to generate ground-truth LES on GPU via `VulkanLESBackend` (still reads back each step for CPU-side encoding).
- When using `--les-backend gpu`, you can enable spectral truncation with `--spectral-truncation exp --trunc-alpha 36 --trunc-power 8`.
- `run_v4_snapshots.py` supports `--encode-backend gpu` to run the encode path on GPU; the first step bootstraps `anchor_idx` on CPU, then subsequent steps use GPU encode to reduce readback.
- `dashiCORE/scripts/run_vulkan_core_mask_majority.py` — GPU smoke test for the `core_mask_majority` compute shader; validates Vulkan carrier dispatch vs CPU majority vote. Requires `VK_ICD_FILENAMES` and python-vulkan/glslc; accepts `--n` (elements per channel) and `--k` (channels).

Recent GPU/vkFFT runs (user side):
- `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json MPLBACKEND=Agg python run_v4_snapshots.py --N 640 --steps 30 --stride 5 --out-dir outputs --backend vulkan --fft-backend vkfft-vulkan --dtype float64 --progress-every 5 --timing`
  - Result: completed; encode=17.69s, learn≈0, rollout≈0, decode_total=0.405s (0.068s/snap). vkFFT/Vulkan binding active (smoke: max error ~1.5e-6 on 256×256 complex64). ~600 MB system RAM; GPU VRAM not recorded.
- `... --N 1024 --steps 300 --stride 50 --backend vulkan --fft-backend vkfft-vulkan --dtype float64`
  - Result: encode overflow → NaNs; aborts at encode with `RuntimeError: Non-finite values in omega before encoding (try float64, smaller dt/Cs)`. Consider reducing `dt`/`Cs` or keeping N below ~1k for this LES integrator.
- Kernel-only reminder: provide a real z0 npz (`--z0-npz path`) or the runner will error (FileNotFound); kernel-only path skips LES entirely.
- Default params (N=64, steps=3000, stride=300, backend=vulkan, dtype=float64)
  - Result: saved `outputs/v4_t3000_compare.png`; timing encode=142.132s, learn=0.131s, rollout=0.235s, decode_total=0.043s (0.007s/snap).
- Example (streaming, saved traj, timing): `MPLBACKEND=Agg python run_v4_snapshots.py --N 640 --steps 30 --stride 5 --out-dir outputs --save-traj outputs/traj_saved.npz --progress-every 5 --timing` → encode=7.967s, learn=0.221s, rollout=0.010s, decode_total=0.321s (0.054s/snapshot); files `outputs/v4_t0005_compare.png` … `v4_t0030_compare.png`.
- At very large grids (e.g., N=6400), float32 LES overflows; use `--dtype float64` or reduce `dt`/`Cs`, and a NaN guard will fail fast if ω contains non-finite values.
- Example (streaming, saved traj, timing): `MPLBACKEND=Agg python run_v4_snapshots.py --N 640 --steps 30 --stride 5 --out-dir outputs --save-traj outputs/traj_saved.npz --progress-every 5 --timing`  
  Output (on this machine): encode=7.967s, learn=0.221s, rollout=0.010s, decode_total=0.321s (0.054s/snapshot). Images saved: `outputs/v4_t0005_compare.png`, …, `outputs/v4_t0030_compare.png`.

## Follow-Ups
- Fix `naw.py` voxel grid shape (matplotlib voxels expects edge-aligned arrays).
- Install `scikit-image` or add a stub to unblock `naw2.py`.
- Save plots to files when running headless (Agg) instead of calling `plt.show()`.
