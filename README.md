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
- `run_v4_snapshots.py` — CLI runner to save triptychs every stride: e.g., `MPLBACKEND=Agg python run_v4_snapshots.py --N 64 --steps 3000 --stride 300 --out-dir outputs --dpi 150 --figsize 14,5 --progress-every 100`. Supports `--pix-width/--pix-height` for exact pixels, `--traj-npz` to reuse a stored trajectory, `--save-traj` to write one, `--no-ground-truth` to skip ω_true/error panels (must pair with `--traj-npz`), and `--timing` to print encode/learn/rollout/decode timings.

## Follow-Ups
- Fix `naw.py` voxel grid shape (matplotlib voxels expects edge-aligned arrays).
- Install `scikit-image` or add a stub to unblock `naw2.py`.
- Save plots to files when running headless (Agg) instead of calling `plt.show()`.
