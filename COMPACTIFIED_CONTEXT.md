# COMPACTIFIED_CONTEXT

Date: 2026-01-24 (MPLBACKEND=Agg runs).

## Repo Map
- `dashi_cfd_operator_v3.py` / `v4.py`: spectral LES rollouts + DASHI residual codec (v4 adds residual closure).
- `dashi_les_vorticity_codec.py` / `_v2.py`: ternary mask codec experiments and rate–distortion sweeps.
- `vortex_tester_mdl.py`: minimal 2D vorticity sandbox with DASHI gating.
- `naw.py` / `naw2.py`: SUSY gauge scan utilities; 3D voxel visualization in `naw.py` and marching-cubes variant in `naw2.py`.
- Assets: many PNG figures; `dashi_signed_branchedflow_codec.npz` weights.

## Execution Outcomes (headless)
- `dashi_cfd_operator_v3.py`: ok. 300-step LES 0.619s; relL2 0.473, corr 0.881; ΔE −1.221e-03, ΔZ −1.077e-01.
- `dashi_cfd_operator_v4.py`: ok. 300-step LES 0.598s; relL2 0.688, corr 0.763; ΔE −7.59e-04, ΔZ −9.25e-04.
- `dashi_les_vorticity_codec.py`: ok. compression_ratio 0.714; relL2 0.03997; corr 0.9992; support_cells 4078.
- `dashi_les_vorticity_codec_v2.py`: ok. sim 1.114s, codec 0.016s; q sweep ratios 4.10→6.95 with relL2 0.091→0.106, corr 0.9958→0.9944.
- `vortex_tester_mdl.py`: ok with FigureCanvasAgg warnings (plots suppressed).
- `naw.py`: error at `plot_3d_isosurface_voxels` (ValueError broadcasting (10,60,60) vs (11,61,61) after 10 g3 slices); log10 divide-by-zero warnings.
- `naw2.py`: missing dependency `skimage.measure.marching_cubes` (install `scikit-image`).
- `CORE_cfd_operator.py`: benchmarks legacy numpy gating vs dashiCORE Carrier path. N=64, steps=120 → legacy 3.05 ms/step; core (accelerated) 3.35; core+fused 3.10 (enstrophy 0.4834). Larger grids (fused, accelerated): N=128 → legacy 8.21 vs core 8.18 ms/step (speedup 1.004×); N=256 → legacy 32.42 vs core 32.65 ms/step (speedup 0.993×); N=1024 → core fused 702.2 ms/step (legacy not rerun), enstrophy 0.00629, mask_mean 0.916; snapshot saved at `outputs/core_1024_final.png`; mask_ops≈0.
- `outputs/v4_t300_compare.png`: three-panel visualization (ω true, decoded+residual, error) at t=300 from the v4 pipeline (N=64).
- v4 now retains top-128 mid-band complex coefficients (fixed indices) and synthesizes only the remaining mid/high energy; proxy dimension grows by 2*K floats, rollout stays linear.
- `run_v4_snapshots.py`: CLI runner to emit comparison triptychs every stride (defaults: N=64, steps=3000, stride=300) into an output directory; supports `--dpi`, `--figsize W,H`, `--pix-width/--pix-height`, `--progress-every`, `--traj-npz` (reuse stored trajectory), `--save-traj` (write trajectory), `--no-ground-truth` (skip true/error panels; requires stored traj), `--dtype {auto,float32,float64}` (auto→float64 when N>1024), `--backend {cpu,accelerated,vulkan}` best-effort, `--fft-backend {numpy,vkfft,vkfft-opencl,vkfft-vulkan}` (vkFFT attempts GPU FFTs; falls back to NumPy), `--kernel-only --z0-npz` to start from a saved proxy (keys: z, mask_low, anchor_idx), and `--timing` to print encode/learn/rollout/decode times. To force an ICD: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json MPLBACKEND=Agg python run_v4_snapshots.py ... --backend vulkan --fft-backend vkfft-vulkan`. Example streaming run (N=640, steps=30, stride=5) saved `outputs/v4_t0005_compare.png`…`v4_t0030_compare.png` with timings: encode=7.97s, learn=0.22s, rollout=0.01s, decode_total=0.32s (0.054s/snapshot). At N=6400 with float32, LES overflows; use float64 or smaller dt/Cs (NaN guard now fails fast).
- `scripts/run_vulkan_core_mask_majority.py`: Vulkan carrier smoke test for `core_mask_majority.comp`; compares GPU majority fusion to CPU reference. Needs `VK_ICD_FILENAMES`, glslc, python-vulkan; flags `--n` (elements per channel) and `--k` (channels).
- Latest user GPU/vkFFT runs:
  - N=640, steps=30 (vkfft-vulkan, dtype=float64) completed; timings encode=16.79s, learn=0.029s, rollout≈0, decode_total=0.369s (0.061s/snap); backend=vulkan; ~600 MB RAM; snapshots saved t=5..30.
  - N=1024, steps=300 (vkfft-vulkan, dtype=float64) overflowed during encode (LES produced NaNs); runner aborted with `RuntimeError: Non-finite values in omega before encoding (try float64, smaller dt/Cs)`. Mitigation: shrink dt/Cs or reduce N.
  - Kernel-only requires a real `--z0-npz` path; placeholder `path/to/z0.npz` will raise FileNotFoundError.
  - GPU note: runs showed system RAM use only; GPU utilization was not observed, suggesting vkFFT fell back to CPU. Check ICD/vkfft bindings and monitor with `nvidia-smi`/`radeontop` to confirm GPU dispatch.
  - Default params (N=64, steps=3000, stride=300, backend=vulkan, dtype=float64): saved `outputs/v4_t3000_compare.png`; timing encode=142.132s, learn=0.131s, rollout=0.235s, decode_total=0.043s (0.007s/snap).

## Notes / Next Actions
- Fix voxel grid sizing in `naw.py` (matplotlib voxels expects edge-sized coordinates, e.g., len+1 along each axis).
- Add dependency set (`numpy`, `matplotlib`, `scikit-image` if keeping `naw2.py`).
- Replace `plt.show()` calls with file saves for reliable headless runs.
- Optimize `CORE_cfd_operator.py` further (e.g., jit/numba for ternary majority, optional GPU backend hook).
- Vulkan decode backend plan (2026-01-25) recorded at `planning/vulkan_decode_stage1.md`; corresponding tasks in `TODO.md` (metrics fields, Stage 1 GPU low-pass, Stage 2 mask kernels, CLI consistency, parity tests, fp32/fp64 policy, shader artifact location).
- Vulkan decode backend implemented (Stage 1–2): new `vulkan_decode_backend.py` with vkFFT/Vulkan pipelines for low-pass + DASHI mask (smooth/threshold/majority), new shaders under `dashiCORE/gpu_shaders/`, and CLI flags `--decode-backend` / `--permissive-backends` wired into `perf_kernel.py` and `run_v4_snapshots.py`. Metrics now log requested/used backends and `gpu_hotloop_active`. Residual synthesis still CPU-side (apply-mask combine pending GPU).
