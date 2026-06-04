# COMPACTIFIED_CONTEXT

Date: 2026-02-06 (cache synced to local ~/.chatgpt_history.sqlite3 as of 2026-02-05).

## Context Freshness (cached + newly pulled 2026-02-06)
- `6976f652-cf80-8324-bc14-e764bddc7316` (Branch · DASHI vs LES, 2026-01-27): Vulkan `perf_kernel.py` run succeeded; next steps were documentation + hygiene after proving GPU path.
- `6974cb58-dbe0-8321-b2b5-ce0429cb4480` (DASHI vs LES, 2026-01-25): Sprint plan “Autonomous Operator Generalization” with objectives, acceptance tests, deliverables; keep GPU/Vulkan path proven, focus on permission gating (no new size math).
- `69757e17-3998-8322-82ec-66e23cc70232` (CFD Folder Shared Fix, 2026-01-25): RX 580/Polaris lacks `VK_KHR_video_encode_queue`; Vulkan video encoders (`h264_vulkan`, `hevc_vulkan`, `av1_vulkan`) are impossible on this GPU. Use CPU VP9 (`libvpx-vp9`) or AMD VAAPI (`h264_vaapi` with `format=nv12,hwupload`, `-vaapi_device /dev/dri/renderD128`). Vulkan remains correct for compute (VkFFT, CFD).
- `6960722d-3e28-8323-991b-912a640ce570` (DASHI Physics, 2026-01-24): Formal interpretation of ω/ω̂/error triptychs; clarifies what plots prove and open gaps.
- `6974163f-4634-8324-870c-7b926e2d2f27` (dashiCORE README Outline, 2026-01-24): Plan to inventory functions and build efficiency surfaces across workloads/backends.
- `6965ba57-f500-8322-96d9-0e3db4de9220` (Branch · DASHI learner context5 -- trading, 2026-01-15): Trading state—BTC hard-blocked; ES/NQ need effect-size; patch plan provided.
- Older conceptual threads kept for provenance: `696da482-ca2c-8322-b792-45dc313a3d06` (DASHI Atom vacuum formalism), `6966fb93-2c38-8327-97e4-b2074aa44e0a` (Phase-3 surfaces complete), `69607177-6760-8323-b0a8-f22afbb6b455` (MDL canonical), `69604a2e-606c-8323-aed1-bd6158f73e5a` (discrete branches from continuous systems), `69634147-dca4-8321-a20d-34635b170aaf` (gauge/invariants/legal inputs), and `696b37ae-17f8-8324-b485-4fc779fd7262` / `696b3f34-ddf4-8323-acfe-94a5091260cb` / `696b2307-8d70-8321-9db0-bc21b4f9f297` (formalizing DASHI kernel).

## Missing exports (no messages cached; needs download/ingest)
- `696504b3-18f8-832d-825c-4f79eda29201`
- `6966ea7e-55e4-8322-8d81-2096ecc0f4e5`
- `690828e5-af20-8320-956c-e1b09cea911d`
If these matter, export them from ChatGPT and re-ingest before updating context.

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
- 2026-06-03: added a separate CPU-first 3D periodic incompressible truth lane
  at `scripts/make_truth_3d.py`. It emits `omega_snapshots` shaped
  `(T,N,N,N,3)`, `velocity_snapshots` by default, `steps`, diagnostics, and
  `meta_json` with `dimension=3`, `projection=leray`, `dealiasing=2/3`,
  `periodic=true`, `dt`, `nu0`, `N`, `seed`, and backend. This is for physical
  bridge falsification only; no NS/Clay promotion is implied. Packet lineage
  labels remain deferred until a 3D truth artifact has been consumed by the
  harness.
- 2026-06-04: added an opt-in `--backend gpu` path for `make_truth_3d.py` via
  `vulkan_truth3d_backend.py`, vendored `dashiCORE` vkFFT/Vulkan helpers, and
  3D SPV shaders under `dashiCORE/spv/comp/*_3d.comp`. CPU remains the default;
  GPU is fail-fast and records device/shader metadata.
- On this host the GPU truth lane requires explicit RADV ICD selection:
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`. Zero-step and
  two-step N=16 GPU smokes reported `fft_plan_backend=vulkan` on
  `AMD Radeon RX 580 Series (RADV POLARIS10)` with finite velocity/vorticity
  fields.
- 2026-06-04: added a diagnostic fp64 Vulkan/vkFFT path for NS harness parity.
  `scripts/probe_vkfft_fp64.py --N 16` reports `shaderFloat64=true` on the
  RX580/RADV host and complex128 vkFFT round-trip max error about `1.4e-15`.
  The sibling `dashi_agda` harness can now request
  `--diagnostic-backend gpu --diagnostic-precision float64`; this is a
  debug/parity lane, not the default fp32 truth-solver speed lane.
- Fix voxel grid sizing in `naw.py` (matplotlib voxels expects edge-sized coordinates, e.g., len+1 along each axis).
- Add dependency set (`numpy`, `matplotlib`, `scikit-image` if keeping `naw2.py`).
- Replace `plt.show()` calls with file saves for reliable headless runs.
- Optimize `CORE_cfd_operator.py` further (e.g., jit/numba for ternary majority, optional GPU backend hook).
- Vulkan decode backend plan (2026-01-25) recorded at `planning/vulkan_decode_stage1.md`; corresponding tasks in `TODO.md` (metrics fields, Stage 1 GPU low-pass, Stage 2 mask kernels, CLI consistency, parity tests, fp32/fp64 policy, shader artifact location).
- Vulkan decode backend implemented (Stage 1–2): new `vulkan_decode_backend.py` with vkFFT/Vulkan pipelines for low-pass + DASHI mask (smooth/threshold/majority), new shaders under `dashiCORE/gpu_shaders/`, and CLI flags `--decode-backend` / `--permissive-backends` wired into `perf_kernel.py` and `run_v4_snapshots.py`. Metrics now log requested/used backends and `gpu_hotloop_active`. Residual synthesis still CPU-side (apply-mask combine pending GPU).
