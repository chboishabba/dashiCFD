# dashiCFD

CFD + DASHI experiments for 2D vorticity rollouts, ternary structural codecs, and SUSY gauge scans. Assets in the repo are mostly matplotlib outputs; simulations run in pure NumPy with optional Vulkan/vkFFT acceleration.

## Docs
- `docs/overview.md` — quickstart, runner map, and backend notes.
- `docs/signed_filament_annihilation.md` — operator semantics for signed filaments.

## How to Run
- Use Python 3.10+ with `numpy` and `matplotlib`; set `MPLBACKEND=Agg` for headless environments.
- Typical commands:
  - `MPLBACKEND=Agg python dashi_cfd_operator_v3.py`
  - `MPLBACKEND=Agg python dashi_cfd_operator_v4.py`
  - `MPLBACKEND=Agg python dashi_les_vorticity_codec_v2.py`
  - `MPLBACKEND=Agg python vortex_tester_mdl.py`
  - `MPLBACKEND=Agg CORE_BACKEND=cpu python CORE_cfd_operator.py`
  - `MPLBACKEND=Agg python run_v4_snapshots.py --N 64 --steps 3000 --stride 300 --out-dir outputs`
  - `MPLBACKEND=Agg python perf_kernel.py --z0-npz outputs/kernel_N128_z0.npz --A-npz outputs/kernel_N128_A.npz --steps 20000 --decode-every 200`
  - `MPLBACKEND=Agg python run_les_gpu.py --N 512 --steps 20000 --stats-every 200 --progress-every 2000`

Sprint 51 signed ternary flip audit:

```bash
python3 scripts/ns_signed_ternary_flip_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint51_signed_ternary_flip_gpu_audit
```

Observed six-run N32/N64 result: `NO2CYCLE_FAILS`. Raw cross-shell
minus-to-plus is `93419828142802.9`, plus-to-minus counter-flow is
`84731761817324.95`, and the signed imbalance fraction is
`0.048767829281919015`; the current failing diagnostic is the v1 no-2-cycle
proxy.

Sprint 52 material source / no-2-cycle amplitude audit:

```bash
python3 scripts/ns_sprint52_material_no2cycle_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint52_material_no2cycle_gpu_audit
```

Observed result: `MATERIAL_SOURCE_GATE_CLOSED_NO2CYCLE_AMPLITUDE_BLOCKED`.
Material true-new positive source is absent; the no-2-cycle amplitude proxy
remains the active blocker.

Sprint 53 no-2-cycle physical amplitude audit:

```bash
python3 scripts/ns_sprint53_no2cycle_physical_amplitude_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint53_no2cycle_physical_gpu_audit
```

Observed result: `MATERIAL_SOURCE_GATE_CLOSED_PHYSICAL_NO2CYCLE_AMPLITUDE_BLOCKED`.
Material true-new positive source remains absent, but the material
net-residue physical-amplitude proxy does not clear the sign-cycle gate:
physical-small fraction is `0.3423412506059137` and
`sigma_physical_cycle_fit = -1.1215088689186317`.

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

Sprint 49 3D GPU truth + material-parent audit:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
python3 scripts/make_truth_3d.py \
  --backend gpu \
  --fft-backend vkfft-vulkan \
  --N 64 \
  --steps 120 \
  --save-every 10 \
  --dt 0.001 \
  --nu0 0.001 \
  --seed 0 \
  --out outputs/truth3d/ns3d_N64_seed0_gpu.npz \
  --progress-every 10

VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
python3 scripts/ns_material_parent_summary.py \
  --truth outputs/truth3d/ns3d_N64_seed0_gpu.npz \
  --out-dir outputs/sprint49_material_parent_N64_seed0_gpu \
  --backend gpu \
  --fft-backend vkfft-vulkan \
  --diagnostic-precision float64 \
  --progress-every 4
```

Current Sprint 49 batch summary:

- `outputs/sprint49_material_parent_gpu_batch/sprint49_material_parent_gpu_batch_summary.csv`
- `outputs/sprint49_material_parent_gpu_batch/sprint49_material_parent_gpu_batch_summary.json`

Sprint 50 full ternary cross-shell audit from existing Sprint 49 artifacts:

```bash
python3 scripts/ns_ternary_cross_shell_matrix.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint50_full_ternary_cross_shell_gpu_audit
```

Current Sprint 50 batch routes as `CROSS_PLUS_FROM_MINUS_DOMINATES`; the
producer uses `parent_state -> child_state` and derives source kind from
`parent_relation` plus shell delta, not from Sprint 49 `classification`.

The current material-parent backend is
`gpu_spectral_gradient_cpu_packets`: fp64 Vulkan/vkFFT spectral derivatives are
computed on GPU, while packet matching and bin reduction are still CPU-side.
The next performance step is GPU packet-bin accumulation to avoid full
derivative readback.

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

NS->EV5 shell-enstrophy adapter (evidence-only projection diagnostics):

```bash
PYTHONPATH=. python scripts/make_truth.py \
  --backend cpu \
  --N 64 \
  --steps 100 \
  --stride 10 \
  --out outputs/truth/ns_ev5

python scripts/ns_ev5_shell_enstrophy.py \
  --truth outputs/truth/ns_ev5_cpu_YYYY-MM-DDTHHMMSS.npz \
  --out-dir outputs/ns_ev5_probe \
  --lane-mode ns-ev5 \
  --burn-in 1 \
  --q-tolerance 0 \
  --cycle-window 8
```

3D periodic incompressible truth artifact for physical bridge falsification:

```bash
python3 scripts/make_truth_3d.py \
  --N 32 \
  --steps 200 \
  --save-every 10 \
  --dt 0.002 \
  --nu0 0.001 \
  --seed 0 \
  --out outputs/truth3d/ns3d_N32_seed0.npz
```

Opt-in SPV/vkFFT lane using vendored `dashiCORE` Vulkan infrastructure:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
python3 -u scripts/make_truth_3d.py \
  --backend gpu \
  --fft-backend vkfft-vulkan \
  --N 32 \
  --steps 200 \
  --save-every 10 \
  --dt 0.002 \
  --nu0 0.001 \
  --seed 0 \
  --out outputs/truth3d/ns3d_N32_seed0_gpu.npz
```

Use the ICD path that exists on the host (`/usr/share/vulkan/icd.d/radeon_icd.json`
on this machine; some older notes use `radeon_icd.x86_64.json`). On the RX
580/gfx803 host, first verify the compatibility environment if Vulkan or ROCm
visibility is uncertain:

```bash
cd /home/c/Documents/code/__OTHER/gfx803_compat_graph/gfx803_flake_v1
nix develop .#base
verify-gfx803-host
```

The 3D truth lane is separate from the existing 2D DASHI/LES pipeline. It
writes `omega_snapshots` with shape `(T,N,N,N,3)`, `velocity_snapshots` with
the same vector shape by default, `steps`, diagnostics, and `meta_json`
declaring `dimension=3`, periodic boundaries, Leray projection, and 2/3
dealiasing. The CPU path is the default; the GPU path is fail-fast and records
the vkFFT backend, Vulkan device info, and 3D SPV shader list. The script fails
closed when saved fields are non-finite,
divergence/curl checks fail, CFL exceeds the configured bound, or fewer than
five vorticity shells are nonzero at or above `K_star`. GPU validation uses
fp32-appropriate divergence/curl tolerance floors and prints the actual
`fft_plan_backend` plus Vulkan device name so CPU/GPU execution is auditable.
Packet lineage labels are intentionally deferred until after the 3D physical
bridge harness runs.

FP64 GPU parity probe for diagnostics:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
python3 scripts/probe_vkfft_fp64.py --N 16
```

On the RX580/RADV host this reports `shaderFloat64=true` and complex128 vkFFT
round-trip error around `1e-15`. The fp64 path is a diagnostic/parity lane, not
the default speed lane; the 3D truth solver still defaults to fp32 GPU kernels.

No truth NPZ is required for CLI validation:

```bash
python scripts/ns_ev5_shell_enstrophy.py \
  --smoke \
  --out-dir outputs/ns_ev5_smoke
```

Residual atom codec probe (phase-aware signed anisotropic atoms; empirical
frame-bound test only):

```bash
python scripts/residual_atom_codec_probe.py \
  --truth outputs/truth_ns_ev5_cpu_2026-06-01T014101.npz \
  --snapshot-index -1 \
  --out outputs/atom_codec_probe_ns_ev5.json \
  --plot outputs/atom_codec_probe_ns_ev5.png \
  --max-atoms 16 \
  --peak-candidates 16 \
  --q 0.05
```

The probe writes selected atom fields, reconstruction metrics, and selected
Gram/frame diagnostics. It replaces random-phase residual synthesis with a
deterministic signed atom search for this experiment, but it is not a Gate3
proof, NS regularity theorem, or production codec replacement. Its selected
lower Gram eigenvalue is only an empirical `A > 0` probe for the
`AtomExtendedCarrierFrameReceipt` frame-bound obligation; it is not a uniform
lower frame-bound proof or continuum norm-comparison theorem.

The adapter writes `manifest.json`, `shell_enstrophy.csv`, `ev5_trace.csv`,
`theta_profile.csv`, and `checks.json`. The default `--lane-mode ns-ev5` uses the corrected
diagnostic dictionary: lane 2 is the bucketed enstrophy-weighted mean shell,
lane 3 is an adjacent cascade-flux ratio diagnostic only, lane 5 is secondary
top-shell occupancy, lane 7 is the dissipative enstrophy tail above `K*(nu)`,
and lane 11 is a `Z/3` phase/coherence proxy outside canonical FRACTRAN rules.
`Q_log` is still emitted as `lane2 + lane7`, but it is retained only as a
falsified scalar diagnostic from earlier checks, not as the live descent
criterion. The live vector EV5 check is lane 7 non-increasing together with
`mean_shell <= K_star + 1`. Lane 2 is a coordinate/boundedness witness rather
than Lyapunov energy, and lane 3 remains an adjacent cascade-flux diagnostic
excluded from Lyapunov/descent logic. Use `--lane-mode legacy-jmax-peak` only
to compare against the former `j_max` / peak-shell lane mapping.

For the corrected lane 7, the adapter reads `nu0`, `nu`, or `viscosity` from
`meta_json`, or accepts `--nu` / `--tail-k-star`. If the normalization guard
fails, `checks.json` fails closed instead of suppressing the falsification.
The theta runtime diagnostic is a finite cutoff/time vector:
`theta(k,t)=|Flux_tail(k,t)|/Diss_tail(k,t)` for fixed cutoffs `k >= K*(nu)`,
`Theta=max_k sup_t theta(k,t)`, and `danger_shell` is the argmax over that
fixed cutoff profile. It assumes no monotonicity in `k`; missing or zero
dissipation fails closed.
Finite-window EV5 cycles are flagged in a single trace, but become falsifying
resonant-cycle evidence only with a forced vs no-forcing comparison manifest.
These outputs are empirical diagnostics only; they do not discharge NS->EV5,
Gate3 norm, or Clay obligations.

Interactive CLI (recommended for day-to-day runs):

```bash
python dashi_cli.py les --interactive
python dashi_cli.py kernel --interactive
python dashi_cli.py plot --interactive
python dashi_cli.py compare --interactive
```

## Runners: which one is canonical?

Use `run_v4_snapshots.py` as the canonical runner for end-to-end DASHI CFD experiments (LES → encode → learn → rollout → decode + plots/metrics). It is the only script that exercises the full v4 pipeline with consistent CLI flags for backends, FFT selection, timing, and optional GPU encode/decode.

Other runners are targeted utilities:
- `run_les_gpu.py` — GPU LES only (no DASHI encode/learn/rollout); use it for LES stability/perf sweeps or to generate standalone LES diagnostics.
- `perf_kernel.py` — kernel-only (A/z) rollout + decode for performance/metrics; use it for backend micro-benchmarking and metrics-only runs.
- `dashi_cfd_operator_v3.py` / `dashi_cfd_operator_v4.py` — pure Python module demos; keep for reference/quick sanity runs but do not treat them as the primary CLI.
- `dashi_cli.py` — convenience wrapper that dispatches to the scripts above; `run_v4_snapshots.py` remains the canonical path under the hood for full runs.

Rule of thumb: if you want a result you’ll carry forward or compare across backends, run `run_v4_snapshots.py`.

## Latest Run Results (last recorded 2026-01-24, headless)
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
- `run_v4_snapshots.py` decode parity options: `--check-decode-parity` compares CPU vs Vulkan decode at snapshot steps and logs rel-L2/corr (plus low-pass/mask parity). `--parity-only` runs the parity check without plotting or ground truth; it implies `--check-decode-parity` and `--no-ground-truth`. When `--decode-backend vulkan` is requested, the runner now fails fast if GPU decode was not used (no silent CPU fallback).
- Vulkan decode semantics: annihilation now incorporates lifetime when deciding survival (long-lived support gets a slightly relaxed coherence threshold). This is a behavioral choice; rebuild SPIR-V after editing shaders via `python dashiCORE/scripts/compile_spv.py`.
- `run_v4_snapshots.py` also accepts `--les-backend {cpu,gpu}` to generate ground-truth LES on GPU via `VulkanLESBackend` (still reads back each step for CPU-side encoding).
- When using `--les-backend gpu`, you can enable spectral truncation with `--spectral-truncation exp --trunc-alpha 36 --trunc-power 8`.
- `run_v4_snapshots.py` supports `--encode-backend gpu` to run the encode path on GPU; the first step bootstraps `anchor_idx` on CPU, then subsequent steps use GPU encode to reduce readback.
- `dashiCORE/scripts/run_vulkan_core_mask_majority.py` — GPU smoke test for the `core_mask_majority` compute shader; validates Vulkan carrier dispatch vs CPU majority vote. Requires `VK_ICD_FILENAMES` and python-vulkan/glslc; accepts `--n` (elements per channel) and `--k` (channels).
 - Update policy: for fresh metrics, re-run the relevant script and append a timestamped summary to `COMPACTIFIED_CONTEXT.md`.

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

## Other Utilities
- `make_kernel_artifacts.py` — generate kernel-only A/z artifacts from a v4 run.
- `scripts/plot_enstrophy.py` — plot enstrophy from JSON or CSV logs.
- `scripts/compare_les_gpu_cpu.py` — sanity check GPU LES vs CPU baseline.
- `scripts/validate_gpu_truth.py` — validate GPU LES against stored truth.
- `scripts/make_truth_3d.py` — pseudo-spectral 3D periodic incompressible
  truth generator for velocity/vorticity bridge artifacts; defaults to CPU and
  has an opt-in SPV/vkFFT GPU lane via `--backend gpu`.
- `scripts/ns_ev5_shell_enstrophy.py` — convert `make_truth.py` snapshots into
  shell-enstrophy and EV5 candidate diagnostic bundles.
- `scripts/run_sweep.py` / `scripts/perf_sampler.py` — parameter sweeps and perf sampling.
