# COMPACTIFIED_CONTEXT

Date: 2026-02-06 (cache synced to local ~/.chatgpt_history.sqlite3 as of 2026-02-05).

## 2026-06-04 Sprint 64 NS Source-Budget Verdict and CKN Norm Switch Plan

- Sprint 63 completed the remaining simple DASHI-native raw-action fork:
  cross-shell parent credit is not raw-action contractive on current artifacts.
  The six-run N32/N64, N128 seed0, and dense N64 seed0 runs all show strong
  noncontractive replenishment in the raw positive-action parent-budget ledger.
- The NS source-budget route is now recorded as diagnostically exhausted under
  the tested objects: normalized packet action, raw action shell summability,
  action-preserving shell reassignment, raw-red direction coherence, and simple
  cross-shell parent-budget contractivity.
- The next NS route is a norm switch, not another color/shell/action-budget
  audit. Sprint 64 should align DASHI `Overflow` semantics with the CKN/ESS
  critical concentration surface: local scale-critical velocity `L3` and,
  when available, pressure `L^(3/2)` on parabolic cylinders.
- Existing truth artifacts contain `velocity_snapshots` and no pressure array,
  so Sprint 64 should produce a velocity-only critical concentration diagnostic
  and mark pressure reconstruction missing. It must keep all Clay/NS promotion
  flags false.
- Implemented `scripts/ns_sprint64_ckn_local_critical_concentration_audit.py`
  and `tests/test_sprint64_ckn_local_critical_concentration_audit.py`.
- Six-run N32/N64 result:
  `outputs/sprint64_ckn_local_critical_concentration_gpu_audit/`,
  `route_decision = LOCAL_CRITICAL_CONCENTRATION_PRESSURE_RECONSTRUCTION_MISSING`,
  `row_count = 18720`, `ascended_fraction = 0.9642628205128205`, and
  `max_local_concentration_ratio = 53.704773818909864`.
- Follow-up N128 seed0 result:
  `outputs/sprint64_ckn_local_critical_concentration_N128_seed0_gpu/`,
  `ascended_fraction = 0.8740818643162394`, and
  `max_local_concentration_ratio = 10.67887709906586`.
- Follow-up dense N64 seed0 save-every=2 result:
  `outputs/sprint64_ckn_local_critical_concentration_N64_seed0_gpu_dense2/`,
  `ascended_fraction = 0.926542577413479`, and
  `max_local_concentration_ratio = 24.25368457905771`.
- Interpretation: the norm switch is implemented as a velocity-only preflight,
  but current artifacts cannot support a full CKN certificate because the
  pressure `L^(3/2)` term is absent.

## 2026-06-04 Sprint 65 Pressure Reconstruction Plan

- Sprint 64 routed as pressure-reconstruction-missing on all current truth
  artifacts. Sprint 65 should derive periodic zero-mean pressure snapshots from
  stored `velocity_snapshots` using the incompressible Poisson equation
  `Delta p = -sum_ij d_i u_j d_j u_i`.
- The producer should write pressure-augmented NPZ artifacts without changing
  the source truth arrays, record residual/gauge diagnostics, and then allow
  Sprint 64 to run with the pressure term present. It remains diagnostic; no
  CKN epsilon theorem or Clay/NS promotion follows.
- Implemented `scripts/ns_sprint65_pressure_reconstruction.py` and
  `tests/test_sprint65_pressure_reconstruction.py`.
- Six-run N32/N64 pressure reconstruction:
  `outputs/sprint65_pressure_reconstruction_gpu_audit/`,
  `max_poisson_relative_residual_rms = 3.5409688067143674e-16`.
- Pressure-present Sprint 64 rerun:
  `outputs/sprint64_ckn_local_critical_concentration_pressure_gpu_audit/`,
  `route_decision = LOCAL_CRITICAL_CONCENTRATION_MIXED`,
  `ascended_fraction = 0.9890491452991453`, and
  `max_local_concentration_ratio = 60.83081878566949`.
- N128 seed0 pressure-present rerun routes `LOCAL_CRITICAL_CONCENTRATION_MIXED`,
  with `ascended_fraction = 0.9127771100427351`.
- Dense N64 seed0 pressure-present rerun routes
  `LOCAL_CRITICAL_CONCENTRATION_MIXED`, with
  `ascended_fraction = 0.9361338797814208`.
- Interpretation: pressure reconstruction removes the artifact-level missing
  pressure gate, but the CKN route remains open/blocked by threshold
  calibration and theorem-grade interpretation of the pressure-inclusive local
  concentration surface.

## 2026-06-04 Sprint 66 CKN r-sweep Calibration Plan

- Sprint 65 made the full pressure-inclusive CKN diagnostic measurable, but a
  fixed `epsilon_critical = 0.01` over broad non-overlapping blocks classifies
  most sampled cylinders as ascended.  That is a proxy-calibration result, not
  a near-singularity verdict.
- Sprint 66 should replace the single fixed-threshold view with a
  candidate-centred r-sweep.  At the hottest packet or field centres it should
  compute
  `C(r) = r^-2 integral_Q (|u|^3 + |p|^(3/2)) dx dt`
  for multiple radii and record whether `C(r)` decays, stays flat, or grows as
  the diagnostic zooms inward.
- Inputs should be pressure-augmented Sprint 65 truth NPZs.  Optional Sprint 59
  raw-action packet CSVs may seed candidate centres; otherwise the producer
  should fall back to top pointwise pressure-inclusive CKN density centres.
- Sprint 66 remains diagnostic only.  It does not apply a CKN
  epsilon-regularity theorem, does not prove a suitable weak solution bridge,
  does not prove continuum-uniform bounds, and does not promote Clay/NS.
- Implemented `scripts/ns_sprint66_ckn_r_sweep_calibration.py` and
  `tests/test_sprint66_ckn_r_sweep_calibration.py`.
- Six-run N32/N64 pressure-present result:
  `outputs/sprint66_ckn_r_sweep_calibration_gpu_audit/`,
  `route_decision = CKN_R_SWEEP_DECAYS_UNDER_ZOOM`, `hotspot_count = 60`,
  `ascended_fraction = 0.43666666666666665`, `decaying_hotspot_count = 60`,
  and `concentrating_hotspot_count = 0`.
- N128 seed0 result:
  `outputs/sprint66_ckn_r_sweep_calibration_N128_seed0_gpu/`,
  `route_decision = CKN_R_SWEEP_DECAYS_UNDER_ZOOM`, `ascended_fraction = 0.116`,
  `decaying_hotspot_count = 10`, and `concentrating_hotspot_count = 0`.
- Dense N64 seed0 result:
  `outputs/sprint66_ckn_r_sweep_calibration_N64_seed0_gpu_dense2/`,
  `route_decision = CKN_R_SWEEP_DECAYS_UNDER_ZOOM`,
  `ascended_fraction = 0.125`, `decaying_hotspot_count = 10`, and
  `concentrating_hotspot_count = 0`.
- Interpretation: the Sprint 64/65 high fixed-block ascended fractions are
  strongly demoted by candidate-centred r-sweeps.  The tested hot spots behave
  like bulk turbulence under zoom, not concentration candidates.  This is
  favorable diagnostic evidence only; it is not a universal CKN theorem.

## 2026-06-04 Sprint 63 Cross-Shell Replenishment Contractivity Plan

- Sprint 60 showed Euclidean, smoothed, and provisional BT shell
  reassignment conserves raw action but leaves the raw-action shell fit flat.
- Sprint 61/62 showed the high raw-red packet population is direction
  incoherent on the available N64, dense N64, and N128 evidence, so the
  immediate CFM coherent-tube rescue is blocked diagnostically.
- The remaining DASHI-native NS fork is now cross-shell replenishment:
  determine whether adjacent/cross-shell parent credit is support/defect
  non-amplifying rather than a true-new source.
- Sprint 63 should join Sprint 49 material-parent edges to Sprint 59 raw-action
  packet rows and compare each child packet's raw positive action against its
  credited parent raw-action budget
  `A_raw_positive(parent) * credited_mass / parent_mass`.
- The audit remains diagnostic only. It does not prove support non-creation,
  defect monotonicity, stretch absorption, no finite-time blowup, or any
  Clay/NS promotion. Its formal target is a future
  `AdjacentCrossShellReplenishmentSummable` theorem.
- Implemented `scripts/ns_sprint63_cross_shell_replenishment_contractivity_audit.py`
  and `tests/test_sprint63_cross_shell_replenishment_contractivity_audit.py`.
- Six-run N32/N64 result:
  `outputs/sprint63_cross_shell_replenishment_contractivity_gpu_audit/`,
  `route_decision = CROSS_SHELL_REPLENISHMENT_MIXED`,
  `contractivity_ratio_total = 2.7665497780287076`,
  `weighted_contractivity_ratio_total = 2.9828906939689044`, and
  `noncontractive_edge_fraction = 0.9532640658694373`.
- Follow-up N128 seed0 result:
  `outputs/sprint63_cross_shell_replenishment_contractivity_N128_seed0_gpu/`,
  `contractivity_ratio_total = 4.371227592340793`,
  `weighted_contractivity_ratio_total = 5.806885413286424`, and
  `noncontractive_edge_fraction = 0.9987681013676589`.
- Follow-up dense N64 seed0 save-every=2 result:
  `outputs/sprint63_cross_shell_replenishment_contractivity_N64_seed0_gpu_dense2/`,
  `contractivity_ratio_total = 2.6548745195597747`,
  `weighted_contractivity_ratio_total = 2.8200893072403987`, and
  `noncontractive_edge_fraction = 0.9901806026277914`.
- Interpretation: the simple raw positive-action parent-budget contractivity
  theorem is blocked on current artifacts. A future NS proof lane needs a
  stronger defect/admissibility quotient or a pivot to CFM/BKM/concentration
  compactness; more color-string/shell/simple-parent-budget diagnostics are no
  longer the shortest path.

## 2026-06-04 Sprint 55 Lagrangian Stretch-Action Audit

- Added `scripts/ns_sprint55_lagrangian_stretch_action_audit.py` and
  `tests/test_sprint55_lagrangian_stretch_action_audit.py`.
- The producer consumes Sprint 49 material-parent tables and truth snapshots,
  follows `parent_packet_id -> child_packet_id` material lineages, and
  accumulates shell/time normalized `omega dot S omega / (|omega|^2 + eps)`.
- Six-run N32/N64 GPU batch:
  `outputs/sprint55_lagrangian_stretch_action_gpu_audit/`.
- Result: `LAGRANGIAN_STRETCH_ACTION_SMALL_DIAGNOSTIC`,
  `action_small_fraction = 0.9985242030696576`, `dangerous_lineage_count = 5`,
  and `sigma_action_fit = -0.5102412568825301`.
- Interpretation: the Sprint 54 direct-stretch evidence is now read as
  Lagrangian accumulated stretch-action evidence, not color strings or
  packet-color counts. Packet-local masks are still unavailable, weighted action
  summability remains open, and all promotion flags remain false.

## 2026-06-04 Sprint 56 Packet-Local Stretch-Action Audit

- Added `scripts/ns_sprint56_packet_local_stretch_action_audit.py` and
  `tests/test_sprint56_packet_local_stretch_action_audit.py`.
- The producer reconstructs packet support masks from Sprint 49 `K_cell` packet
  IDs and `packet_grid`, computes packet-local accumulated
  `omega dot S omega / |omega|^2`, and records direction-change separation.
- Six-run N32/N64 GPU batch:
  `outputs/sprint56_packet_local_stretch_action_gpu_audit/`.
- Result: `PACKET_LOCAL_ACTION_SUMMABILITY_BLOCKED`,
  `packet_local_available_fraction = 1.0`, `action_small_fraction =
  0.8108028335301063`, `dangerous_lineage_count = 641`, and
  `sigma_packet_local_action_fit = -0.4822543927548197`.
- Interpretation: Sprint 55's shell-lineage action-small signal was
  overoptimistic. Packet-local masks make the accumulated-action route blocked
  under current cadence/resolution. All promotion flags remain false.

## 2026-06-04 Sprint 57 Vessel/Action Reconciliation Audit

- Added `scripts/ns_sprint57_vessel_action_reconciliation_audit.py` and
  `tests/test_sprint57_vessel_action_reconciliation_audit.py`.
- The producer compares Sprint 49/56 Euclidean `K_cell` packet-local stretch
  action against whole-domain and covered-mask `omega dot S omega` action.
- Six-run N32/N64 GPU batch:
  `outputs/sprint57_vessel_action_reconciliation_gpu_audit/`.
- Result: `PACKET_ACTION_UNDERCOUNTS_COVERED_STRETCH`,
  `epsilon_raw_positive_vs_covered = -0.8161321565334568`,
  `epsilon_raw_positive_vs_global = -0.9608719590659198`, and
  `epsilon_normalized_positive_vs_global = 113.58553013012235`.
- Interpretation: Sprint 56 is not explained by simple Euclidean packet
  double-counting. Raw packet action under-reconstructs vessel action, while
  normalized packet action is inflated relative to global normalized action.
  All promotion flags remain false.

## 2026-06-04 Sprint 58 Normalized Packet-Action Inflation Audit

- Added `scripts/ns_sprint58_normalized_action_inflation_audit.py` and
  `tests/test_sprint58_normalized_action_inflation_audit.py`.
- The producer decomposes Sprint 56/57's normalized action mismatch as
  sum-of-local-packet-ratios versus ratio-of-sums over the covered/global
  vessel ledger.
- Six-run N32/N64 GPU batch:
  `outputs/sprint58_normalized_action_inflation_gpu_audit/`.
- Result: `NORMALIZED_ACTION_NONADDITIVE_RATIO_INFLATION`,
  `sum_ratios_over_ratio_of_sums_covered = 4904.346096600663`,
  `sum_ratios_over_ratio_of_sums_global = 11471.817018880183`, and
  `low_enstrophy_denominator_fraction = 0.012394729693018202`.
- Interpretation: packet-normalized `A+` is not vessel-additive. The next NS
  object should be raw positive action or energy-weighted normalized action.
  All promotion flags remain false.

## 2026-06-04 Sprint 49 Material-Parent GPU Batch

- Added `scripts/ns_material_parent_summary.py` and
  `tests/test_material_parent_summary.py`.
- Current producer backend `gpu_spectral_gradient_cpu_packets` computes fp64
  Vulkan/vkFFT spectral derivatives for stretch-state classification, then does
  packet matching/bin reduction on CPU.
- Ran GPU truth + material-parent batches for N32/N64 seed0/seed1 on RADV RX
  580. Batch summary:
  `outputs/sprint49_material_parent_gpu_batch/sprint49_material_parent_gpu_batch_summary.json`.
- Result: `weighted_true_new = 0` and `sigma_true_new = 0` in all four runs;
  tracking uncertainty is zero or small; weighted cross-shell source dominates.
  All runs route to `ADJACENT_PACKET_THEOREM_INSUFFICIENT`.
- Next performance step: compute/read back compact `stretch_state` or packet-bin
  accumulators on GPU instead of full derivative tensors.

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
