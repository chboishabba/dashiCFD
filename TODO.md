# TODO (2026-01-25)

## Vulkan decode backend (Stage 1–2)
- [x] Add metrics fields: decode/op/fft requested vs used; `gpu_hotloop_active`; strict/permissive fallback switch.
- [x] Implement Stage 1 GPU low-pass decode: persistent buffers, scatter low-k, vkFFT IFFT, timings, optional readback only.
- [x] Implement Stage 2 mask on GPU: smooth → ternary threshold → majority (iterative dispatch), int8 mask/support (combine/apply still CPU-side for residual synthesis).
- [x] Wire `--decode-backend {cpu,vulkan}` + strict/permissive flag consistently into `perf_kernel.py` and `run_v4_snapshots.py`; log stage timings.
- [ ] Add parity/smoke tests: CPU vs Vulkan low-pass MAE/energy/enstrophy + hash/log checks; ensure kernel-only perf path unchanged.

## Misc cleanup
- [ ] Decide fp32 vs fp64 decode default on AMD; document chosen policy.
- [ ] Choose location/policy for compiled `.spv` artifacts (single source of truth).

## Docs
- [x] Add `docs/overview.md` quickstart + runner map + backend notes.
- [x] Refresh `README.md` with runner pointers and utility scripts list.

## 3D NS truth bridge
- [x] Add CPU-first `scripts/make_truth_3d.py` that emits 3D periodic incompressible velocity/vorticity truth NPZ artifacts.
- [x] Add opt-in SPV/vkFFT 3D backend using vendored `dashiCORE` Vulkan infrastructure; keep CPU default and fail-fast GPU semantics.
- [x] Run GPU `make_truth_3d.py --backend gpu` smoke on target RADV/RX580 host with explicit ICD; validated zero-step and two-step N=16 artifacts.
- [x] Add fp64 Vulkan/vkFFT probe and diagnostic-only spectral backend for CPU/GPU harness parity; RX580/RADV reports `shaderFloat64=true`, complex128 vkFFT round-trip max error `~1.4e-15`, and N32 harness active-row `Q_K` parity `~3e-14` relative.
- [x] Add Sprint 49 material-parent producer `scripts/ns_material_parent_summary.py` for GPU truth artifacts. Current backend `gpu_spectral_gradient_cpu_packets` uses Vulkan/vkFFT fp64 spectral derivatives for stretch state and CPU packet matching/bin reduction.
- [x] Run Sprint 49 GPU material-parent batches for N32/N64 seed0/seed1. Outcome: true-new source is zero across all four runs; tracking is small when present; weighted cross-shell source dominates and routes to `ADJACENT_PACKET_THEOREM_INSUFFICIENT`.
- [x] Add Sprint 50 full ternary cross-shell matrix producer `scripts/ns_ternary_cross_shell_matrix.py`. The six-run N32/N64 batch routes as `CROSS_PLUS_FROM_MINUS_DOMINATES`; source kind is controlled by `parent_relation` plus shell delta, while transition entries use `parent_state -> child_state`.
- [x] Add Sprint 51 signed ternary flip producer `scripts/ns_signed_ternary_flip_audit.py`. The six-run N32/N64 batch routes as `NO2CYCLE_FAILS`; raw minus-to-plus is largely balanced by plus-to-minus, but the v1 packet-ID no-2-cycle proxy reports persistent sign-cycle failures.
- [x] Add Sprint 52 material source / no-2-cycle amplitude producer `scripts/ns_sprint52_material_no2cycle_audit.py`. The six-run N32/N64 batch routes as `MATERIAL_SOURCE_GATE_CLOSED_NO2CYCLE_AMPLITUDE_BLOCKED`: material true-new source is absent, while no-2-cycle amplitude remains blocked under the v1 material-packet proxy.
- [x] Add Sprint 53 no-2-cycle physical amplitude producer `scripts/ns_sprint53_no2cycle_physical_amplitude_audit.py`. The six-run N32/N64 batch routes as `MATERIAL_SOURCE_GATE_CLOSED_PHYSICAL_NO2CYCLE_AMPLITUDE_BLOCKED`: material true-new source remains absent, but the material net-residue physical-amplitude proxy reports only `0.3423412506059137` small failing cycles and `sigma_physical_cycle_fit = -1.1215088689186317`.
- [ ] Compare longer GPU rollout parity against CPU before using GPU truth artifacts as primary harness evidence.
- [ ] Run external physical bridge harness against `outputs/truth3d/ns3d_N32_seed0.npz` and record whether `C_K` stays bounded.
- [ ] Move Sprint 49 packet-bin accumulation onto GPU so the producer reads back `stretch_state` or compact packet-bin accumulators instead of full derivative tensors.
- [ ] Add optional lineage labels (`ternary_label_snapshots`, `packet_id_snapshots`, `shell_id_snapshots`) after the 3D bridge falsification path is working.
