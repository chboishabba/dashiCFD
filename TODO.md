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
- [ ] Run external physical bridge harness against `outputs/truth3d/ns3d_N32_seed0.npz` and record whether `C_K` stays bounded.
- [ ] Add optional lineage labels (`ternary_label_snapshots`, `packet_id_snapshots`, `shell_id_snapshots`) after the 3D bridge falsification path is working.
