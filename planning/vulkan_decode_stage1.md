# Vulkan Decode Backend Plan (Stage 1–2)
Date: 2026-01-25

## Objective
Stand up a real `decode_backend=vulkan` path that keeps decode hot-loop work on GPU (no CuPy/Torch), using vkFFT and Vulkan compute shaders, and that reports explicit device/timing/perf_flags so CPU fallbacks are impossible to misread.

## Scope (this milestone)
- Stage 1: GPU low-pass decode
  - Persistent device buffers for `oh_k`, `omega_lp`, scratch.
  - Scatter low-k coefficients to GPU buffer; inverse FFT via vkFFT → `omega_lp` on GPU.
  - Stage timings logged; optional readback only.
- Stage 2: Mask pipeline on GPU
  - GPU smoothing (separable box/triangle) and ternary threshold → int8 mask {-1,0,+1}.
  - Majority iterations on GPU (initially 1 dispatch per iter for clarity).
  - Apply mask/support gating on GPU.

## Success Criteria
- Metrics/JSON show `decode_backend_requested=decode_backend_used="vulkan"` and `decode_device="gpu"`.
- `perf_flags` absent for decode fallbacks; explicit flag if user requests GPU decode and we downgrade.
- Stage timings emitted: scatter_lowk_ms, ifft_lp_ms, mask_ms, combine_ms, readback_ms.
- Low-pass parity: GPU vs CPU decode (no residual) mean abs error ≤1e-6 (float32 acceptable) on N=128 sanity case.
- Hashes stable per-backend; no “GPU lies” (i.e., backend tags match actual device dispatch).

## Constraints / Non-goals
- No CuPy/Torch; Vulkan + vkFFT only.
- Prefer fp32 inside decode for speed; CPU path remains fp64 as reference.
- Energy targeting for residual bands can be approximate in this milestone (log achieved values; no GPU reduction yet).
- Not implementing residual synthesis GPU path in this milestone (Stage 3 later).

## Tasks
1) Instrumentation
   - Add requested/used backend fields for op/decode/fft + `gpu_hotloop_active` to metrics.
   - Perf flags for decode downgrade vs raise (strict/permissive switch).
2) GPU low-pass decode (Stage 1)
   - Allocate persistent device buffers (oh_k, omega_lp, scratch).
   - Scatter low-k kernel → vkFFT inverse → omega_lp kept on GPU.
   - Per-stage timing hooks.
3) GPU mask pipeline (Stage 2)
   - `smooth.comp` (separable), `ternary_threshold.comp`, `majority.comp` (iterative dispatch).
   - Store mask as int8 ternary; optional support buffer for faster multiply.
   - Apply mask combine kernel: omega_hat = omega_lp + float(mask)*omega_resid (resid optional later).
4) CLI + runners
   - Ensure `--decode-backend {cpu,vulkan}` exists in `perf_kernel.py` and `run_v4_snapshots.py` with strict/permissive flag.
   - Log decode stage timings in both runners; ensure headless compatibility.
5) Parity + smoke tests
   - CPU vs Vulkan low-pass decode: compare MAE/energy/enstrophy; hash parity per backend.
   - Kernel-only perf run (no decode) remains unaffected; ensure no regressions.

## Open Questions
- Do we require fp64 decode parity on AMD GPUs, or is fp32 decode acceptable if metrics are logged?
- Strict vs permissive default: should perf harness hard-fail on any decode fallback?
- Where to store compiled `.spv` artifacts (keep single source `dashiCORE/gpu_shaders/*.comp`?). 

## Checkpoints
- C1: Stage 1 complete, metrics prove GPU decode low-pass with timings and no fallback.
- C2: Stage 2 mask kernels validated on GPU vs CPU reference (Hamming distance, MAE).
