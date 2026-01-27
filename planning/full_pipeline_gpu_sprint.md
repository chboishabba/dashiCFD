# Full GPU Pipeline Sprint Plan

Goal: move LES + encode/decode + proxy rollout to GPU end-to-end, keep data resident on-device, and prove wall-clock wins vs CPU LES.

## Scope (in/out)
- In: LES simulation, spectral ops (FFT + derivatives), encode/decode, proxy GEMV, residual decode, snapshot visualization pipeline.
- Out (for now): visualization GUI, model training beyond linear operator, multi-node scaling.

## Success Criteria
- End-to-end run (same N/steps) is faster than CPU LES baseline with equal or better accuracy.
- GPU residency for major tensors (omega, psi, Z, operator A, residual) with minimal host transfers.
- Deterministic results within tolerated error bounds.

## Workstreams

### 1) Profiling & Baseline (1–2 days)
- Add detailed timing (sim, encode, rollout, decode, plot, video).
- Record baseline LES-only runtime for N=256/512/1024 with fixed seeds.
- Produce a comparison table: CPU LES vs mixed pipeline vs GPU pipeline.

Deliverables:
- Metrics JSONs and a short markdown table in planning/ with timings and settings.

### 2) Data Residency & Memory Plan (2–3 days)
- Define GPU-resident buffers and lifetimes:
  - omega, psi, u, v, KX/KY/K2, Z, A, residuals.
- Minimize host copies; centralize transfers at start/end only.
- Decide on memory modes (host_visible vs device_local) per stage.

Deliverables:
- Memory map diagram + list of host<->device transfers.

### 3) LES on GPU (5–8 days)
- Port spectral operators:
  - FFT/ifft via vkFFT (already available).
  - Derivatives and Smagorinsky viscosity in Vulkan compute.
- Implement RK2 step in GPU compute.
- Keep KX/KY/K2 and intermediate buffers on GPU.

Deliverables:
- Vulkan compute kernels for LES steps.
- Numerical parity tests vs CPU for short rollouts.

### 4) Encode/Decode on GPU (5–8 days)
- Port encode_proxy to GPU:
  - Mask generation, thresholding, smoothing, and ternary ops.
- Port decode_with_residual path to GPU:
  - Residual synthesis and reconstruction kernels.
- Ensure codec operates on GPU buffers directly.

Deliverables:
- GPU encode/decode kernels with parity checks and performance logs.

### 5) Proxy Rollout GPU Hot Loop (2–4 days)
- Move GEMV to GPU (already exists) with device-local storage for A and Z.
- Add batched GEMV / fused steps if possible.
- Verify that op_backend uses the same Vulkan device/queue as FFT/LES.

Deliverables:
- Single-device Vulkan context shared across all GPU stages.

### 6) Pipeline Integration & Async (3–5 days)
- Tie LES -> encode -> learn -> rollout -> decode with minimal sync.
- Use fences/events to overlap transfers and compute where safe.

Deliverables:
- End-to-end GPU pipeline running with minimal host I/O.

### 7) Validation & Benchmarks (2–4 days)
- Accuracy: L2 error, correlation, energy/enstrophy drift vs baseline.
- Speed: wall time for same N/steps, CPU vs GPU.
- Sensitivity: N=256/512/1024, strides, float32/64.

Deliverables:
- Benchmark report + plots in outputs/.

## Risks & Mitigations
- Vulkan binding limits or slow host-visible memory: switch to device_local + staging.
- Precision differences from float32: add float64 path or mixed precision checks.
- GPU not saturated: batch steps or fuse kernels.
- vkFFT Vulkan plan overhead: cache plans and reuse handles.

## Tracking
- Use `--log-metrics` for every benchmark run.
- Store results under planning/benchmarks/ with timestamped JSON and a summary md.

