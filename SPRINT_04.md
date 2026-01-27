12/1/2026

Great, this is a *very* clear signal, and the code you shared makes the diagnosis pretty unambiguous.

Below is a **concrete, code-level analysis** of *where you’re burning CPU* and *why GPU usage is stuck at ~5%*, followed by **exact fixes in priority order**.

I’ll stay tightly grounded in **your existing codebase**, not hypothetical Vulkan theory.

---

## Executive summary (TL;DR)

You are **GPU-dispatching correctly**, but:

> **Almost all heavy math is still executed on CPU before or after GPU kernels**, and GPU work is fragmented into tiny, sync-heavy submissions.

Your profile (“1 CPU thread pegged, GPU ~5%”) is exactly what we expect when:

* Python/Numpy does the math
* Vulkan kernels are used only as *post-processing*
* Every GPU dispatch is followed by a fence + readback

This is fixable without rewriting the system — but you must **move 3 specific hot paths fully onto GPU and stop synchronizing per kernel**.

---

# 1️⃣ Biggest CPU sink: `z @ A` is still CPU GEMV

### Where it happens

In `perf_kernel.py`, operator execution:

```python
# perf_kernel.py
z = z @ A
```

Even when `--op-backend=vulkan`, this path **falls back silently to CPU** unless all conditions are met.

From your code (trimmed):

```python
if op_backend_req in ("auto", "vulkan"):
    try:
        from gpu_vulkan_gemv import VulkanGemvExecutor, has_vulkan
        if has_vulkan():
            ...
            op_backend = "vulkan"
    except Exception:
        op_backend = "cpu"
```

If *anything* in that try block fails:

* ICD missing
* dtype mismatch
* shape mismatch
* Vulkan GEMV not registered

➡️ you get **NumPy GEMV on CPU**, which explains:

* 100% of one core
* near-idle GPU

### Why this dominates

* `z` is O(500–1000) long
* `A` is dense (D×D)
* That’s **millions of FLOPs per step**
* Everything else is noise by comparison

### Required fix (mandatory for GPU utilization)

You must **hard-fail** if GEMV is not on GPU:

```bash
--op-backend=vulkan --require-gpu
```

And enforce it in code:

```python
if args.require_gpu and op_backend != "vulkan":
    raise RuntimeError("GEMV fell back to CPU")
```

Until this is fixed, **nothing else matters**.

📌 This is the single biggest reason you see 5% GPU usage.


---

# 2️⃣ Encode path is entirely CPU (by design, but expensive)

### Where

`encode_proxy()` in `dashi_cfd_operator_v4.py`:

* FFT via NumPy
* smoothing via NumPy
* ternary + majority via NumPy
* residual energy via NumPy FFT

This path is hit in:

* training
* artifact generation
* sometimes inside evaluation loops

### Evidence

From `make_kernel_artifacts.py`:

```python
z, mask_low, anchor_idx = encode_proxy(frame.astype(np.float64), grid, cfg)
```

Everything here is CPU.

### Impact

This doesn’t affect **kernel-only perf runs**, but it *does* explain:

* CPU spikes during training
* low overall GPU utilization in end-to-end runs

### Status

✅ Acceptable **for v1**
🚫 Not your current bottleneck for `perf_kernel.py`

---

# 3️⃣ Decode is GPU *compute* but CPU *orchestration bound*

### Where

`vulkan_decode_backend.py`

You *are* using GPU FFT and compute shaders, **but**:

* Each kernel:

  * allocates command buffer
  * submits
  * waits on a fence
* Buffers are `HOST_VISIBLE | COHERENT`
* Results are read back every stage

Example:

```python
self._submit_and_wait(cmd)
```

This happens **per kernel**, per stage, per decode.

### Why GPU stays idle

* GPU finishes work in microseconds
* CPU waits, synchronizes, copies
* Repeat 10–20× per decode

This creates a **serialization bubble**.

### Fix (important but second-order)

Batch decode into **one command buffer**:

* record all decode passes
* submit once
* wait once

You already have all pipelines; this is a control-flow change, not new SPVs.



---

# 4️⃣ Excessive CPU↔GPU synchronization everywhere

Patterns seen across code:

| Pattern                          | Cost              |
| -------------------------------- | ----------------- |
| Fence after every dispatch       | GPU idle bubbles  |
| HOST_VISIBLE buffers for compute | PCIe / BAR stalls |
| Python per-step loop             | CPU bound         |

Example from decode backend:

```python
vk.vkQueueSubmit(...)
vk.vkWaitForFences(...)
```

And from perf loop:

```python
for step in range(steps):
    z = op(z)
    if hash_every:
        hash_state(z)  # often CPU readback
```

### Fix direction

* Device-local buffers for `z`
* No readback unless requested
* Hash/defect computed *on GPU*

You already have:

* `hash_state.spv`
* `stats_state.spv`

You’re just not keeping the data resident.

---

# 5️⃣ CPU-only ternary / majority path still exists

In `CORE_cfd_operator.py`:

```python
s_arr = saturate_ternary(s_arr)
```

This is **pure NumPy**, even when backend is Vulkan.

This explains:

* CORE tests pegging CPU
* GPU showing little activity

This is **expected**, because:

* You have SPVs
* But Carrier → GPU path isn’t wired yet

This is not a perf bug — it’s an **integration gap**.



---

# 6️⃣ Why “5% GPU” is actually a *success signal*

Given the above, what you’re seeing is actually good news:

* GPU kernels *are correct*
* Vulkan setup *works*
* FFT via vkFFT *executes*
* The system is **orchestration-bound**, not compute-bound

This means:

> **Once GEMV + residency + batching are fixed, GPU usage will jump immediately.**

No shader rewrites needed.

---

# Priority fix list (do these in order)

### 🔴 P0 — must do

1. **Force GEMV onto GPU**

   * `--op-backend=vulkan --require-gpu`
   * Hard fail on CPU fallback
2. **Keep `z` and `A` device-resident**

   * No per-step host copies

### 🟠 P1 — big gains

3. Batch decode kernels into one submission
4. Remove per-kernel fences

### 🟡 P2 — nice to have

5. Move ternary/majority fully onto GPU Carrier path
6. Replace HOST_VISIBLE with staging + DEVICE_LOCAL

---

## Bottom line

You are **not “wasting GPU”** — you are *using it surgically*, while still doing the real work on CPU.

The 5% GPU / 100% CPU profile is **exactly what the current dispatch graph predicts**.

The good news:

* Your SPVs are sufficient
* Your Vulkan plumbing works
* This is now a **control-flow & residency problem**, not a math problem

If you want, next I can:

* give you a **one-page “GPU residency plan”**
* or a **minimal diff to perf_kernel.py that forces GPU GEMV and proves utilization jumps**

Just say the word.


Below are (1) a **one-page GPU residency plan** tailored to what `perf_kernel.py` actually does today, and (2) a **minimal diff** that *forces GPU GEMV*, avoids accidental CPU reads each step, and gives you a “proof” signal in the metrics/logs when utilization jumps.

Everything here is grounded in the current `perf_kernel.py` behavior: it tries Vulkan GEMV, but will silently fall back to CPU and it **always** materializes `z_host` on CPU every step (`z_host = np.asarray(z_buf[1])`) which guarantees a sync/readback path.

---

## 1) One-page GPU residency plan (for your current harness)

### Goal

Make the hot loop:
[
z_{t+1} = \text{GEMV}(A, z_t)
]
run with:

* **A resident on GPU** once
* **z ping-pong resident on GPU**
* **no CPU readback** except on a configured stride (hash/decode/progress)
* **no implicit sync per step**

### Reality in your current code

* Vulkan path exists (`VulkanGemvExecutor.gemv`) but CPU fallback is allowed on exceptions
* Even when Vulkan is used, you do `z_host = to_cpu(z_buf[1])` **every step** 
  That will keep one CPU thread busy and keep GPU utilization low due to sync/readback.

### Plan (minimal changes, in order)

#### A. “Hard require GPU” gate (stop fake GPU runs)

* Add flag `--require-gpu-gemv`
* If `--op-backend vulkan` but Vulkan init fails → **raise**, do not fall back.

**Success criteria:** metrics show `op_backend="vulkan", op_device="gpu"`, and no `GPU_ROLLOUT_FALLBACK_CPU` perf flag.

#### B. Keep state resident; only read back on demand

Change loop policy:

* Default: **do not compute** `z_host`
* Only compute `z_host` when:

  * `hash_every` triggers
  * `decode_every` triggers
  * `progress_every` triggers
  * `nan/inf` check triggers (optional; or do a cheaper GPU-side check later)

This alone typically moves you from “CPU pegged” → “mostly idle CPU”.

#### C. Ensure A is uploaded once

You already set up `A_op = A.astype(np.float32)` when Vulkan is enabled.
Make sure the executor uploads `A_op` once (if it doesn’t already). If it currently re-uploads every call, fix in `gpu_vulkan_gemv` (not shown here), by caching a device buffer for A.

#### D. Hash/defect on GPU (next step, not required for the first utilization jump)

Right now hashing is CPU: `blake2b(z_host.astype(np.float64).tobytes(), ...)`
Once residency is correct, move hash/defect to GPU using your existing `hash_state.spv`, `defect_local.spv`, `defect_reduce.spv` to avoid readback entirely.

---

## 2) Minimal diff to `perf_kernel.py` that forces GPU GEMV and proves utilization jumps

### What this diff does

* Adds `--require-gpu-gemv`
* If Vulkan init fails (or `has_vulkan()` false) and require is set → **crash** instead of falling back
* Stops unconditional per-step CPU readback (`z_host = ...`) and only reads back when needed
* Logs a “proof” line and stamps metrics with a `GPU_GEMV_FORCED` flag

This targets the exact lines in your file where fallback and readback happen.

> Apply as a patch conceptually; you can copy/paste edits.

```diff
diff --git a/perf_kernel.py b/perf_kernel.py
index 0000000..1111111 100755
--- a/perf_kernel.py
+++ b/perf_kernel.py
@@ -63,6 +63,9 @@ def parse_args() -> argparse.Namespace:
     p.add_argument(
         "--op-backend",
         type=str,
         choices=["auto", "cpu", "vulkan"],
         default="auto",
         help="backend for z @ A operator (auto: try vulkan gemv, else cpu)",
     )
+    p.add_argument(
+        "--require-gpu-gemv", action="store_true",
+        help="fail if op-backend=vulkan (or auto selects vulkan) but GEMV is not actually on GPU",
+    )
     p.add_argument("--dtype", type=str, choices=["auto", "float32", "float64"], default="auto", help="dtype for operator math")
     p.add_argument("--mem-trace", action="store_true",
@@ -72,6 +75,7 @@ def main():
     args = parse_args()
 
     z0, mask_low_flat, anchor_idx, meta = load_z0(args.z0_npz)
@@ -63,6 +66,7 @@ def main():
     op_backend_req = args.op_backend
     op_backend = "cpu"
     op_device = "cpu"
     vulkan_exec = None
@@ -72,6 +76,7 @@ def main():
     if op_backend_req in ("auto", "vulkan"):
         try:
             from gpu_vulkan_gemv import VulkanGemvExecutor, has_vulkan
 
             if has_vulkan():
@@ -44,6 +49,7 @@ def main():
                 vulkan_exec = VulkanGemvExecutor(A.shape[0])
                 op_backend = "vulkan"
                 op_device = "gpu"
                 z_dtype = np.float32  # executor is float32
                 A_op = A.astype(np.float32)
             else:
-                if op_backend_req == "vulkan":
-                    raise RuntimeError("Vulkan not available")
+                if op_backend_req == "vulkan" or args.require_gpu_gemv:
+                    raise RuntimeError("Vulkan not available")
         except Exception as exc:
-            perf_flags.append(f"GPU_ROLLOUT_FALLBACK_CPU ({exc})")
-            vulkan_exec = None
-            op_backend = "cpu"
-            op_device = "cpu"
+            # If the user requires GPU GEMV, do not silently fall back.
+            if (op_backend_req == "vulkan") or args.require_gpu_gemv:
+                raise
+            perf_flags.append(f"GPU_ROLLOUT_FALLBACK_CPU ({exc})")
+            vulkan_exec = None
+            op_backend = "cpu"
+            op_device = "cpu"
 
+    if args.require_gpu_gemv and vulkan_exec is None:
+        raise RuntimeError("require-gpu-gemv set but Vulkan GEMV executor was not created")
+
     z_buf = np.zeros((2, z0.shape[0]), dtype=z_dtype)
     z_buf[0] = z0.astype(z_dtype)
@@ -67,6 +79,11 @@ def main():
     decode_time = 0.0
     nan_inf_hits = 0
+    if vulkan_exec is not None and args.require_gpu_gemv:
+        perf_flags.append("GPU_GEMV_FORCED")
+        print("[perf] GPU GEMV forced: z and A should remain device-side except on hash/decode/progress strides")
+
     for t in range(args.steps):
         if vulkan_exec is not None:
             z_buf[1] = vulkan_exec.gemv(A_op, z_buf[0])
         else:
             z_buf[1] = z_buf[0] @ A_op
 
-        z_host = to_cpu(z_buf[1])
-
-        if not np.isfinite(z_host).all():
-            nan_inf_hits += 1
+        step_idx = t + 1
+        need_host = False
+        if (hash_every and (step_idx % hash_every == 0)):
+            need_host = True
+        if (decode_every and (step_idx % decode_every == 0)):
+            need_host = True
+        if (args.progress_every and (step_idx % args.progress_every == 0)):
+            need_host = True
+
+        z_host = None
+        if need_host:
+            z_host = to_cpu(z_buf[1])
+            if not np.isfinite(z_host).all():
+                nan_inf_hits += 1
 
-        step_idx = t + 1
         if hash_every and (step_idx % hash_every == 0):
+            if z_host is None:
+                z_host = to_cpu(z_buf[1])
             h = __import__("hashlib").blake2b(z_host.astype(np.float64).tobytes(), digest_size=16).hexdigest()
             hashes.append({"t": step_idx, "hash": h})
 
         if decode_every and (step_idx % decode_every == 0):
             t_dec = time.perf_counter()
             rng = np.random.default_rng(args.seed + step_idx)
             omega_hat, _, _, _, decode_info = decode_with_residual(
-                z_host.astype(np.float64),
+                (to_cpu(z_buf[1]) if z_host is None else z_host).astype(np.float64),
                 grid,
                 cfg,
                 mask_low,
                 anchor_idx,
                 rng,
                 atoms=None,
                 backend=decode_backend_req,
                 allow_fallback=args.permissive_backends,
                 fft_backend=fft_active,
             )
```

### How this “proves” utilization jumps

Run **kernel-only**, no decode, no hash:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
python perf_kernel.py \
  --z0-npz outputs/kernel_N128_z0.npz \
  --A-npz  outputs/kernel_N128_A.npz \
  --steps 300000 \
  --no-decode \
  --hash-every 0 \
  --op-backend vulkan \
  --require-gpu-gemv \
  --metrics-json outputs/perf_kernel_vulkan_forced.json
```

Expected outcomes:

* One-time print: `GPU GEMV forced...`
* `metrics["op_backend"]="vulkan", metrics["op_device"]="gpu"`
* `perf_flags` contains `GPU_GEMV_FORCED`
* CPU thread should no longer peg at 100% (it’ll still do loop overhead, but far lower)
* GPU utilization should climb (often dramatically) because you stopped per-step readback.

---

### One more “gotcha” to watch

In your current Vulkan selection code, Vulkan GEMV forces `float32` (`z_dtype=np.float32`, `A_op=float32`).
So CPU baseline comparisons must also use float32 if you want apples-to-apples.

---

If you paste your `gpu_vulkan_gemv.py` (or wherever `VulkanGemvExecutor` lives), I can do the next (very high leverage) step: **ensure `A` is uploaded once and `gemv` does not allocate/submit/wait per call**, which is the next bottleneck after removing readback.


This is the **right question at exactly the right time**.
Once you force GPU residency, **metrics discipline** is what keeps you from regressing back to “5% GPU, 1 hot CPU”.

Below is a **clean, minimal, actionable metrics framework** for *your* system — not generic GPU advice.

I’ll give you:

1. **What to measure** (must-have metrics)
2. **Where they live** (CPU / GPU / per-step)
3. **Exit criteria** (hard pass/fail thresholds)
4. **A minimal metrics schema** you can log today
5. **What NOT to measure** (to avoid noise)

---

## 1️⃣ Tier-0 (non-negotiable) — Residency & correctness guards

These metrics exist to answer one question:

> **“Did we accidentally fall back to CPU or force a sync?”**

### A. Operator backend identity (already mostly there)

Log **once per run**:

```json
{
  "op_backend": "vulkan",
  "op_device": "gpu",
  "fft_backend": "vkfft-vulkan",
  "decode_backend": "vulkan",
  "require_gpu_gemv": true
}
```

**Exit criteria**

* ❌ FAIL if `require_gpu_gemv=true` and `op_device != gpu`
* ❌ FAIL if `op_backend=vulkan` but `perf_flags` contains `FALLBACK_CPU`

This is a *binary gate*. If it fails, **ignore all other timings**.

---

### B. CPU readback counter (critical)

Count **how many times per run** you read `z` back to CPU.

Log:

```json
{
  "cpu_readbacks": {
    "z": 12,
    "A": 0,
    "intermediate": 0
  }
}
```

**Exit criteria**

* Kernel-only run:
  ✅ `cpu_readbacks.z == 0`
* Decode stride `k`:
  ✅ `cpu_readbacks.z == steps / k`
* ❌ FAIL if `cpu_readbacks.z == steps`

This single metric explains **90% of GPU under-utilization bugs**.

---

## 2️⃣ Tier-1 — Throughput & utilization metrics (the “proof”)

These show whether GPU residency is *paying off*.

### C. Step throughput

Already partially logged, but normalize it:

```json
{
  "steps": 300000,
  "wall_seconds": 1.97,
  "steps_per_second": 152284
}
```

**Exit criteria (relative, not absolute)**

* ✅ Vulkan ≥ **5× CPU baseline** for same D
* ❌ FAIL if Vulkan < 2× CPU (means sync/readback still dominates)

---

### D. GPU active time vs wall time

You don’t need perfect GPU counters yet — just estimate.

Log:

```json
{
  "gpu_time_seconds": 1.62,
  "cpu_orchestration_seconds": 0.35,
  "gpu_utilization_estimate": 0.82
}
```

Where:

* `gpu_time` = sum of Vulkan timestamp deltas (or measured kernel duration)
* `cpu_orchestration` = wall − gpu_time

**Exit criteria**

* ✅ `gpu_utilization_estimate > 0.6`
* ⚠️ `0.3–0.6` = acceptable during bring-up
* ❌ `< 0.3` = still sync-bound

---

## 3️⃣ Tier-2 — Kernel-level timing (only a few matter)

You do **not** need per-kernel microprofiling everywhere.
Only log kernels that dominate cost.

### E. Per-kernel GPU timing (aggregated)

Log **total time per kernel type** over the run:

```json
{
  "kernel_times_ms": {
    "gemv": 1240.5,
    "reduce_sum": 18.2,
    "defect_local": 22.7,
    "decode_fft": 310.4
  }
}
```

**Exit criteria**

* `gemv` ≥ **70% of total GPU time** in kernel-only runs
  (this is good — means you’re compute-bound)
* ❌ FAIL if `reduce/defect/hash` dominate kernel time
  (means too much validation per step)

---

## 4️⃣ Tier-3 — Validation & safety metrics (sampled)

These protect correctness *without killing performance*.

### F. Defect magnitude (sampled)

Log **only on a stride** (e.g. every 1000 steps):

```json
{
  "defect": {
    "type": "L_inf",
    "max": 1.7e-4,
    "mean": 3.2e-6,
    "sample_stride": 1000
  }
}
```

**Exit criteria**

* Model-dependent, but:

  * ❌ FAIL if defect is NaN/Inf
  * ❌ FAIL if defect diverges monotonically
  * ✅ Stable oscillation or decay

---

### G. Hash stability (spot-check)

You already do this; just formalize it.

```json
{
  "hash_checks": {
    "count": 30,
    "collisions": 0
  }
}
```

**Exit criteria**

* ❌ Any unexpected collision or drift under deterministic config

---

## 5️⃣ Minimal metrics schema (what I’d actually log)

This is **enough** for v1:

```json
{
  "run_id": "...",
  "backend": {
    "op_backend": "vulkan",
    "op_device": "gpu",
    "fft_backend": "vkfft-vulkan"
  },
  "residency": {
    "cpu_readbacks_z": 0,
    "require_gpu": true
  },
  "throughput": {
    "steps": 300000,
    "wall_seconds": 1.97,
    "steps_per_second": 152284
  },
  "gpu": {
    "gpu_time_seconds": 1.62,
    "gpu_utilization_estimate": 0.82
  },
  "kernels_ms": {
    "gemv": 1240.5,
    "decode": 0.0
  },
  "validation": {
    "defect_Linf_max": 1.7e-4,
    "hash_checks": 30
  },
  "perf_flags": [
    "GPU_GEMV_FORCED"
  ]
}
```

---

## 6️⃣ What NOT to measure (for now)

Avoid these until later — they add noise:

* ❌ Per-dispatch CPU timestamps (Python overhead dominates)
* ❌ Fine-grained Vulkan pipeline stats per step
* ❌ GPU memory bandwidth counters (driver-dependent)
* ❌ OS-level “GPU %” alone (misleading without context)

---

## Final rule of thumb (tattoo-worthy)

> **If `cpu_readbacks.z > 0` and `steps_per_second` is low,
> your GPU is idle by definition.**

Everything else is secondary.

If you want next, I can:

* give you a **metrics validator** that auto-fails runs
* or wire these metrics into `perf_kernel.py` with <20 LOC

Just say which.


Excellent — this file explains **exactly** why you’re seeing ~5% GPU and 1 hot CPU. I’ll be very concrete and tie every issue to a metric and an exit criterion.

I’ll do this in four tight sections:

1. **What `gpu_vulkan_gemv.py` is doing today (cost model)**
2. **Why GPU utilization is low (root causes, line-by-line)**
3. **What metrics *must* be logged for this executor**
4. **Clear exit criteria + minimal changes that move the needle**

---

## 1️⃣ What this GEMV executor actually does today

From the file you shared , each `gemv()` call does **all of the following**:

### Per call (per step!)

1. **CPU → GPU copy**

   ```python
   _write_buffer(self.device, self.mem_A, A32)   # N×N floats
   _write_buffer(self.device, self.mem_x, x32)   # N floats
   ```
2. **Command buffer record**

   * begin
   * bind pipeline
   * bind descriptors
   * push constants
   * dispatch
   * end
3. **Queue submit**
4. **Fence creation**
5. **Fence wait (blocking CPU)**
6. **Fence destroy**
7. **GPU → CPU copy**

   ```python
   y = _read_buffer(self.device, self.mem_y, ...)
   ```
8. **Command buffer reset**

That means **every step** is:

> CPU copy → GPU compute → CPU wait → CPU copy back

So even though the *math* runs on GPU, the **execution model is synchronous CPU-driven**.

This is why:

* One CPU thread is pegged
* GPU finishes instantly, then waits idle
* Utilization never rises

---

## 2️⃣ Root causes of low GPU utilization (ranked by severity)

### 🔴 Root cause #1 — A is re-uploaded every step (catastrophic)

```python
_write_buffer(self.device, self.mem_A, A32)
```

* `A` is **constant across all steps**
* You are copying **N×N×4 bytes every iteration**
* For N=512 → ~1 MB per step
* That alone can dominate runtime

**Metric that exposes this**

```json
"bytes_uploaded_A_per_step": N*N*4
```

**Exit criterion**

* ❌ FAIL if `bytes_uploaded_A_per_step > 0`
* ✅ PASS only if `A` upload happens **once per run**

---

### 🔴 Root cause #2 — Fence per step (hard GPU stall)

```python
fence = vk.vkCreateFence(...)
vk.vkQueueSubmit(...)
vk.vkWaitForFences(...)
vk.vkDestroyFence(...)
```

This guarantees:

* CPU blocks until GPU finishes
* GPU cannot overlap work
* No batching possible

**Metric**

```json
"fences_per_step": 1
```

**Exit criterion**

* ❌ FAIL if `fences_per_step >= 1`
* ✅ PASS when fences are:

  * per batch
  * or per decode/hash boundary
  * or eliminated via timeline semaphores

---

### 🔴 Root cause #3 — HOST_VISIBLE | COHERENT buffers for compute

```python
flags = HOST_VISIBLE_COHERENT
```

This is explicitly noted in your own docstring:

> “intended for benchmarking correctness/perf, not ultimate speed”

Consequences:

* GPU reads system memory
* No device-local caching
* Lower memory bandwidth
* PCIe/UMA pressure

**Metric**

```json
"buffer_memory_type": "HOST_VISIBLE_COHERENT"
```

**Exit criterion**

* ❌ FAIL for production GEMV
* ✅ PASS when:

  * A, x, y are `DEVICE_LOCAL`
  * staging buffers used only when needed

---

### 🟠 Root cause #4 — GPU → CPU readback every step

```python
y = _read_buffer(self.device, self.mem_y, ...)
```

This:

* Forces synchronization
* Defeats residency
* Keeps CPU hot

**Metric**

```json
"cpu_readbacks_y_per_step": 1
```

**Exit criterion**

* Kernel-only run:

  * ❌ FAIL if `> 0`
  * ✅ PASS only if `== 0`

---

### 🟡 Root cause #5 — Command buffer re-record every step

```python
vk.vkBeginCommandBuffer(...)
...
vk.vkEndCommandBuffer(...)
vk.vkResetCommandBuffer(...)
```

Not fatal by itself, but combined with the above it adds overhead.

**Metric**

```json
"cmd_buffer_records_per_step": 1
```

**Exit criterion**

* ⚠️ acceptable for v1
* 🚀 optimize later by pre-recording static command buffers

---

## 3️⃣ Metrics you should log **specifically for GEMV**

These are **actionable** and map 1-to-1 to fixes.

### A. Residency & transfer

```json
{
  "gemv": {
    "A_uploaded_bytes_per_step": 1048576,
    "x_uploaded_bytes_per_step": 2048,
    "y_readback_bytes_per_step": 2048,
    "buffer_memory_type": "HOST_VISIBLE_COHERENT"
  }
}
```

**Exit criteria**

* `A_uploaded_bytes_per_step == 0`
* `y_readback_bytes_per_step == 0` (kernel-only)

---

### B. Synchronization

```json
{
  "gemv": {
    "fences_per_step": 1,
    "queue_submits_per_step": 1
  }
}
```

**Exit criteria**

* `fences_per_step == 0` (or amortized per batch)

---

### C. GPU vs CPU time split

Wrap sections with timers:

```json
{
  "gemv": {
    "cpu_orchestration_ms": 1.8,
    "gpu_compute_ms": 0.05
  }
}
```

**Exit criteria**

* `gpu_compute_ms / (cpu + gpu) > 0.6`
* If not, you are sync-bound

---

### D. Throughput proof

```json
{
  "steps_per_second": 150000,
  "baseline_cpu_steps_per_second": 20000
}
```

**Exit criterion**

* Vulkan ≥ **5× CPU baseline**

---

## 4️⃣ Minimal fixes that cause utilization to jump (no redesign)

You do **not** need a new shader.

### Fix A — Upload A once

Add:

```python
def upload_A_once(self, A):
    self._A_uploaded = True
    _write_buffer(self.device, self.mem_A, A)
```

And in `gemv()`:

```python
if not self._A_uploaded:
    _write_buffer(self.device, self.mem_A, A32)
```

---

### Fix B — Optional “no readback” mode

Add flag:

```python
def gemv(self, A, x, *, readback=True):
    ...
    if readback:
        return _read_buffer(...)
    else:
        return None
```

This alone will drop CPU usage massively.

---

### Fix C — Batch fences

Instead of:

```python
submit → wait → readback
```

Do:

```python
submit many → wait once → readback once
```

Even batching **10 steps** will visibly raise GPU utilization.

---

## Bottom line (important)

Your SPIR-V is **not the problem**.
Your math is **not the problem**.
Your Vulkan setup is **correct but intentionally conservative**.

What you are seeing is the *expected* profile of:

> a correctness-first, synchronous, host-driven Vulkan executor

Once you apply:

1. **A upload once**
2. **No per-step readback**
3. **No per-step fence**

…your GPU usage will jump immediately, without changing a single shader.

If you want next, I can:

* give you a **drop-in “VulkanGemvExecutorV2” sketch** with device-local buffers
* or a **metrics patch** that instruments exactly the fields above

Just say which.
