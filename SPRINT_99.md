Short answer: **no — you do *not* need any new SPIR-V for multi-GPU or networked execution.**
Long answer: you *do* need a **host-side execution model and contracts**, but SPIR-V stays exactly the same.

Below is the clean, formal breakdown so this doesn’t stay fuzzy.

---

## 1️⃣ Why SPIR-V does **not** change for multi-GPU / networked

SPIR-V kernels are already:

* **Pure functions over buffers**
* **Device-agnostic**
* **Stateless across dispatches**
* **Explicit about inputs/outputs**

That means they already satisfy the only requirement for distributed execution:

> *A kernel must not care where its buffers live.*

Multi-GPU and networked execution are **questions of buffer placement, scheduling, and reduction**, not of kernel semantics.

So:

* ❌ no `multi_gpu.spv`
* ❌ no `network_reduce.spv`
* ❌ no “distributed operator” kernels

That would actually be *wrong abstraction*.

---

## 2️⃣ What *does* change: execution topology (host-side)

You need to define **where buffers live and how results are merged**.

Think in three layers:

```
[ SPIR-V kernels ]  ← unchanged
[ Execution topology ]  ← NEW (host-side)
[ Transport / sync ]  ← NEW (host-side)
```

---

## 3️⃣ The only three multi-GPU patterns you need

### 🟦 Pattern A — Data parallel (replica / shard)

**Same kernel, different slices of data**

Example:

* Split `z` into chunks
* Run `operator_step` on GPU0, GPU1, GPU2
* Concatenate results

**SPIR-V impact:** none
**Host responsibility:**

* Partition buffers
* Dispatch identical pipelines on multiple devices

This already works today with your kernels.

---

### 🟩 Pattern B — Model parallel (block operator)

**Split operator A into blocks**

Example:
[
z_{t+1} =
\begin{bmatrix}
A_{00} & A_{01} \
A_{10} & A_{11}
\end{bmatrix}
\begin{bmatrix}
z_0 \
z_1
\end{bmatrix}
]

Each GPU computes one block-row:

* GPU0: `A00 z0 + A01 z1`
* GPU1: `A10 z0 + A11 z1`

**SPIR-V impact:** none
You already have:

* `gemv`
* `spmv`
* `scatter_add_atomic` (for accumulation)

**Host responsibility:**

* Upload block matrices to each GPU
* Exchange `z` shards between devices (or via host)
* Sum partial results

No new kernel needed.

---

### 🟥 Pattern C — Reduction / consensus (networked)

**Each device computes partial invariant, then reduce**

Examples:

* defect sum
* defect max
* norm
* MDL cost

You already have:

* `reduce_sum`
* `reduce_max`
* `defect_reduce`

**SPIR-V stops at “local reduction”.**

**Host does:**

* gather scalars
* final reduce (CPU or GPU0)
* broadcast result if needed

Again: **no new SPIR-V**.

---

## 4️⃣ What *must* be specified (but not in SPIR-V)

You *do* need to define **formal contracts** so the system is deterministic and sane.

### A. Buffer ownership contract

For each buffer:

* single-writer vs multi-writer
* reduction semantics (sum, max, overwrite)
* device of record

This lives in:

* Python
* orchestration layer
* config / metadata

---

### B. Synchronization contract

Define:

* step barriers (after `operator_step`)
* reduction barriers (after defect)
* broadcast points

Again: **host-side only**.

---

### C. Determinism tiering (important)

You already implicitly use this idea.

| Tier   | Allowed ops                         |
| ------ | ----------------------------------- |
| Tier 0 | single GPU, deterministic           |
| Tier 1 | multi-GPU, deterministic reductions |
| Tier 2 | atomic FP reductions (non-bitwise)  |
| Tier 3 | async / eventual consistency        |

No SPIR-V change — just rules.

---

## 5️⃣ What *would* require new SPIR-V? (and why you don’t want it)

The only time you’d need new kernels is if you wanted:

* GPU-direct P2P inside kernel (vendor-specific)
* NCCL-style collectives in shader code
* dynamic device discovery inside SPIR-V

All of these:

* break portability
* break determinism
* break Vulkan’s abstraction model

Correct decision: **do not do this**.

---

## 6️⃣ Minimal additions you *might* add later (optional)

These are **optimizations**, not requirements:

* `reduce_sum_block_local.spv`
  (already effectively exists)
* `reduce_max_block_local.spv`
* `scatter_add_atomic` (already present)

You already have everything you need.

---

## 7️⃣ Canonical answer (for docs / team)

You can literally write this:

> **Multi-GPU and networked execution in DASHI is implemented entirely at the orchestration layer.
> SPIR-V kernels are device-agnostic and require no modification.
> All distributed behavior is expressed via buffer partitioning, replication, and host-side reduction.**

That is the *correct* architectural stance.

---

## Final verdict

* ✅ **No new SPIR-V required**
* ✅ Existing kernels are sufficient
* ❌ Adding “multi-GPU kernels” would be an abstraction error
* 🎯 Focus should be on:

  * buffer residency
  * scheduling
  * reduction semantics
  * metrics & determinism

If you want next, I can:

* formalize a **multi-GPU execution contract** (1 page)
* or sketch a **2-GPU operator_step scheduler** using your current code

Just tell me which.


Yes — your intuition is **correct**, and it’s actually the *right* mental model for what you’re building.

What you’re describing is essentially a **serial / pipelined multi-GPU topology**, not a parallel collective one. And that choice fits DASHI *much* better than NCCL-style fan-out.

Let me make this precise and connect it to DMA, determinism, and why this beats “parallel all-reduce” for your case.

---

## The model you’re describing (formalised)

> **“Each device DMA’s the state produced by n−1”**

That is a **centipede / conveyor / pipeline topology**:

```
GPU₀ ──DMA──▶ GPU₁ ──DMA──▶ GPU₂ ──DMA──▶ … ──DMA──▶ GPUₖ
```

Each device:

1. Receives state `zᵢ`
2. Applies a deterministic operator `Kᵢ`
3. Emits `zᵢ₊₁`

No fan-out. No reduction tree. No consensus until the end.

This is **serial composition**, not parallel aggregation.

---

## Why this is the *right* topology for DASHI

### 1️⃣ DASHI operators compose, they don’t average

Your core step is:

[
z_{t+1} = \mathcal{K}(z_t)
]

Not:
[
z_{t+1} = \frac{1}{N}\sum_i \mathcal{K}_i(z_t)
]

So:

* **Serial composition is natural**
* All-reduce is *semantically wrong* unless you’re computing invariants

Each GPU can own:

* a chart
* a resolution band
* a block of the operator
* a stage of refinement

This matches your **PDA / admissibility / MDL** worldview exactly.

---

### 2️⃣ DMA pipeline preserves determinism

Parallel reductions introduce:

* non-associative FP sums
* order-dependent atomics
* race-dependent rounding

Your centipede pipeline gives you:

| Property             | Pipeline | All-reduce |
| -------------------- | -------- | ---------- |
| Deterministic        | ✅        | ❌ (FP)     |
| Reproducible         | ✅        | ❌          |
| Debuggable           | ✅        | ❌          |
| Semantics-preserving | ✅        | ⚠️         |
| Simple               | ✅        | ❌          |

You get **total order** for free.

---

### 3️⃣ This maps cleanly onto DMA / P2P / PCIe

Whether it’s:

* PCIe
* NVLink
* shared BAR
* host-mediated copy
* network RDMA

…the abstraction is identical:

```text
Device i writes buffer
↓
Device i+1 reads buffer
```

No collective required.

On Vulkan / CUDA / ROCm this becomes:

* copy queue
* transfer queue
* async memcpy
* fence/semaphore chain

Again: **no SPIR-V change needed**.

---

## How this looks architecturally (host-side)

### State ownership

At any time:

* exactly **one device owns the canonical `z`**
* others are either idle or preparing next stage

### Scheduling

Pseudocode:

```python
z = initial_state

for device in devices:
    upload(z, device)
    dispatch(operator_step, device)
    z = dma_read(device)   # or device→device transfer
```

This can be:

* blocking (simplest)
* overlapped (next device uploads while previous computes)

---

## When parallelism *does* make sense (and where)

You still use parallelism, just at the **right level**:

### ✅ Inside a device

* GEMV
* FFT
* reductions
* warps / waves

### ✅ Across devices — but *serially*

* different scales
* different charts
* different operator blocks
* different refinement passes

### ❌ Across devices via reduction

Except for:

* defect checks
* norms
* MDL cost
* diagnostics

Those remain **end-of-pipeline reductions**, not part of the step.

---

## Conceptual match to your formalism (important)

What you’ve reinvented here is basically:

* **Functor composition** across devices
* Not monoidal reduction
* Not data parallel SGD

Each GPU is a **morphism**:
[
z \xrightarrow{K_0} z_1 \xrightarrow{K_1} z_2 \xrightarrow{K_2} \dots
]

That’s why this feels *right* — it’s aligned with:

* PDA closure
* admissible evolution
* MDL contraction
* chart refinement

---

## Practical advantages (non-theoretical)

* Easier debugging: you can snapshot `z` after each GPU
* Easier profiling: cost per stage is isolated
* Easier fallback: drop a GPU, pipeline still works
* Easier scaling: add a stage, not a collective

---

## Final verdict (clear and strong)

✅ **Yes — serial / centipede DMA is the correct model**
✅ It matches DASHI semantics
✅ It preserves determinism
✅ It avoids collective complexity
✅ It requires **zero new SPIR-V**

If you’d like next, I can:

* sketch a **2–4 GPU pipeline scheduler** in Python/Vulkan terms
* define a **formal determinism contract** for serial multi-GPU
* or show how to overlap compute + DMA to keep all GPUs busy

Just say which direction you want to go.


formalize this as a “semantic depth scheduler” spec

give you a worked example (3 GPUs, 1 CPU)

or show how to overlap DMA + compute so weak nodes never stall strong ones

Yes: **each node streams to the next** is the right default topology for your world. Below is a crisp spec + worked example + overlap plan.

---

## 1) “Each node streams to the next” topology

Model the system as a **directed chain** of stages:

[
z^{(0)} \xrightarrow{;S_0;} z^{(1)} \xrightarrow{;S_1;} z^{(2)} \xrightarrow{;S_2;} \dots \xrightarrow{;S_{K-1};} z^{(K)}
]

Where each stage (S_i) is implemented by one device (GPU or CPU), and communicates via **streamed state buffers** (DMA/P2P when possible, host/network otherwise).

Key property: **no fan-out, no all-reduce** in the main loop. Reductions (defect, norms) are *side-band*.

---

## 2) Semantic Depth Scheduler (SDS) — spec

### 2.1 Definitions

**State**

* `State` is a typed payload with metadata:

  * `shape`, `dtype`, `layout`
  * `semantic_depth d ∈ {0..Dmax}`
  * `stage_id` (where produced)
  * `epoch/step` (t)

**Operator / Kernel**
Each operator (op) has an annotation:

* `op.id`
* `op.semantic_depth_in` and `op.semantic_depth_out`
* `op.cost_model` (e.g. FLOPs, bytes, dispatches)
* `op.latency_sensitive ∈ {true,false}`
* `op.determinism_tier ∈ {0,1,2,3}`
* `op.required_features` (fp16/fp32/fp64, atomics, subgroup, etc.)
* `op.io_class ∈ {CORE,PQ,OBSERVER,IO}`

**Device**
Each device has a capability descriptor:

* `dev.id`, `dev.type ∈ {GPU,CPU}`
* `dev.roles ⊆ {coarse, refine, close}` (or numeric `max_semantic_depth`)
* `dev.fp32_gflops_est`, `dev.mem_bw_gbs_est`
* `dev.latency_class ∈ {low,med,high}`
* `dev.supports_features` (atomics float, fp64, etc.)
* `dev.links` to neighbors: `{to_next: p2p|host|net, bandwidth, latency}`

**Link**
A directed link `L(i→i+1)` with:

* `mode ∈ {P2P, HOST_STAGING, RDMA, TCP}`
* `bw`, `latency`
* `max_inflight` (how many buffers can be queued)

---

### 2.2 Scheduling constraints (hard rules)

**R1 — Monotone semantic depth**

* For the main pipeline, semantic depth must be non-decreasing:
  [
  d(z^{(i+1)}) \ge d(z^{(i)})
  ]
* No “backward” sending to weaker stages.

**R2 — Stage role compatibility**

* `op.semantic_depth_out <= dev.max_semantic_depth`
* and `op.required_features ⊆ dev.supports_features`

**R3 — Determinism tier gating**

* If `run.determinism_tier = 0`, then no atomic-fp, no unordered reductions.
* If `tier = 2`, atomic-fp allowed (recorded as perf flag).

**R4 — Residency preference**

* If an op consumes/produces `State` already resident on device, schedule it there unless:

  * it violates R1–R3, or
  * the device is overloaded and the op is marked `migratable=true`.

**R5 — Stream contract**

* Each stage must implement:

  * `enqueue_in(StateRef)`
  * `try_dequeue_out() -> StateRef | None`
* Stages communicate only via these queues (no shared mutable state).

---

### 2.3 Objective (soft rules / scoring)

Scheduler chooses an assignment and buffering such that:

* **Minimize end-to-end step latency**
* **Avoid stalls at high-depth stages**
* **Maximize device utilization**
* **Minimize transfer bytes across slow links**

A simple scoring function for candidate stage assignment:

[
score(op,dev) =
w_1 \cdot \hat{t}*{compute}(op,dev) +
w_2 \cdot \hat{t}*{xfer}(state, link(dev \to next)) +
w_3 \cdot penalty(latency_sensitive \land dev.latency_class \neq low)
]

---

### 2.4 Execution model (what runs each “tick”)

Each stage runs as an event loop:

1. If input queue has a buffer and compute slot free → dispatch kernels
2. When compute completes → enqueue output buffer to next stage link
3. Maintain `inflight` window to keep pipeline full (see overlap plan)

---

## 3) Worked example: 3 GPUs + 1 CPU

### Hardware

* **CPU0** (host): good scalar, poor throughput, low GPU interop overhead
* **GPU0 (weak)**: older card, moderate bandwidth, high latency link
* **GPU1 (mid)**: decent card
* **GPU2 (strong)**: best card, lowest latency to display/IO, should never stall

### Stage roles

* CPU0: `role=coarse-prep` (IO decode, PQ LUT build if needed, packing)
* GPU0: `role=coarse` (heavy/shallow ops)
* GPU1: `role=refine` (mid-band correction, sparse refinement)
* GPU2: `role=close` (admissibility/defect/MDL + final step)

### Choose a pipeline (example)

Let `z` be the “proxy state”.

**Stage 0 (CPU0):** prepare / pack

* Optional PQ encode prep, bitpacking, metadata
* Output: `z0` (maybe reduced precision)

**Stage 1 (GPU0 coarse):**

* `gemv_tiled` on coarse operator block (or block-sparse)
* `fma3` residual mix
* Output: `z1`

**Stage 2 (GPU1 refine):**

* `spmv_csr` sparse correction
* `nonlinear_squash` (optional)
* Output: `z2`

**Stage 3 (GPU2 close):**

* `operator_step` final (or small micro-GEMM)
* `clamp`
* `defect_local` + `reduce_max` every N steps
* Output: `z3` (canonical state)

### Capability matrix (feature flags for this example)

* `COARSE_BLOCK_OP`: `gemv_tiled`, `fma3`
* `REFINE_SPARSE`: `spmv_csr`, optional `nonlinear_*`
* `CLOSE_VALIDATE`: `operator_step`, `clamp`, `defect_*`, `reduce_max`

**Routing rule**

* Anything that increases semantic depth must go **rightward**:

  * coarse (d=1) → refine (d=2) → close (d=3)

### What streams between nodes?

Not always full-resolution. This is crucial.

* CPU0 → GPU0: packed or fp16/fp32 `z`
* GPU0 → GPU1: fp32 `z` (maybe smaller via pruning)
* GPU1 → GPU2: fp32 `z` + side-band diagnostics (optional)

---

## 4) Overlap DMA + compute so weak nodes never stall strong ones

This is the key: **windowed pipelining** with **double/triple buffering**.

### 4.1 The rule

Maintain an **inflight window W** such that each stage always has something to do:

* Stage i computes on buffer `k`
* while simultaneously receiving buffer `k+1`
* and sending buffer `k-1`

This is classic producer/consumer pipelining.

### 4.2 Minimum buffering

* **Double buffering** per link is the minimum (`W=2`)
* **Triple buffering** is safer (`W=3`) if links jitter (network/host staging)

### 4.3 Timeline (W=3)

At steady state:

| Time | CPU0     | GPU0        | Link0    | GPU1        | Link1    | GPU2        |
| ---- | -------- | ----------- | -------- | ----------- | -------- | ----------- |
| t0   | prep k+2 | compute k+1 | xfer k   | compute k   | xfer k-1 | compute k-1 |
| t1   | prep k+3 | compute k+2 | xfer k+1 | compute k+1 | xfer k   | compute k   |
| t2   | prep k+4 | compute k+3 | xfer k+2 | compute k+2 | xfer k+1 | compute k+1 |

GPU2 never waits because GPU1 is always “one ahead”, and GPU0 is “one ahead” of GPU1.

### 4.4 Backpressure

Each link queue has `max_inflight = W`. If a downstream stage is slower, upstream stages will eventually block—**but that is correct**. The goal is to ensure the *strongest* device is the one that *never* blocks.

### 4.5 Practical Vulkan primitives (no SPIR-V changes)

* Use **separate queues** if available:

  * `transfer_queue` for DMA
  * `compute_queue` for kernels
* Use **timeline semaphores** to chain:

  * GPU0 compute completion → GPU0→GPU1 transfer
  * GPU1 transfer completion → GPU1 compute
* Avoid fences per step; fence only when you must observe results.

### 4.6 “Never stall strong node” policy

Make GPU2 the sink. If GPU2 would stall:

* increase W
* reduce payload size upstream (prune / compress)
* shift some work from GPU2 to GPU1 (only if it doesn’t break semantic depth)

---

## 5) Concrete “SDS config” for the example (what you’d actually write)

```yaml
determinism_tier: 1
window: 3

stages:
  - id: cpu0
    device: cpu
    max_semantic_depth: 1
    ops: [pq_pack_bits, pq_lut_build]   # optional
  - id: gpu0
    device: gpu_weak
    max_semantic_depth: 2
    ops: [gemv_tiled, fma3]
  - id: gpu1
    device: gpu_mid
    max_semantic_depth: 3
    ops: [spmv_csr, nonlinear_admissibility_squash]
  - id: gpu2
    device: gpu_strong
    max_semantic_depth: 4
    ops: [operator_step, clamp, defect_local, reduce_max]

links:
  - from: cpu0
    to: gpu0
    mode: HOST_STAGING
  - from: gpu0
    to: gpu1
    mode: P2P
  - from: gpu1
    to: gpu2
    mode: P2P
```

---

## 6) What to implement first (minimal, high impact)

1. Implement stage queues + window W=2 (double buffer)
2. Ensure **no readback** in the main loop
3. Only observe defect/hash every N steps
4. Add compression/pruning between stages if link is slow

---

If you want, I can turn this into:

* a `scheduler.py` skeleton (pure Python) with queues + windowed pipelining, **or**
* a Vulkan-flavoured pseudocode showing timeline semaphore values for each stage.

Which one do you want?


### Is the kernel formalism like a DAG/CA?

It’s *closest to a CA with a DAG-shaped dependency graph*, depending on which level you’re looking at.

* **As a Cellular Automaton (CA):**
  If your state lives on a lattice/sheet and each step applies *local* rules (push/pop/explode/defect/closure) to update the carrier, then yes: it behaves like a CA (or “PDE-like CA”) where time is discrete and update rules are uniform.

* **As a DAG:**
  The *implementation* of one time step is naturally a **DAG of kernels**: each SPIR-V dispatch is a node, buffers are edges, and barriers/semaphores enforce ordering. A whole rollout is then a repeated unrolling of that DAG.

So: **semantically CA**, **operationally DAG**.

---

### How does it compare to Wolfram’s work?

Wolfram’s Physics Project (and related computational universe ideas) typically emphasizes:

* **rewrite rules on hypergraphs** (local transformations)
* **causal invariance** (different update orders yield equivalent results)
* emergence of spacetime/causal structure from those rewrites

Your kernel formalism overlaps in spirit (local rules, emergent structure), but differs in *what’s primary*:

**Shared DNA**

* Local update rules → global structure
* Causality emerges from locality + sequencing
* A “state evolution as computation” worldview

**Key differences**

1. **You explicitly optimize/regularize with MDL + admissibility**
   Wolfram’s framework tends to study the rule space and invariance properties; you’ve baked in *selection pressure* (MDL) and *closure constraints* (PDA/admissibility) as first-class operators.

2. **You treat representations as compressible carriers (support × sign, PQ, etc.)**
   That’s more like “physics + codec” than pure rewriting. Your pipeline has an explicit “compress ↔ evolve ↔ decode” loop.

3. **Your determinism contract is engineering-driven**
   Wolfram leans on causal invariance to tolerate update order variation. You tend to want reproducible results across hardware tiers; if you allow nondeterminism, it’s tiered and intentional.

4. **You’re closer to “operator algebra over fields” than “graph rewriting” (today)**
   You *can* represent rewriting as kernels, but your current working set (GEMV, sparse ops, defect reductions, admissibility projection) reads like a constrained operator calculus. Wolfram starts from rewrite rules as primitives.

A good way to say it internally:

> Wolfram explores *rule-space with causal invariance*.
> DASHI builds a *closed operator algebra with explicit compression + admissibility + MDL selection*, and then schedules it across heterogeneous hardware.

---

## `scheduler.py` skeleton (pure Python) — queues + windowed pipelining

This is a minimal, runnable skeleton that models:

* stages as workers (CPU/GPU abstracted)
* bounded queues (the “window W”)
* overlapped “compute” + “transfer” with separate threads
* no SPIR-V specifics: you plug those in later

```python
# scheduler.py
from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict, List


@dataclass(frozen=True)
class State:
    step: int
    semantic_depth: int
    payload: Any          # e.g. device buffer handle, numpy array, etc.
    meta: Dict[str, Any]  # optional diagnostics


@dataclass(frozen=True)
class OpSpec:
    name: str
    depth_in: int
    depth_out: int
    latency_sensitive: bool = False


@dataclass
class StageSpec:
    name: str
    max_semantic_depth: int
    ops: List[OpSpec]
    # compute_fn transforms input State -> output State (may keep payload on device)
    compute_fn: Callable[[State], State]
    # transfer_fn simulates moving payload to next device (may be no-op for same-device)
    transfer_fn: Callable[[State], State]


class Stage(threading.Thread):
    """
    A pipeline stage:
      in_q  -> compute -> transfer -> out_q
    Queues are bounded to enforce the inflight window W (backpressure).
    """
    def __init__(self, spec: StageSpec, in_q: "queue.Queue[State]", out_q: "queue.Queue[State]",
                 stop_evt: threading.Event, metrics: Dict[str, Any]):
        super().__init__(daemon=True)
        self.spec = spec
        self.in_q = in_q
        self.out_q = out_q
        self.stop_evt = stop_evt
        self.metrics = metrics

    def run(self) -> None:
        name = self.spec.name
        self.metrics.setdefault(name, {"in": 0, "out": 0, "compute_s": 0.0, "xfer_s": 0.0})

        while not self.stop_evt.is_set():
            try:
                st = self.in_q.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            out = self.spec.compute_fn(st)
            t1 = time.perf_counter()
            out2 = self.spec.transfer_fn(out)
            t2 = time.perf_counter()

            self.metrics[name]["in"] += 1
            self.metrics[name]["compute_s"] += (t1 - t0)
            self.metrics[name]["xfer_s"] += (t2 - t1)

            # Block if downstream is full: this is the inflight window backpressure.
            self.out_q.put(out2)

            self.metrics[name]["out"] += 1
            self.in_q.task_done()


class Pipeline:
    def __init__(self, specs: List[StageSpec], window: int):
        assert len(specs) >= 1
        self.specs = specs
        self.window = window
        self.stop_evt = threading.Event()
        self.metrics: Dict[str, Any] = {}

        # Create bounded queues between stages
        self.queues: List["queue.Queue[State]"] = [queue.Queue(maxsize=window) for _ in range(len(specs) + 1)]
        self.stages: List[Stage] = []
        for i, spec in enumerate(specs):
            self.stages.append(Stage(spec, self.queues[i], self.queues[i + 1], self.stop_evt, self.metrics))

    def start(self) -> None:
        for s in self.stages:
            s.start()

    def stop(self) -> None:
        self.stop_evt.set()
        for s in self.stages:
            s.join(timeout=0.5)

    def push(self, st: State) -> None:
        self.queues[0].put(st)

    def try_pop(self, timeout: float = 0.0) -> Optional[State]:
        try:
            return self.queues[-1].get(timeout=timeout)
        except queue.Empty:
            return None

    def join(self) -> None:
        # Wait until everything currently enqueued is processed
        for q in self.queues:
            q.join()


# --- Example compute/transfer fns (replace with Vulkan dispatch / DMA later) ---

def mk_compute(stage_name: str, depth_out: int, compute_ms: float) -> Callable[[State], State]:
    def _fn(st: State) -> State:
        # semantic depth monotone check (you can harden this)
        if st.semantic_depth > depth_out:
            raise RuntimeError(f"{stage_name}: semantic depth would go backwards {st.semantic_depth}->{depth_out}")
        time.sleep(compute_ms / 1000.0)  # simulate work
        return State(step=st.step, semantic_depth=depth_out, payload=st.payload, meta={**st.meta, "last": stage_name})
    return _fn

def mk_xfer(stage_name: str, xfer_ms: float) -> Callable[[State], State]:
    def _fn(st: State) -> State:
        time.sleep(xfer_ms / 1000.0)  # simulate DMA/network
        return st
    return _fn


if __name__ == "__main__":
    # Worked example: CPU -> GPU0(weak) -> GPU1(mid) -> GPU2(strong)
    specs = [
        StageSpec("cpu0_prep", 1,
                  ops=[OpSpec("pack/meta", 0, 1)],
                  compute_fn=mk_compute("cpu0_prep", depth_out=1, compute_ms=1.0),
                  transfer_fn=mk_xfer("cpu0->gpu0", xfer_ms=0.3)),
        StageSpec("gpu0_coarse", 2,
                  ops=[OpSpec("gemv_block+mix", 1, 2)],
                  compute_fn=mk_compute("gpu0_coarse", depth_out=2, compute_ms=0.5),
                  transfer_fn=mk_xfer("gpu0->gpu1", xfer_ms=0.2)),
        StageSpec("gpu1_refine", 3,
                  ops=[OpSpec("spmv+nonlinear", 2, 3)],
                  compute_fn=mk_compute("gpu1_refine", depth_out=3, compute_ms=0.4),
                  transfer_fn=mk_xfer("gpu1->gpu2", xfer_ms=0.15)),
        StageSpec("gpu2_close", 4,
                  ops=[OpSpec("operator_step+clamp+defect", 3, 4, latency_sensitive=True)],
                  compute_fn=mk_compute("gpu2_close", depth_out=4, compute_ms=0.25),
                  transfer_fn=mk_xfer("gpu2->sink", xfer_ms=0.0)),
    ]

    pipe = Pipeline(specs, window=3)
    pipe.start()

    # Feed steps
    N = 50
    t0 = time.perf_counter()
    for k in range(N):
        pipe.push(State(step=k, semantic_depth=0, payload=f"z[{k}]", meta={}))
        # Keep feeding; windowed queues provide backpressure automatically.

    # Drain outputs
    got = 0
    while got < N:
        out = pipe.try_pop(timeout=0.1)
        if out is None:
            continue
        got += 1

    t1 = time.perf_counter()
    pipe.stop()

    print(f"Completed {N} in {t1 - t0:.3f}s")
    print("Metrics:")
    for k, v in pipe.metrics.items():
        print(k, v)
```

**How this demonstrates “weak nodes never stall strong”:**

* Set `window=3` (triple buffering).
* If GPU2 is fastest, it will continuously consume as long as upstream can keep ≥1 item in flight.
* If GPU0 is slow, it will backpressure at `cpu0->gpu0` while GPU2 keeps working until the window drains. Increase window and/or reduce transfer payload size upstream.

---

## Vulkan-flavoured pseudocode: timeline semaphore chaining per stage

Assume:

* each GPU has a `computeQ` and optional `transferQ`
* one **timeline semaphore per link**: `sem_0to1`, `sem_1to2`, `sem_2to3`
* each stage also has a local `sem_done_i` if needed, but you can often reuse link semaphores

We’ll pipeline buffers `b = 0..W-1` (ring buffer), step `t`.

### Notation

* `signal(sem, val)` means queue submission signals timeline semaphore to `val`
* `wait(sem, val)` means queue submission waits until semaphore reaches `val`

### Values

Let `V(t,b)` be a monotonically increasing counter, e.g. `V = t*W + b + 1`.

---

### Stage i: receive → compute → send

#### CPU0 → GPU0 (host staging)

Host copies into GPU0 buffer `z0[b]` and then signals `sem_0to1`:

```text
HostMemcpy(z0[b], payload)
vkQueueSubmit(transferQ_gpu0,
  signal sem_0to1 = V(t,b)
)
```

#### GPU0 compute (wait for input availability)

```text
vkQueueSubmit(computeQ_gpu0,
  wait   sem_0to1 = V(t,b),
  dispatch kernels: [gemv_block, fma3] using z0[b] -> z1[b],
  signal sem_1to2 = V(t,b)
)
```

#### GPU0 → GPU1 transfer (P2P or staged)

If separate transfer queue:

```text
vkQueueSubmit(transferQ_gpu0,
  wait   sem_1to2 = V(t,b),     # compute finished producing z1[b]
  cmdCopyBuffer z1[b] -> gpu1_in[b],
  signal sem_1to2_xfer = V(t,b)
)
```

Or if computeQ does the copy too, you can skip `sem_1to2_xfer` and signal directly to GPU1’s wait semaphore via external semaphores (platform-dependent). Conceptually:

* GPU1 waits on “z1 arrived”.

#### GPU1 compute

```text
vkQueueSubmit(computeQ_gpu1,
  wait   sem_1to2_xfer = V(t,b),
  dispatch kernels: [spmv_csr, nonlinear_squash] gpu1_in[b] -> z2[b],
  signal sem_2to3 = V(t,b)
)
```

#### GPU1 → GPU2 transfer

```text
vkQueueSubmit(transferQ_gpu1,
  wait   sem_2to3 = V(t,b),
  cmdCopyBuffer z2[b] -> gpu2_in[b],
  signal sem_2to3_xfer = V(t,b)
)
```

#### GPU2 compute (the “never stall” sink)

```text
vkQueueSubmit(computeQ_gpu2,
  wait   sem_2to3_xfer = V(t,b),
  dispatch kernels: [operator_step, clamp] (and defect every N),
  signal sem_sink = V(t,b)
)
```

### Host observation (rare)

Only every `hash_every` or `decode_every`:

```text
vkWaitSemaphores(sem_sink, V(t,b))
vkMapMemory / readback (or run hash_state on GPU and read 16 bytes)
```

---

### How overlap happens

With window `W=3`, while GPU2 computes `(t,b=0)`, upstream can be:

* GPU1 transferring `(t,b=1)`
* GPU0 computing `(t,b=2)`
* CPU preparing `(t+1,b=0)`

The only rule is: **never reuse buffer slot b until sem_sink reaches V(t,b)**.

---

If you want, I can adapt both the Python skeleton and the Vulkan pseudocode to your *actual* kernels/flags (e.g., `operator_step + defect every 1000 + hash_state`) and show exactly where the semaphores increment and where to avoid fence-per-step.

