#!/usr/bin/env python3
"""
Kernel-only performance + stability harness for the DASHI proxy operator.

Usage (throughput mode):
    python perf_kernel.py \
      --z0-npz outputs/z0.npz \
      --A-npz outputs/A.npz \
      --steps 300000 \
      --no-decode \
      --hash-every 1000 \
      --metrics-json outputs/perf_kernel.json

This treats the learned operator A as the dynamical system and benchmarks
z_{t+1} = z_t @ A with optional decoded sanity snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from dashi_cfd_operator_v4 import (
    ProxyConfig,
    decode_with_residual,
    encode_proxy,
    energy_from_omega,
    enstrophy,
    learn_linear_operator,
    make_grid,
    set_fft_executor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--z0-npz", type=Path, required=True, help="npz with z, mask_low (flat), anchor_idx, metadata")
    p.add_argument("--A-npz", type=Path, default=None, dest="A_npz", help="npz containing operator under key 'A'")
    p.add_argument("--fit-A-from-traj", type=Path, dest="fit_A_traj", default=None, help="npz with key 'traj' to learn A (ignores --A-npz)")
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge regularization for learning A")
    p.add_argument("--steps", type=int, default=100000, help="rollout steps (default: 100k)")
    p.add_argument("--decode-every", type=int, default=0, help="decode every K steps; 0 disables decode")
    p.add_argument("--no-decode", action="store_true", help="force skip decode regardless of stride")
    p.add_argument("--decode-backend", type=str, choices=["cpu", "vulkan"], default="cpu", help="decode backend (vulkan uses vkFFT where available, otherwise falls back)")
    p.add_argument("--hash-every", type=int, default=0, help="hash z every K steps; 0 disables hashing")
    p.add_argument("--metrics-json", type=Path, default=None, help="optional metrics output file")
    p.add_argument("--progress-every", type=int, default=0, help="print progress every K steps (0=silent)")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed for decode")
    p.add_argument("--backend", type=str, choices=["cpu", "accelerated", "vulkan"], default="cpu", help="dashiCORE backend for ternary ops")
    p.add_argument("--fft-backend", type=str, choices=["numpy", "vkfft", "vkfft-opencl", "vkfft-vulkan"], default="numpy", help="FFT backend for decode")
    p.add_argument(
        "--op-backend",
        type=str,
        choices=["auto", "cpu", "vulkan"],
        default="auto",
        help="backend for z @ A operator (auto: try vulkan gemv, else cpu)",
    )
    p.add_argument("--dtype", type=str, choices=["auto", "float32", "float64"], default="auto", help="dtype for operator math")
    p.add_argument("--mem-trace", action="store_true", help="use tracemalloc to record peak Python allocations")
    p.add_argument("--permissive-backends", action="store_true", help="allow GPU backend fallbacks to CPU instead of raising")
    return p.parse_args()


def select_backend(backend: str) -> str:
    """Attempt to set dashiCORE backend; return the active name."""
    active = "cpu"
    if backend == "cpu":
        return active
    try:
        import sys as _sys

        core_root = Path(__file__).resolve().parent / "dashiCORE"
        if str(core_root) not in _sys.path:
            _sys.path.insert(0, str(core_root))
        from dashi_core.backend.registry import set_backend

        if backend == "accelerated":
            set_backend("accelerated")
            active = "accelerated"
        elif backend == "vulkan":
            try:
                from gpu_vulkan_backend import probe_and_register_vulkan_backend  # type: ignore

                backend_obj, _ = probe_and_register_vulkan_backend()
                if backend_obj is not None:
                    set_backend("vulkan")
                    active = "vulkan"
            except Exception:
                active = "cpu"
    except Exception:
        active = "cpu"
    return active


def select_fft_backend(name: str):
    if name == "numpy":
        set_fft_executor(None)
        return "numpy"
    try:
        import sys as _sys

        core_root = Path(__file__).resolve().parent / "dashiCORE"
        if str(core_root) not in _sys.path:
            _sys.path.insert(0, str(core_root))
        from gpu_vkfft_adapter import VkFFTExecutor  # type: ignore

        handles = None
        if name == "vkfft-vulkan":
            try:
                from gpu_vulkan_dispatcher import create_vulkan_handles  # type: ignore

                handles = create_vulkan_handles()
            except Exception:
                handles = None
        fft_exec = VkFFTExecutor(handles=handles, fft_backend=name)
        set_fft_executor(fft_exec)
        return name
    except Exception:
        set_fft_executor(None)
        return "numpy"


def load_z0(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    data = np.load(npz_path)
    for key in ("z", "mask_low", "anchor_idx"):
        if key not in data:
            raise SystemExit(f"{npz_path} missing key '{key}'")
    meta_required = ("N", "k_cut", "resid_mid_cut", "topk_mid", "dashi_tau", "dashi_smooth_k", "dtype_code", "seed", "dt")
    for key in meta_required:
        if key not in data:
            raise SystemExit(f"{npz_path} missing metadata key '{key}'")

    z0 = data["z"].astype(np.float64)
    mask_low_flat = data["mask_low"].astype(bool)
    anchor_idx = data["anchor_idx"].astype(np.int64)
    meta = {k: data[k][()] for k in meta_required if k in data}
    return z0, mask_low_flat, anchor_idx, meta


def load_operator(args: argparse.Namespace, Z: np.ndarray | None, cfg: ProxyConfig, grid):
    if args.fit_A_traj is not None:
        data = np.load(args.fit_A_traj)
        if "traj" not in data:
            raise SystemExit("--fit-A-from-traj npz must contain key 'traj'")
        traj = data["traj"]
        Z_list = []
        anchor_idx = None
        mask_low0 = None
        for frame in traj:
            z, mask_low, anchor_idx = encode_proxy(frame.astype(np.float64), grid, cfg, anchor_idx=anchor_idx)
            if mask_low0 is None:
                mask_low0 = mask_low
            Z_list.append(z)
        Z_enc = np.stack(Z_list, axis=0)
        A = learn_linear_operator(Z_enc, ridge=args.ridge)
        return A
    if args.A_npz is None:
        raise SystemExit("Provide --A-npz or --fit-A-from-traj")
    A_data = np.load(args.A_npz)
    if "A" not in A_data:
        raise SystemExit("--A-npz must contain key 'A'")
    return A_data["A"].astype(np.float64)


def main():
    args = parse_args()

    z0, mask_low_flat, anchor_idx, meta = load_z0(args.z0_npz)
    N = int(meta["N"])
    dtype_code = int(meta["dtype_code"])
    sim_dtype = {0: np.float32, 1: np.float64}.get(dtype_code, np.float64)
    if args.dtype == "float32":
        sim_dtype = np.float32
    elif args.dtype == "float64":
        sim_dtype = np.float64

    cfg = ProxyConfig(
        k_cut=float(meta["k_cut"]),
        resid_mid_cut=float(meta["resid_mid_cut"]),
        dashi_tau=float(meta["dashi_tau"]),
        dashi_smooth_k=int(meta["dashi_smooth_k"]),
        topk_mid=int(meta["topk_mid"]),
    )

    grid = make_grid(N)
    mask_low = mask_low_flat.reshape(N, N)

    backend_active = select_backend(args.backend)
    fft_active = select_fft_backend(args.fft_backend)
    decode_backend_req = args.decode_backend
    decode_backend_used = decode_backend_req
    decode_device = "cpu"

    A = load_operator(args, None, cfg, grid).astype(sim_dtype)
    if z0.shape[0] != A.shape[0]:
        raise SystemExit(f"z0 dim {z0.shape[0]} does not match A shape {A.shape}")

    decode_every = 0 if args.no_decode else max(0, args.decode_every)
    hash_every = max(0, args.hash_every)

    # Optional memory trace
    tracemalloc_started = False
    if args.mem_trace:
        try:
            import tracemalloc

            tracemalloc.start()
            tracemalloc_started = True
        except Exception:
            tracemalloc_started = False

    # Operator backend selection
    op_backend_req = args.op_backend
    op_backend = "cpu"
    op_device = "cpu"
    vulkan_exec = None
    A_op = A
    z_dtype = sim_dtype
    perf_flags = []

    if op_backend_req in ("auto", "vulkan"):
        try:
            from gpu_vulkan_gemv import VulkanGemvExecutor, has_vulkan

            if has_vulkan():
                # Ensure an ICD is selected for headless use.
                if "VK_ICD_FILENAMES" not in os.environ:
                    candidates = [
                        Path("/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"),
                        Path("/usr/share/vulkan/icd.d/amd_icd64.json"),
                        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
                    ] + list(Path("/usr/share/vulkan/icd.d").glob("*.json")) + list(Path("/etc/vulkan/icd.d").glob("*.json"))
                    for icd in candidates:
                        if icd.is_file():
                            os.environ["VK_ICD_FILENAMES"] = str(icd)
                            break
                vulkan_exec = VulkanGemvExecutor(A.shape[0])
                op_backend = "vulkan"
                op_device = "gpu"
                z_dtype = np.float32  # executor is float32
                A_op = A.astype(np.float32)
            else:
                if op_backend_req == "vulkan":
                    raise RuntimeError("Vulkan not available")
        except Exception as exc:
            perf_flags.append(f"GPU_ROLLOUT_FALLBACK_CPU ({exc})")
            vulkan_exec = None
            op_backend = "cpu"
            op_device = "cpu"

    z_buf = np.zeros((2, z0.shape[0]), dtype=z_dtype)
    z_buf[0] = z0.astype(z_dtype)

    def to_cpu(x):
        return np.asarray(x)

    hashes = []
    decode_metrics = []

    t_roll_start = time.perf_counter()
    decode_time = 0.0
    nan_inf_hits = 0
    for t in range(args.steps):
        if vulkan_exec is not None:
            z_buf[1] = vulkan_exec.gemv(A_op, z_buf[0])
        else:
            z_buf[1] = z_buf[0] @ A_op

        z_host = to_cpu(z_buf[1])

        if not np.isfinite(z_host).all():
            nan_inf_hits += 1

        step_idx = t + 1
        if hash_every and (step_idx % hash_every == 0):
            h = __import__("hashlib").blake2b(z_host.astype(np.float64).tobytes(), digest_size=16).hexdigest()
            hashes.append({"t": step_idx, "hash": h})

        if decode_every and (step_idx % decode_every == 0):
            t_dec = time.perf_counter()
            rng = np.random.default_rng(args.seed + step_idx)
            omega_hat, _, _, _, decode_info = decode_with_residual(
                z_host.astype(np.float64),
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
            decode_time += time.perf_counter() - t_dec
            decode_backend_used = decode_info.get("backend_used", decode_backend_used)
            decode_device = decode_info.get("device", decode_device)
            flags = decode_info.get("flags", [])
            if flags:
                perf_flags.extend(flags)
            Eh = energy_from_omega(omega_hat, grid[1], grid[2], grid[3])
            Zh = enstrophy(omega_hat)
            entry = {"t": step_idx, "energy": Eh, "enstrophy": Zh}
            if decode_info.get("timings"):
                entry["timings_ms"] = decode_info["timings"]
            decode_metrics.append(entry)

        if args.progress_every and (step_idx % args.progress_every == 0):
            print(f"[rollout] t={step_idx}/{args.steps}")

        z_buf[0], z_buf[1] = z_buf[1], z_buf[0]

    t_roll = time.perf_counter() - t_roll_start

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024.0

    peak_tracemalloc_mb = None
    if tracemalloc_started:
        import tracemalloc

        current, peak = tracemalloc.get_traced_memory()
        peak_tracemalloc_mb = peak / (1024 * 1024)
        tracemalloc.stop()

    steps = args.steps
    ns_per_step = 1e9 * t_roll / max(steps, 1)
    steps_per_sec = steps / t_roll if t_roll > 0 else float("inf")
    decode_per_snap_ms = 1000 * decode_time / max(len(decode_metrics), 1)

    if backend_active != "cpu" and op_device == "cpu" and op_backend == "cpu":
        perf_flags.append("GPU_ROLLOUT_NOT_IMPLEMENTED_VULKAN")
    if decode_backend_req == "vulkan" and decode_backend_used != "vulkan":
        perf_flags.append("GPU_DECODE_FELLBACK_CPU")

    metrics = {
        "N": N,
        "D": int(z0.shape[0]),
        "steps": steps,
        "dt": float(meta["dt"]),
        "backend": backend_active,
        "fft_backend": fft_active,
        "dtype": str(sim_dtype),
        "rollout_s": float(t_roll),
        "ns_per_step": float(ns_per_step),
        "steps_per_sec": float(steps_per_sec),
        "decode_total_s": float(decode_time),
        "decode_per_snap_ms": float(decode_per_snap_ms),
        "decode_stride": decode_every,
        "hash_every": hash_every,
        "hashes": hashes,
        "decode_metrics": decode_metrics,
        "nan_inf_hits": int(nan_inf_hits),
        "rss_max_mb": float(rss_mb),
        "peak_tracemalloc_mb": float(peak_tracemalloc_mb) if peak_tracemalloc_mb is not None else None,
        "op_backend_requested": args.op_backend,
        "op_backend_used": op_backend,
        "op_device": op_device,
        "decode_backend_requested": decode_backend_req,
        "decode_backend_used": decode_backend_used,
        "decode_device": decode_device,
        "fft_backend_requested": args.fft_backend,
        "fft_backend_used": fft_active,
        "gpu_hotloop_active": bool(op_backend == "vulkan" and op_device == "gpu"),
        "perf_flags": perf_flags,
    }

    print(f"[perf] rollout={t_roll:.3f}s  ns/step={ns_per_step:.1f}  steps/s={steps_per_sec:.1f}  backend={backend_active} fft={fft_active} dtype={sim_dtype}")
    if perf_flags:
        print(f"[perf warning] {'; '.join(perf_flags)} (see dashiCORE/VK_SPV.md)")
    if decode_every:
        print(f"[decode] total={decode_time:.3f}s  per_snap={decode_per_snap_ms:.3f} ms  snaps={len(decode_metrics)}")
    if hash_every:
        print(f"[hash] {len(hashes)} hashes (every {hash_every} steps)")
    if nan_inf_hits:
        print(f"[warn] encountered non-finite values {nan_inf_hits} times during rollout")

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[metrics] wrote {args.metrics_json}")


if __name__ == "__main__":
    main()
