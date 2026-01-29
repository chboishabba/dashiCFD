#!/usr/bin/env python3
"""
Generate snapshot triptychs for dashi_cfd_operator_v4 at regular intervals.

Each saved PNG contains (ω true, ω̂ decoded+residual, error) at a target timestep.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

from dashi_cfd_operator_v4 import (
    simulate_les_trajectory,
    make_grid,
    ProxyConfig,
    encode_proxy,
    circular_kmask,
    learn_linear_operator,
    decode_with_residual,
    smagorinsky_nu,
    step_rk2,
    smooth2d,
    fft2,
    ifft2,
    set_fft_executor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=64, help="grid size (default: 64)")
    p.add_argument("--steps", type=int, default=3000, help="total rollout steps (default: 3000)")
    p.add_argument("--stride", type=int, default=300, help="save every STRIDE steps (default: 300)")
    p.add_argument("--dt", type=float, default=0.01, help="time step (default: 0.01)")
    p.add_argument("--nu0", type=float, default=1e-4, help="base viscosity (default: 1e-4)")
    p.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant (default: 0.17)")
    p.add_argument("--seed", type=int, default=0, help="LES seed (default: 0)")
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge regularization (default: 1e-3)")
    p.add_argument("--k-cut", type=float, default=8.0, dest="k_cut", help="proxy low-k cutoff (default: 8)")
    p.add_argument("--resid-mid-cut", type=float, default=12.0, dest="resid_mid_cut", help="residual mid-band cutoff (default: 12)")
    p.add_argument("--tau", type=float, default=0.35, dest="dashi_tau", help="DASHI ternary threshold (default: 0.35)")
    p.add_argument("--smooth-k", type=int, default=11, dest="dashi_smooth_k", help="DASHI smoothing window (default: 11)")
    p.add_argument("--residual-seed", type=int, default=12345, dest="residual_seed", help="residual RNG seed (default: 12345)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"), help="output directory (default: outputs/)")
    p.add_argument("--prefix", type=str, default="v4", help="filename prefix (default: v4)")
    p.add_argument("--dpi", type=int, default=120, help="output DPI (default: 120)")
    p.add_argument("--figsize", type=str, default="12,4", help="figure size inches as W,H (default: 12,4)")
    p.add_argument("--pix-width", type=int, default=None, dest="pix_width", help="explicit pixel width; overrides figsize if set")
    p.add_argument("--pix-height", type=int, default=None, dest="pix_height", help="explicit pixel height; overrides figsize if set")
    p.add_argument("--progress-every", type=int, default=0, dest="progress_every", help="print progress every K steps (0 = silent)")
    p.add_argument(
        "--graphic-every",
        type=int,
        default=0,
        dest="graphic_every",
        help="update a live matplotlib window every K saved snapshots (0 = disabled)",
    )
    p.add_argument("--traj-npz", type=Path, default=None, help="npz with key 'traj' to reuse ground truth instead of recomputing")
    p.add_argument("--no-ground-truth", action="store_true", help="skip ω_true/error panels; requires --traj-npz if encoding is needed")
    p.add_argument("--z0-npz", type=Path, default=None, help="proxy state file with keys z, mask_low, anchor_idx to start kernel-only")
    p.add_argument("--kernel-only", action="store_true", help="skip LES entirely; requires --z0-npz")
    p.add_argument("--save-traj", type=Path, default=None, help="path to save computed trajectory npz (key 'traj')")
    p.add_argument("--timing", action="store_true", help="print timing for encode/learn/rollout/decode loop")
    p.add_argument("--timing-detail", action="store_true", help="print detailed timing breakdown (simulation/encode/decode/plot/video)")
    p.add_argument("--dtype", type=str, choices=["auto", "float32", "float64"], default="auto", help="LES dtype (default auto: float64 when N>1024 else float32)")
    p.add_argument("--backend", type=str, choices=["cpu", "accelerated", "vulkan"], default="cpu", help="dashiCORE backend for ternary ops (if available)")
    p.add_argument("--fft-backend", type=str, choices=["numpy", "vkfft", "vkfft-opencl", "vkfft-vulkan"], default="numpy", help="FFT backend (vkFFT routes FFTs to GPU when available)")
    p.add_argument("--op-backend", type=str, choices=["auto", "cpu", "vulkan"], default="cpu", help="backend for proxy rollout matmul (auto: try vulkan gemv)")
    p.add_argument("--decode-backend", type=str, choices=["cpu", "vulkan"], default="cpu", help="decode backend for snapshots (default: cpu)")
    p.add_argument("--les-backend", type=str, choices=["cpu", "gpu"], default="cpu", help="LES backend for ground truth (default: cpu)")
    p.add_argument("--encode-backend", type=str, choices=["cpu", "gpu"], default="cpu", help="encode backend (default: cpu)")
    p.add_argument("--encode-batch", action="store_true", help="batch GPU encode dispatches (experimental)")
    p.add_argument("--encode-batch-steps", type=int, default=1, help="encode batch size in timesteps (default: 1)")
    p.add_argument("--no-gpu-timing", action="store_true", help="disable GPU-side timing instrumentation")
    p.add_argument("--spectral-truncation", type=str, choices=["none", "exp"], default="none", help="spectral truncation filter for GPU LES (default: none)")
    p.add_argument("--trunc-alpha", type=float, default=36.0, help="exp truncation alpha (default: 36)")
    p.add_argument("--trunc-power", type=float, default=8.0, help="exp truncation power (default: 8)")
    p.add_argument("--permissive-backends", action="store_true", help="allow GPU backend fallback to CPU instead of raising")
    p.add_argument("--A-npz", type=Path, default=None, dest="A_npz", help="npz containing learned operator under key 'A' (required for --kernel-only)")
    p.add_argument("--no-decode", action="store_true", help="skip decoding/plotting (throughput mode)")
    p.add_argument("--hash-every", type=int, default=0, dest="hash_every", help="hash proxy state every K steps (0=off)")
    p.add_argument("--log-metrics", type=Path, default=None, dest="log_metrics", help="optional JSON file to record timing/hashes")
    return p.parse_args()


def _get_vulkan_device_name(handles) -> str | None:
    if handles is None:
        return None
    try:
        import vulkan as vk  # type: ignore
    except Exception:
        return None
    try:
        props = vk.vkGetPhysicalDeviceProperties(handles.physical_device)
        name = getattr(props, "deviceName", None)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "ignore").rstrip("\x00")
        return str(name) if name else None
    except Exception:
        return None


def simulate_les_trajectory_stream(
    N: int,
    steps: int,
    dt: float,
    nu0: float,
    Cs: float,
    seed: int = 0,
    dtype=np.float32,
):
    np.random.seed(seed)
    dx, KX, KY, K2 = make_grid(N)
    omega = smooth2d(np.random.randn(N, N).astype(dtype), 11)
    omega = (omega - omega.mean()) / (omega.std() + 1e-12)
    yield 0, omega.astype(dtype)
    for step in range(steps):
        psi = ifft2(fft2(omega) / K2)
        u = ifft2(1j * KY * fft2(psi))
        v = -ifft2(1j * KX * fft2(psi))
        nu_t = np.maximum(0.0, smagorinsky_nu(u, v, KX, KY, Cs, dx)).astype(dtype)
        omega = step_rk2(omega, nu0 + nu_t, dt, KX, KY, K2).astype(dtype)
        yield step + 1, omega


def simulate_les_trajectory_stream_gpu(
    N: int,
    steps: int,
    dt: float,
    nu0: float,
    Cs: float,
    seed: int = 0,
    fft_backend: str = "vkfft-vulkan",
    spectral_truncation: str = "none",
    trunc_alpha: float = 36.0,
    trunc_power: float = 8.0,
    with_timings: bool = False,
    timing_enabled: bool = True,
):
    from vulkan_les_backend import VulkanLESBackend, init_random_omega

    backend = VulkanLESBackend(
        N,
        dt=dt,
        nu0=nu0,
        Cs=Cs,
        fft_backend=fft_backend,
        spectral_truncation=spectral_truncation,
        trunc_alpha=trunc_alpha,
        trunc_power=trunc_power,
        timing_enabled=timing_enabled,
    )
    omega0 = init_random_omega(N, seed=seed)
    backend.set_initial_omega(omega0)
    if with_timings:
        yield 0, omega0.astype(np.float32, copy=False), backend.get_last_timings()
    else:
        yield 0, omega0.astype(np.float32, copy=False)
    for step in range(steps):
        backend.step()
        omega = backend.read_omega()
        if with_timings:
            yield step + 1, omega, backend.get_last_timings()
        else:
            yield step + 1, omega


def main():
    args = parse_args()
    import time
    t_wall_start = time.perf_counter()
    cpu_wall_start = time.process_time()
    run_ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    snap_ts = list(range(args.stride, args.steps + 1, args.stride))
    # dtype selection
    if args.dtype == "auto":
        sim_dtype = np.float64 if args.N > 1024 else np.float32
    else:
        sim_dtype = np.float64 if args.dtype == "float64" else np.float32

    # backend setup (optional)
    backend_selected = "cpu"
    fft_executor = None
    fft_backend_used = "numpy"
    vulkan_handles = None
    if args.backend != "cpu":
        try:
            import sys as _sys
            core_root = Path(__file__).resolve().parent / "dashiCORE"
            if str(core_root) not in _sys.path:
                _sys.path.insert(0, str(core_root))
            from dashi_core.backend.registry import set_backend, list_backends  # type: ignore
            if args.backend == "accelerated":
                set_backend("accelerated")
                backend_selected = "accelerated"
            elif args.backend == "vulkan":
                try:
                    from gpu_vulkan_backend import probe_and_register_vulkan_backend  # type: ignore
                    backend, icd = probe_and_register_vulkan_backend()
                    if backend is not None:
                        set_backend("vulkan")
                        backend_selected = "vulkan"
                    else:
                        print("[warn] Vulkan backend unavailable; falling back to CPU")
                except Exception as e:  # pragma: no cover
                    print(f"[warn] Vulkan backend init failed ({e}); falling back to CPU")
        except Exception as e:
            print(f"[warn] backend selection unavailable ({e}); using CPU")
            backend_selected = "cpu"

    # FFT backend setup (optional vkFFT)
    if args.fft_backend != "numpy":
        try:
            import sys as _sys
            core_root = Path(__file__).resolve().parent / "dashiCORE"
            if str(core_root) not in _sys.path:
                _sys.path.insert(0, str(core_root))
            from gpu_vkfft_adapter import VkFFTExecutor  # type: ignore
            handles = None
            if args.fft_backend == "vkfft-vulkan":
                try:
                    from gpu_vulkan_dispatcher import create_vulkan_handles  # type: ignore
                    handles = create_vulkan_handles()
                except Exception as e:
                    print(f"[warn] vkFFT Vulkan handles unavailable ({e}); using NumPy FFT")
            fft_executor = VkFFTExecutor(handles=handles, fft_backend=args.fft_backend, timing_enabled=not args.no_gpu_timing)
            set_fft_executor(fft_executor)
            if handles is not None:
                vulkan_handles = handles
        except Exception as e:
            print(f"[warn] fft-backend {args.fft_backend} unavailable ({e}); using NumPy FFT")
            set_fft_executor(None)

    cfg = ProxyConfig(
        k_cut=args.k_cut,
        resid_mid_cut=args.resid_mid_cut,
        dashi_tau=args.dashi_tau,
        dashi_smooth_k=args.dashi_smooth_k,
    )

    encoder = None
    if (not args.kernel_only) and args.encode_backend == "gpu":
        try:
            from vulkan_encode_backend import VulkanEncodeBackend  # type: ignore

            encoder = VulkanEncodeBackend(
                args.N,
                cfg,
                fft_backend=args.fft_backend,
                spectral_truncation=args.spectral_truncation,
                trunc_alpha=args.trunc_alpha,
                trunc_power=args.trunc_power,
                batch_dispatch=args.encode_batch,
                timing_enabled=not args.no_gpu_timing,
            )
        except Exception as e:
            print(f"[warn] GPU encode backend unavailable ({e}); falling back to CPU encode")
            encoder = None

    # Baseline LES or loaded trajectory (stream by default) or kernel-only
    if args.kernel_only:
        if args.z0_npz is None:
            raise SystemExit("--kernel-only requires --z0-npz (z, mask_low, anchor_idx)")
        if args.A_npz is None:
            raise SystemExit("--kernel-only requires --A-npz containing key 'A'")
        data = np.load(args.z0_npz)
        for key in ("z", "mask_low", "anchor_idx"):
            if key not in data:
                raise SystemExit(f"--z0-npz missing key '{key}'")
        z0 = data["z"]
        mask_low0 = data["mask_low"].astype(bool)
        anchor_idx = data["anchor_idx"].astype(np.int64)
        # metadata fallbacks from z0 file if present
        if "N" in data:
            args.N = int(data["N"][()])
        if mask_low0.ndim == 1:
            if mask_low0.size != args.N * args.N:
                raise SystemExit("--z0-npz mask_low length does not match N*N")
            mask_low0 = mask_low0.reshape(args.N, args.N)
        grid = make_grid(args.N)
        Z = np.stack([z0])
        traj_stream = []
        traj_save = None
        omega_snap = {}
        args.steps = max(args.steps, 1)
        snap_ts = list(range(args.stride, args.steps + 1, args.stride))
        A_data = np.load(args.A_npz)
        if "A" not in A_data:
            raise SystemExit("--A-npz must contain key 'A'")
        A = A_data["A"]
    elif args.traj_npz is not None:
        data = np.load(args.traj_npz)
        if "traj" not in data:
            raise SystemExit("npz must contain key 'traj'")
        traj = data["traj"]
        args.steps = min(args.steps, traj.shape[0] - 1)
        dx, KX, KY, K2 = make_grid(args.N)
        grid = (dx, KX, KY, K2)
        traj_stream = ((t, traj[t]) for t in range(args.steps + 1))
        traj_save = None
        omega_snap = {t: traj[t] for t in snap_ts} if not args.no_ground_truth else {}
    else:
        if args.no_ground_truth:
            raise SystemExit("--no-ground-truth requires --traj-npz to supply encoded data")
        dx, KX, KY, K2 = make_grid(args.N)
        grid = (dx, KX, KY, K2)
        if args.les_backend == "gpu":
            sim_with_timings = bool(args.timing or args.timing_detail)
            traj_stream = simulate_les_trajectory_stream_gpu(
                args.N,
                args.steps,
                args.dt,
                args.nu0,
                args.Cs,
                args.seed,
                fft_backend=args.fft_backend,
                spectral_truncation=args.spectral_truncation,
                trunc_alpha=args.trunc_alpha,
                trunc_power=args.trunc_power,
                with_timings=sim_with_timings,
                timing_enabled=not args.no_gpu_timing,
            )
        else:
            traj_stream = simulate_les_trajectory_stream(
                args.N,
                args.steps,
                args.dt,
                args.nu0,
                args.Cs,
                args.seed,
                dtype=sim_dtype,
            )
        traj_save = [] if args.save_traj is not None else None
        omega_snap = {}

    # Encode trajectory
    t_enc_start = time.perf_counter()
    cpu_enc_start = time.process_time()
    t_sim = 0.0
    t_encode_proxy = 0.0
    t_enc_cpu = 0.0
    t_sim_cpu = 0.0
    t_encode_proxy_cpu = 0.0
    t_sim_gpu_wait = 0.0
    t_encode_gpu_wait = 0.0
    t_sim_gpu_time = 0.0
    t_encode_gpu_time = 0.0
    if args.kernel_only:
        Z = np.stack([z0])
        t_enc = 0.0
    else:
        Z = []
        mask_low0 = None
        anchor_idx = None
        traj_iter = iter(traj_stream)
        batch_steps = max(int(args.encode_batch_steps), 1)
        batch_items = []
        t_enc_report_start = time.perf_counter()
        while True:
            t_sim_start = time.perf_counter()
            cpu_sim_start = time.process_time()
            try:
                item = next(traj_iter)
            except StopIteration:
                item = None
            t_sim += time.perf_counter() - t_sim_start
            t_sim_cpu += time.process_time() - cpu_sim_start
            if item is not None:
                sim_timings = None
                if isinstance(item, tuple) and len(item) == 3:
                    t, omega, sim_timings = item
                else:
                    t, omega = item
                if sim_timings and "gpu_wait_ms" in sim_timings:
                    t_sim_gpu_wait += float(sim_timings["gpu_wait_ms"]) / 1000.0
                if sim_timings and "gpu_time_ms" in sim_timings:
                    t_sim_gpu_time += float(sim_timings["gpu_time_ms"]) / 1000.0
                if (not args.no_ground_truth) and (t in snap_ts):
                    omega_snap[t] = omega.copy()
                if traj_save is not None:
                    traj_save.append(omega.copy())
                batch_items.append((t, omega))

            if item is None or len(batch_items) >= batch_steps:
                if not batch_items:
                    break
                t_encode_start = time.perf_counter()
                cpu_encode_start = time.process_time()
                ts = [pair[0] for pair in batch_items]
                omegas = np.stack([pair[1] for pair in batch_items], axis=0)
                if encoder is None:
                    z_list = []
                    for omega in omegas:
                        z, mask_low, anchor_idx = encode_proxy(omega.astype(np.float64), grid, cfg, anchor_idx=anchor_idx)
                        z_list.append(z)
                    z_batch = np.stack(z_list, axis=0)
                else:
                    if mask_low0 is None:
                        mask_low0 = circular_kmask(grid[1], grid[2], cfg.k_cut)
                    z_batch, anchor_idx_gpu = encoder.encode_proxy_batch(
                        omegas,
                        mask_low0,
                        anchor_idx if anchor_idx is not None else None,
                    )
                    mask_low = mask_low0
                    if anchor_idx is None and anchor_idx_gpu is not None:
                        anchor_idx = anchor_idx_gpu.astype(np.int64)
                        if args.progress_every:
                            print("[encode] bootstrapped anchor_idx on GPU for encode")
                t_encode_proxy += time.perf_counter() - t_encode_start
                t_encode_proxy_cpu += time.process_time() - cpu_encode_start
                if encoder is not None:
                    enc_timings = encoder.get_last_timings()
                    if "gpu_wait_ms" in enc_timings:
                        t_encode_gpu_wait += float(enc_timings["gpu_wait_ms"]) / 1000.0
                    if "gpu_time_ms" in enc_timings:
                        t_encode_gpu_time += float(enc_timings["gpu_time_ms"]) / 1000.0
                if mask_low0 is None:
                    mask_low0 = mask_low
                for idx, t in enumerate(ts):
                    z = z_batch[idx]
                    Z.append(z)
                    if args.progress_every and (t % args.progress_every == 0):
                        elapsed = time.perf_counter() - t_enc_report_start
                        done = max(t, 1)
                        total = max(args.steps, 1)
                        rate = done / max(elapsed, 1e-9)
                        est_total = elapsed * (total / done)
                        eta = max(est_total - elapsed, 0.0)
                        print(f"[encode] t={t}/{args.steps}  elapsed={elapsed:.1f}s  est_total={est_total:.1f}s  eta={eta:.1f}s  steps/s={rate:.1f}")
                batch_items = []
        Z = np.stack(Z, axis=0)
        t_enc = time.perf_counter() - t_enc_start
        t_enc_cpu = time.process_time() - cpu_enc_start
        if traj_save is not None:
            arr = np.stack(traj_save, axis=0)
            save_path = args.save_traj
            if save_path.suffix:
                save_path = save_path.with_name(f"{save_path.stem}_{run_ts}{save_path.suffix}")
            else:
                save_path = save_path.with_name(f"{save_path.name}_{run_ts}.npz")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(save_path, traj=arr)
            if args.progress_every:
                print(f"[save] wrote trajectory to {save_path}")

    # Learn linear operator and rollout
    if args.kernel_only:
        t_learn = 0.0
        t_learn_cpu = 0.0
    else:
        t_learn_start = time.perf_counter()
        t_learn_cpu_start = time.process_time()
        A = learn_linear_operator(Z, ridge=args.ridge)
        t_learn = time.perf_counter() - t_learn_start
        t_learn_cpu = time.process_time() - t_learn_cpu_start

    t_roll_start = time.perf_counter()
    t_roll_cpu_start = time.process_time()
    t_roll_gpu_wait = 0.0
    t_roll_gpu_time = 0.0
    Z_arr = np.asarray(Z)
    D = Z_arr.shape[1]

    # Operator backend (CPU or Vulkan GEMV)
    op_backend_req = args.op_backend
    op_backend = "cpu"
    op_device = "cpu"
    vulkan_exec = None
    A_op = A
    Z_dtype = Z_arr.dtype
    perf_flags = []
    decode_backend_req = args.decode_backend
    decode_backend_used = decode_backend_req
    decode_device = "cpu"
    if op_backend_req in ("auto", "vulkan"):
        try:
            from gpu_vulkan_gemv import VulkanGemvExecutor, has_vulkan

            if has_vulkan():
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
                if vulkan_handles is not None:
                    vulkan_exec = VulkanGemvExecutor(D, handles=vulkan_handles, timing_enabled=not args.no_gpu_timing)
                else:
                    vulkan_exec = VulkanGemvExecutor(D, timing_enabled=not args.no_gpu_timing)
                op_backend = "vulkan"
                op_device = "gpu"
                A_op = A.astype(np.float32)
                Z_dtype = np.float32
                vulkan_handles = vulkan_exec.handles
            else:
                if op_backend_req == "vulkan":
                    raise RuntimeError("Vulkan not available")
        except Exception as exc:
            perf_flags.append(f"GPU_ROLLOUT_FALLBACK_CPU ({exc})")
            vulkan_exec = None
            op_backend = "cpu"
            op_device = "cpu"

    Zhat = np.zeros((2, D), dtype=Z_dtype)
    Zhat[0] = Z_arr[0].astype(Z_dtype)
    def to_cpu(x):
        return np.asarray(x)

    import hashlib
    hashes = []
    Zhat_snap = {}
    for t in range(args.steps):
        if vulkan_exec is not None:
            Zhat[1] = vulkan_exec.gemv(A_op, Zhat[0])
            roll_timings = vulkan_exec.get_last_timings()
            if "gpu_wait_ms" in roll_timings:
                t_roll_gpu_wait += float(roll_timings["gpu_wait_ms"]) / 1000.0
            if "gpu_time_ms" in roll_timings:
                t_roll_gpu_time += float(roll_timings["gpu_time_ms"]) / 1000.0
        else:
            Zhat[1] = Zhat[0] @ A_op

        z_host = to_cpu(Zhat[1])
        if args.progress_every and (t % args.progress_every == 0):
            elapsed = time.perf_counter() - t_roll_start
            done = t + 1
            total = max(args.steps, 1)
            rate = done / max(elapsed, 1e-9)
            est_total = elapsed * (total / done)
            eta = max(est_total - elapsed, 0.0)
            print(f"[rollout] t={done}/{args.steps}  elapsed={elapsed:.1f}s  est_total={est_total:.1f}s  eta={eta:.1f}s  steps/s={rate:.1f}")
        if (t + 1) in snap_ts:
            Zhat_snap[t + 1] = z_host.copy()
        if args.hash_every and ((t + 1) % args.hash_every == 0):
            h = hashlib.blake2b(z_host.astype(np.float64).tobytes(), digest_size=16).hexdigest()
            hashes.append({"t": t + 1, "hash": h})
        Zhat[0], Zhat[1] = Zhat[1], Zhat[0]
    t_roll = time.perf_counter() - t_roll_start
    t_roll_cpu = time.process_time() - t_roll_cpu_start

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = f"{args.prefix}_{run_ts}"

    t_decode_total = 0.0
    t_plot_total = 0.0
    t_video_total = 0.0
    t_decode_cpu = 0.0
    t_plot_cpu = 0.0
    t_video_cpu = 0.0
    t_decode_gpu_wait = 0.0
    t_decode_gpu_time = 0.0
    graphic_warned = False
    graphic_enabled = args.graphic_every > 0
    if graphic_enabled:
        import matplotlib

        if matplotlib.get_backend().lower() == "agg":
            graphic_enabled = False
            graphic_warned = True
            print("[warn] --graphic-every ignored because MPLBACKEND=Agg")
        else:
            plt.ion()

    saved_pngs = []
    if (not args.no_decode) and len(snap_ts) > 0:
        for t in snap_ts:
            t_dec_start = time.perf_counter()
            t_dec_cpu_start = time.process_time()
            z_curr = Zhat_snap.get(t, to_cpu(Zhat[0]))
            rng = np.random.default_rng(args.residual_seed + 1000003 * t)
            omega_hat, _, _, _, decode_info = decode_with_residual(
                z_curr,
                grid,
                cfg,
                mask_low0,
                anchor_idx,
                rng,
                atoms=None,
                backend=decode_backend_req,
                allow_fallback=args.permissive_backends,
                fft_backend=args.fft_backend,
                observer="visualize",
                timing_enabled=not args.no_gpu_timing,
            )
            decode_backend_used = decode_info.get("backend_used", decode_backend_used)
            decode_device = decode_info.get("device", decode_device)
            if decode_info.get("flags"):
                perf_flags.extend(decode_info["flags"])
            t_decode_total += time.perf_counter() - t_dec_start
            t_decode_cpu += time.process_time() - t_dec_cpu_start
            timings = decode_info.get("timings") or {}
            if "gpu_wait_ms" in timings:
                t_decode_gpu_wait += float(timings["gpu_wait_ms"]) / 1000.0
            if "gpu_time_ms" in timings:
                t_decode_gpu_time += float(timings["gpu_time_ms"]) / 1000.0

            t_plot_start = time.perf_counter()
            t_plot_cpu_start = time.process_time()
            # Resolve figure size
            if args.pix_width is not None or args.pix_height is not None:
                dpi = args.dpi
                w_in = (args.pix_width or int(args.figsize.split(",")[0])) / dpi
                h_in = (args.pix_height or int(args.figsize.split(",")[1])) / dpi
                figsize = (w_in, h_in)
            else:
                w_str, h_str = args.figsize.split(",")
                figsize = (float(w_str), float(h_str))
                dpi = args.dpi

            if args.no_ground_truth:
                fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
                im = ax.imshow(omega_hat, origin="lower", cmap="viridis")
                ax.set_title(rf"$\hat{{\omega}}$ decoded+residual (t={t})$")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                out_path = out_dir / f"{run_prefix}_t{t:04d}_decoded.png"
            else:
                omega_true = omega_snap[t]
                fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
                titles = [
                    rf"$\omega$ true (t={t})$",
                    rf"$\hat{{\omega}}$ decoded+residual (t={t})$",
                    r"error $\omega-\hat{\omega}$",
                ]
                for ax, data, title in zip(axes, [omega_true, omega_hat, omega_true - omega_hat], titles):
                    im = ax.imshow(data, origin="lower", cmap="viridis")
                    ax.set_title(title)
                    ax.axis("off")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                out_path = out_dir / f"{run_prefix}_t{t:04d}_compare.png"

            fig.savefig(out_path, dpi=dpi)
            saved_pngs.append(out_path)
            t_plot_total += time.perf_counter() - t_plot_start
            t_plot_cpu += time.process_time() - t_plot_cpu_start
            if graphic_enabled and (args.graphic_every > 0) and (len(snap_ts) > 0):
                if (t // args.stride) % max(args.graphic_every // args.stride, 1) == 0:
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
            plt.close(fig)
            print(f"saved {out_path}")
            if args.progress_every:
                print(f"[snapshot] done t={t}")

    if saved_pngs:
        webm_name = f"{run_prefix}_snapshots.webm"
        webm_path = out_dir / webm_name
        list_path = out_dir / f"{run_prefix}_frames.txt"
        try:
            t_video_start = time.perf_counter()
            t_video_cpu_start = time.process_time()
            with open(list_path, "w", encoding="utf-8") as f:
                for path in saved_pngs:
                    f.write(f"file '{path.as_posix()}'\n")
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-r",
                "10",
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuv420p",
                str(webm_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            t_video_total += time.perf_counter() - t_video_start
            t_video_cpu += time.process_time() - t_video_cpu_start
            for path in saved_pngs:
                try:
                    path.unlink()
                except OSError:
                    pass
            list_path.unlink(missing_ok=True)
            print(f"[webm] wrote {webm_path}")
        except FileNotFoundError:
            print("[warn] ffmpeg not found; leaving PNGs in place")
            if list_path.exists():
                list_path.unlink(missing_ok=True)
        except subprocess.CalledProcessError:
            print("[warn] ffmpeg failed; leaving PNGs in place")
            if list_path.exists():
                list_path.unlink(missing_ok=True)

    if fft_executor is not None:
        try:
            plan_backends = {ctx.backend for ctx in fft_executor._plans.values()}
            if plan_backends:
                fft_backend_used = "+".join(sorted(plan_backends))
            else:
                fft_backend_used = "numpy"
        except Exception:
            fft_backend_used = args.fft_backend

    device_name = _get_vulkan_device_name(vulkan_handles)
    print(
        "[summary] "
        f"device={device_name or 'unknown'} "
        f"ternary_backend={backend_selected} "
        f"op_backend={op_backend}/{op_device} "
        f"fft_backend={fft_backend_used} "
        f"decode_backend={decode_backend_used}/{decode_device}"
    )

    if args.timing:
        per_frame = t_decode_total / max(len(snap_ts), 1)
        print(f"[timing] encode={t_enc:.3f}s  learn={t_learn:.3f}s  rollout={t_roll:.3f}s  decode_total={t_decode_total:.3f}s  decode_per_snap={per_frame:.3f}s  backend={backend_selected} dtype={sim_dtype}")
    if args.timing_detail:
        t_wall = time.perf_counter() - t_wall_start
        cpu_wall = time.process_time() - cpu_wall_start
        sim_rate = (args.steps / t_sim) if t_sim > 0 else 0.0
        rollout_rate = (args.steps / t_roll) if t_roll > 0 else 0.0
        decode_rate = (len(snap_ts) / t_decode_total) if t_decode_total > 0 else 0.0
        print(
            "[timing-detail] "
            f"wall={t_wall:.3f}s "
            f"sim={t_sim:.3f}s ({sim_rate:.2f} steps/s) "
            f"encode={t_encode_proxy:.3f}s "
            f"learn={t_learn:.3f}s "
            f"rollout={t_roll:.3f}s ({rollout_rate:.2f} steps/s) "
            f"decode={t_decode_total:.3f}s ({decode_rate:.2f} frames/s) "
            f"plot={t_plot_total:.3f}s "
            f"video={t_video_total:.3f}s"
        )
        def _pct(part: float, whole: float) -> float:
            return (100.0 * part / whole) if whole > 0 else 0.0

        def _perf_line(name: str, wall: float, cpu: float, gpu_wait: float = 0.0, gpu_time: float = 0.0) -> None:
            cpu = min(cpu, wall)
            wait = max(wall - cpu, 0.0)
            gpu_wait = max(min(gpu_wait, wall), 0.0)
            gpu_time = max(min(gpu_time, wall), 0.0)
            gpu_time_str = ""
            if gpu_time > 0.0:
                gpu_time_str = f" gpu={gpu_time:.3f}s ({_pct(gpu_time, wall):.1f}%)"
            print(
                "[perf] "
                f"{name} "
                f"wall={wall:.3f}s "
                f"cpu={cpu:.3f}s ({_pct(cpu, wall):.1f}%) "
                f"wait={wait:.3f}s ({_pct(wait, wall):.1f}%) "
                f"gpu_wait={gpu_wait:.3f}s ({_pct(gpu_wait, wall):.1f}%)"
                f"{gpu_time_str}"
            )

        _perf_line("overall", t_wall, cpu_wall)
        _perf_line("sim", t_sim, t_sim_cpu, t_sim_gpu_wait, t_sim_gpu_time)
        _perf_line("encode", t_encode_proxy, t_encode_proxy_cpu, t_encode_gpu_wait, t_encode_gpu_time)
        _perf_line("learn", t_learn, t_learn_cpu)
        _perf_line("rollout", t_roll, t_roll_cpu, t_roll_gpu_wait, t_roll_gpu_time)
        _perf_line("decode", t_decode_total, t_decode_cpu, t_decode_gpu_wait, t_decode_gpu_time)
        _perf_line("plot", t_plot_total, t_plot_cpu)
        _perf_line("video", t_video_total, t_video_cpu)

    t_wall = time.perf_counter() - t_wall_start

    if args.log_metrics is not None:
        import json
        metrics_path = args.log_metrics
        if metrics_path.suffix:
            metrics_path = metrics_path.with_name(f"{metrics_path.stem}_{run_ts}{metrics_path.suffix}")
        else:
            metrics_path = metrics_path.with_name(f"{metrics_path.name}_{run_ts}.json")
        metrics = {
            "run_ts": run_ts,
            "N": args.N,
            "steps": args.steps,
            "stride": args.stride,
            "dt": args.dt,
            "backend": backend_selected,
            "fft_backend_requested": args.fft_backend,
            "fft_backend_used": fft_backend_used,
            "dtype": str(sim_dtype),
            "op_backend": op_backend,
            "op_device": op_device,
            "decode_backend_requested": decode_backend_req,
            "decode_backend_used": decode_backend_used,
            "decode_device": decode_device,
            "device_name": device_name or "unknown",
            "gpu_hotloop_active": bool(op_backend == "vulkan" and op_device == "gpu"),
            "encode_s": float(t_enc),
            "encode_cpu_s": float(t_enc_cpu),
            "sim_s": float(t_sim),
            "sim_cpu_s": float(t_sim_cpu),
            "sim_gpu_wait_s": float(t_sim_gpu_wait),
            "sim_gpu_time_s": float(t_sim_gpu_time),
            "encode_proxy_s": float(t_encode_proxy),
            "encode_proxy_cpu_s": float(t_encode_proxy_cpu),
            "encode_gpu_wait_s": float(t_encode_gpu_wait),
            "encode_gpu_time_s": float(t_encode_gpu_time),
            "learn_s": float(t_learn),
            "learn_cpu_s": float(t_learn_cpu),
            "rollout_s": float(t_roll),
            "rollout_cpu_s": float(t_roll_cpu),
            "rollout_gpu_wait_s": float(t_roll_gpu_wait),
            "rollout_gpu_time_s": float(t_roll_gpu_time),
            "decode_total_s": float(t_decode_total),
            "decode_cpu_s": float(t_decode_cpu),
            "decode_gpu_wait_s": float(t_decode_gpu_wait),
            "decode_gpu_time_s": float(t_decode_gpu_time),
            "decode_per_snap_s": float(t_decode_total / max(len(snap_ts), 1)),
            "plot_s": float(t_plot_total),
            "plot_cpu_s": float(t_plot_cpu),
            "video_s": float(t_video_total),
            "video_cpu_s": float(t_video_cpu),
            "wall_s": float(t_wall),
            "cpu_wall_s": float(time.process_time() - cpu_wall_start),
            "hash_every": args.hash_every,
            "hashes": hashes,
        }
        if backend_selected != "cpu" and op_device == "cpu":
            metrics["perf_flags"] = ["GPU_ROLLOUT_NOT_IMPLEMENTED_VULKAN"]
        if decode_backend_req == "vulkan" and decode_backend_used != "vulkan":
            metrics.setdefault("perf_flags", []).append("GPU_DECODE_FELLBACK_CPU")
        if perf_flags:
            metrics.setdefault("perf_flags", []).extend(perf_flags)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if args.progress_every:
            print(f"[metrics] wrote {metrics_path}")


if __name__ == "__main__":
    main()
