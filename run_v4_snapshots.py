#!/usr/bin/env python3
"""
Generate snapshot triptychs for dashi_cfd_operator_v4 at regular intervals.

Each saved PNG contains (ω true, ω̂ decoded+residual, error) at a target timestep.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dashi_cfd_operator_v4 import (
    simulate_les_trajectory,
    make_grid,
    ProxyConfig,
    encode_proxy,
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
    p.add_argument("--dtype", type=str, choices=["auto", "float32", "float64"], default="auto", help="LES dtype (default auto: float64 when N>1024 else float32)")
    p.add_argument("--backend", type=str, choices=["cpu", "accelerated", "vulkan"], default="cpu", help="dashiCORE backend for ternary ops (if available)")
    p.add_argument("--fft-backend", type=str, choices=["numpy", "vkfft", "vkfft-opencl", "vkfft-vulkan"], default="numpy", help="FFT backend (vkFFT routes FFTs to GPU when available)")
    p.add_argument("--A-npz", type=Path, default=None, dest="A_npz", help="npz containing learned operator under key 'A' (required for --kernel-only)")
    p.add_argument("--no-decode", action="store_true", help="skip decoding/plotting (throughput mode)")
    p.add_argument("--hash-every", type=int, default=0, dest="hash_every", help="hash proxy state every K steps (0=off)")
    p.add_argument("--log-metrics", type=Path, default=None, dest="log_metrics", help="optional JSON file to record timing/hashes")
    return p.parse_args()


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


def main():
    args = parse_args()
    snap_ts = list(range(args.stride, args.steps + 1, args.stride))
    # dtype selection
    if args.dtype == "auto":
        sim_dtype = np.float64 if args.N > 1024 else np.float32
    else:
        sim_dtype = np.float64 if args.dtype == "float64" else np.float32

    # backend setup (optional)
    backend_selected = "cpu"
    fft_executor = None
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
            fft_executor = VkFFTExecutor(handles=handles, fft_backend=args.fft_backend)
            set_fft_executor(fft_executor)
        except Exception as e:
            print(f"[warn] fft-backend {args.fft_backend} unavailable ({e}); using NumPy FFT")
            set_fft_executor(None)

    cfg = ProxyConfig(
        k_cut=args.k_cut,
        resid_mid_cut=args.resid_mid_cut,
        dashi_tau=args.dashi_tau,
        dashi_smooth_k=args.dashi_smooth_k,
    )

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
        traj_stream = simulate_les_trajectory_stream(args.N, args.steps, args.dt, args.nu0, args.Cs, args.seed, dtype=sim_dtype)
        traj_save = [] if args.save_traj is not None else None
        omega_snap = {}

    # Encode trajectory
    import time
    t_enc_start = time.perf_counter()
    if args.kernel_only:
        Z = np.stack([z0])
        t_enc = 0.0
    else:
        Z = []
        mask_low0 = None
        anchor_idx = None
        for t, omega in traj_stream:
            if (not args.no_ground_truth) and (t in snap_ts):
                omega_snap[t] = omega.copy()
            if traj_save is not None:
                traj_save.append(omega.copy())
            z, mask_low, anchor_idx = encode_proxy(omega.astype(np.float64), grid, cfg, anchor_idx=anchor_idx)
            if mask_low0 is None:
                mask_low0 = mask_low
            Z.append(z)
            if args.progress_every and (t % args.progress_every == 0):
                print(f"[encode] t={t}/{args.steps}")
        Z = np.stack(Z, axis=0)
        t_enc = time.perf_counter() - t_enc_start
        if traj_save is not None:
            arr = np.stack(traj_save, axis=0)
            args.save_traj.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_traj, traj=arr)
            if args.progress_every:
                print(f"[save] wrote trajectory to {args.save_traj}")

    # Learn linear operator and rollout
    if args.kernel_only:
        t_learn = 0.0
    else:
        t_learn_start = time.perf_counter()
        A = learn_linear_operator(Z, ridge=args.ridge)
        t_learn = time.perf_counter() - t_learn_start

    t_roll_start = time.perf_counter()
    Z_arr = np.asarray(Z)
    D = Z_arr.shape[1]
    Zhat = np.zeros((2, D), dtype=Z_arr.dtype)
    Zhat[0] = Z_arr[0]
    import hashlib
    hashes = []
    Zhat_snap = {}
    for t in range(args.steps):
        Zhat[1] = Zhat[0] @ A
        if args.progress_every and (t % args.progress_every == 0):
            print(f"[rollout] t={t+1}/{args.steps}")
        if (t + 1) in snap_ts:
            Zhat_snap[t + 1] = Zhat[1].copy()
        if args.hash_every and ((t + 1) % args.hash_every == 0):
            h = hashlib.blake2b(Zhat[1].astype(np.float64).tobytes(), digest_size=16).hexdigest()
            hashes.append({"t": t + 1, "hash": h})
        Zhat[0], Zhat[1] = Zhat[1], Zhat[0]
    t_roll = time.perf_counter() - t_roll_start

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_decode_total = 0.0
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

    if (not args.no_decode) and len(snap_ts) > 0:
        for t in snap_ts:
            t_dec_start = time.perf_counter()
            z_curr = Zhat_snap.get(t, Zhat[0])
            rng = np.random.default_rng(args.residual_seed + 1000003 * t)
            omega_hat, _, _, _ = decode_with_residual(z_curr, grid, cfg, mask_low0, anchor_idx, rng, atoms=None)
            t_decode_total += time.perf_counter() - t_dec_start

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
                out_path = out_dir / f"{args.prefix}_t{t:04d}_decoded.png"
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
                out_path = out_dir / f"{args.prefix}_t{t:04d}_compare.png"

            fig.savefig(out_path, dpi=dpi)
            if graphic_enabled and (args.graphic_every > 0) and (len(snap_ts) > 0):
                if (t // args.stride) % max(args.graphic_every // args.stride, 1) == 0:
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
            plt.close(fig)
            print(f"saved {out_path}")
            if args.progress_every:
                print(f"[snapshot] done t={t}")

    if args.timing:
        per_frame = t_decode_total / max(len(snap_ts), 1)
        print(f"[timing] encode={t_enc:.3f}s  learn={t_learn:.3f}s  rollout={t_roll:.3f}s  decode_total={t_decode_total:.3f}s  decode_per_snap={per_frame:.3f}s  backend={backend_selected} dtype={sim_dtype}")

    if args.log_metrics is not None:
        import json
        metrics = {
            "N": args.N,
            "steps": args.steps,
            "stride": args.stride,
            "dt": args.dt,
            "backend": backend_selected,
            "fft_backend": args.fft_backend,
            "dtype": str(sim_dtype),
            "encode_s": float(t_enc),
            "learn_s": float(t_learn),
            "rollout_s": float(t_roll),
            "decode_total_s": float(t_decode_total),
            "decode_per_snap_s": float(t_decode_total / max(len(snap_ts), 1)),
            "hash_every": args.hash_every,
            "hashes": hashes,
        }
        args.log_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if args.progress_every:
            print(f"[metrics] wrote {args.log_metrics}")


if __name__ == "__main__":
    main()
