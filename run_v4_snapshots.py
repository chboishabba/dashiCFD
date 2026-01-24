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
    p.add_argument("--traj-npz", type=Path, default=None, help="npz with key 'traj' to reuse ground truth instead of recomputing")
    p.add_argument("--no-ground-truth", action="store_true", help="skip ω_true/error panels; requires --traj-npz if encoding is needed")
    p.add_argument("--save-traj", type=Path, default=None, help="path to save computed trajectory npz (key 'traj')")
    p.add_argument("--timing", action="store_true", help="print timing for encode/learn/rollout/decode loop")
    return p.parse_args()


def main():
    args = parse_args()
    snap_ts = list(range(args.stride, args.steps + 1, args.stride))
    cfg = ProxyConfig(
        k_cut=args.k_cut,
        resid_mid_cut=args.resid_mid_cut,
        dashi_tau=args.dashi_tau,
        dashi_smooth_k=args.dashi_smooth_k,
    )

    # Baseline LES or loaded trajectory
    if args.traj_npz is not None:
        data = np.load(args.traj_npz)
        if "traj" not in data:
            raise SystemExit("npz must contain key 'traj'")
        traj = data["traj"]
        args.steps = min(args.steps, traj.shape[0] - 1)
        dx, KX, KY, K2 = make_grid(args.N)
        grid = (dx, KX, KY, K2)
    else:
        if args.no_ground_truth:
            raise SystemExit("--no-ground-truth requires --traj-npz to supply encoded data")
        traj, grid, _ = simulate_les_trajectory(
            N=args.N,
            steps=args.steps,
            dt=args.dt,
            nu0=args.nu0,
            Cs=args.Cs,
            seed=args.seed,
        )
        if args.save_traj is not None:
            args.save_traj.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_traj, traj=traj)
            if args.progress_every:
                print(f"[save] wrote trajectory to {args.save_traj}")

    # Encode trajectory
    mask_low0 = None
    anchor_idx = None
    Z = []
    import time
    t_enc_start = time.perf_counter()
    for t in range(args.steps + 1):
        z, mask_low, anchor_idx = encode_proxy(traj[t], grid, cfg, anchor_idx=anchor_idx)
        if mask_low0 is None:
            mask_low0 = mask_low
        Z.append(z)
        if args.progress_every and (t % args.progress_every == 0):
            print(f"[encode] t={t}/{args.steps}")
    Z = np.stack(Z, axis=0)
    t_enc = time.perf_counter() - t_enc_start

    # Learn linear operator and rollout
    t_learn_start = time.perf_counter()
    A = learn_linear_operator(Z, ridge=args.ridge)
    t_learn = time.perf_counter() - t_learn_start

    t_roll_start = time.perf_counter()
    Zhat = np.zeros_like(Z)
    Zhat[0] = Z[0]
    for t in range(args.steps):
        Zhat[t + 1] = Zhat[t] @ A
        if args.progress_every and (t % args.progress_every == 0):
            print(f"[rollout] t={t+1}/{args.steps}")
    t_roll = time.perf_counter() - t_roll_start

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_decode_total = 0.0
    for t in snap_ts:
        t_dec_start = time.perf_counter()
        rng = np.random.default_rng(args.residual_seed + 1000003 * t)
        omega_hat, _, _, _ = decode_with_residual(Zhat[t], grid, cfg, mask_low0, anchor_idx, rng)
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
            omega_true = traj[t]
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
        plt.close(fig)
        print(f"saved {out_path}")
        if args.progress_every:
            print(f"[snapshot] done t={t}")

    if args.timing:
        per_frame = t_decode_total / max(len(snap_ts), 1)
        print(f"[timing] encode={t_enc:.3f}s  learn={t_learn:.3f}s  rollout={t_roll:.3f}s  decode_total={t_decode_total:.3f}s  decode_per_snap={per_frame:.3f}s")


if __name__ == "__main__":
    main()
