#!/usr/bin/env python3
"""
Generate kernel artifacts (z0.npz and A.npz) from an LES rollout using
dashi_cfd_operator_v4 defaults. The artifacts match the schema expected by
perf_kernel.py and run_v4_snapshots.py in kernel-only mode.

Example:
    python make_kernel_artifacts.py \
      --N 128 --steps 400 --dt 0.01 --seed 0 \
      --out-prefix outputs/kernel_N128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dashi_cfd_operator_v4 import (
    ProxyConfig,
    encode_proxy,
    learn_linear_operator,
    make_grid,
    simulate_les_trajectory,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=128, help="grid size (default: 128)")
    p.add_argument("--steps", type=int, default=400, help="LES steps for learning A (default: 400)")
    p.add_argument("--dt", type=float, default=0.01, help="time step (default: 0.01)")
    p.add_argument("--nu0", type=float, default=1e-4, help="base viscosity (default: 1e-4)")
    p.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant (default: 0.17)")
    p.add_argument("--seed", type=int, default=0, help="LES seed (default: 0)")
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge regularization for A (default: 1e-3)")
    p.add_argument("--k-cut", type=float, default=8.0, dest="k_cut", help="proxy low-k cutoff (default: 8)")
    p.add_argument("--resid-mid-cut", type=float, default=12.0, dest="resid_mid_cut", help="residual mid-band cutoff (default: 12)")
    p.add_argument("--dashi-tau", type=float, default=0.35, dest="dashi_tau", help="DASHI ternary threshold (default: 0.35)")
    p.add_argument("--dashi-smooth-k", type=int, default=11, dest="dashi_smooth_k", help="DASHI smoothing window (default: 11)")
    p.add_argument("--topk-mid", type=int, default=128, dest="topk_mid", help="mid-band coeffs to preserve (default: 128)")
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64", help="storage dtype for z/A (default: float64)")
    p.add_argument("--out-prefix", type=Path, default=Path("outputs/kernel"), help="prefix path for artifacts (default: outputs/kernel)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = ProxyConfig(
        k_cut=args.k_cut,
        resid_mid_cut=args.resid_mid_cut,
        dashi_tau=args.dashi_tau,
        dashi_smooth_k=args.dashi_smooth_k,
        topk_mid=args.topk_mid,
    )

    traj, grid, _ = simulate_les_trajectory(N=args.N, steps=args.steps, dt=args.dt, nu0=args.nu0, Cs=args.Cs, seed=args.seed)

    # Encode trajectory
    Z = []
    anchor_idx = None
    mask_low0 = None
    for frame in traj:
        z, mask_low, anchor_idx = encode_proxy(frame.astype(np.float64), grid, cfg, anchor_idx=anchor_idx)
        if mask_low0 is None:
            mask_low0 = mask_low
        Z.append(z)
    Z = np.stack(Z, axis=0).astype(args.dtype)
    z0 = Z[0]

    # Learn operator
    A = learn_linear_operator(Z, ridge=args.ridge).astype(args.dtype)

    # Metadata schema
    dtype_code = 0 if args.dtype == "float32" else 1
    meta = dict(
        N=np.int64(args.N),
        k_cut=np.float64(args.k_cut),
        resid_mid_cut=np.float64(args.resid_mid_cut),
        topk_mid=np.int64(args.topk_mid),
        dashi_tau=np.float64(args.dashi_tau),
        dashi_smooth_k=np.int64(args.dashi_smooth_k),
        dtype_code=np.int64(dtype_code),
        seed=np.int64(args.seed),
        dt=np.float64(args.dt),
    )

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    base = out_prefix
    z0_path = Path(str(base) + "_z0.npz")
    A_path = Path(str(base) + "_A.npz")

    np.savez(
        z0_path,
        z=z0,
        mask_low=mask_low0.reshape(-1),
        anchor_idx=anchor_idx,
        **meta,
    )
    np.savez(A_path, A=A)

    print(f"[artifacts] wrote {z0_path} and {A_path}")


if __name__ == "__main__":
    main()
