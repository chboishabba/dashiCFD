#!/usr/bin/env python3
"""
Validate GPU LES vs CPU LES on matched initial conditions and parameters.

Example:
  python scripts/validate_gpu_truth.py --N 128 --steps 2000 --stride 200 --fft-backend vkfft-vulkan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from dashi_cfd_operator_v4 import (
    energy_from_omega,
    enstrophy,
    make_grid,
    poisson_solve_minus_lap,
    smagorinsky_nu,
    step_rk2,
    velocity_from_psi,
)
from vulkan_les_backend import VulkanLESBackend, init_random_omega


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=64, help="grid size (default: 64)")
    p.add_argument("--steps", type=int, default=200, help="number of steps (default: 200)")
    p.add_argument("--stride", type=int, default=20, help="snapshot stride (default: 20)")
    p.add_argument("--dt", type=float, default=0.01, help="time step (default: 0.01)")
    p.add_argument("--nu0", type=float, default=1e-4, help="base viscosity (default: 1e-4)")
    p.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant (default: 0.17)")
    p.add_argument("--seed", type=int, default=0, help="initial condition seed (default: 0)")
    p.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="GPU FFT backend (default: vkfft-vulkan)")
    p.add_argument("--spectral-truncation", type=str, default="none", help="GPU spectral truncation (default: none)")
    p.add_argument("--trunc-alpha", type=float, default=36.0, help="GPU truncation alpha (default: 36.0)")
    p.add_argument("--trunc-power", type=float, default=8.0, help="GPU truncation power (default: 8.0)")
    p.add_argument("--out", type=Path, default=Path("outputs/validate_gpu_truth"), help="output prefix (default: outputs/validate_gpu_truth)")
    p.add_argument("--progress-every", type=int, default=0, help="print progress every K steps (default: 0)")
    p.add_argument("--timing-detail", action="store_true", help="print timing summary")
    p.add_argument("--rel-l2-max", type=float, default=None, help="optional max rel-L2 threshold for pass/fail")
    p.add_argument("--corr-min", type=float, default=None, help="optional min correlation threshold for pass/fail")
    p.add_argument("--mean-abs-max", type=float, default=None, help="optional max mean-abs error threshold for pass/fail")
    p.add_argument("--plot", action="store_true", help="save a summary plot (outputs/..._plot.png)")
    return p.parse_args()


def _cpu_step(
    omega: np.ndarray,
    *,
    nu0: float,
    Cs: float,
    dt: float,
    KX: np.ndarray,
    KY: np.ndarray,
    K2: np.ndarray,
    dx: float,
) -> np.ndarray:
    psi = poisson_solve_minus_lap(omega, K2)
    u, v = velocity_from_psi(psi, KX, KY)
    nu_t = np.maximum(0.0, smagorinsky_nu(u, v, KX, KY, Cs, dx))
    return step_rk2(omega, nu0 + nu_t, dt, KX, KY, K2)


def main() -> None:
    args = parse_args()
    run_ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_prefix = args.out
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(str(out_prefix) + f"_{run_ts}.json")

    dx, KX, KY, K2 = make_grid(args.N)
    omega0 = init_random_omega(args.N, seed=args.seed).astype(np.float64)

    gpu = VulkanLESBackend(
        args.N,
        dt=args.dt,
        nu0=args.nu0,
        Cs=args.Cs,
        fft_backend=args.fft_backend,
        spectral_truncation=args.spectral_truncation,
        trunc_alpha=args.trunc_alpha,
        trunc_power=args.trunc_power,
    )
    gpu.set_initial_omega(omega0.astype(np.float32))

    snapshot_steps = set(range(0, args.steps + 1, max(int(args.stride), 1)))
    records: List[Dict[str, float]] = []

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    omega_cpu = omega0.copy()
    for step in range(args.steps + 1):
        if step in snapshot_steps:
            omega_gpu = gpu.read_omega().astype(np.float64)
            diff = omega_gpu - omega_cpu
            rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(omega_cpu) + 1e-12))
            mean_abs = float(np.mean(np.abs(diff)))
            corr = float(np.corrcoef(omega_cpu.ravel(), omega_gpu.ravel())[0, 1])
            records.append(
                {
                    "t": float(step),
                    "rel_l2": rel_l2,
                    "mean_abs": mean_abs,
                    "corr": corr,
                    "energy_cpu": float(energy_from_omega(omega_cpu, KX, KY, K2)),
                    "energy_gpu": float(energy_from_omega(omega_gpu, KX, KY, K2)),
                    "enstrophy_cpu": float(enstrophy(omega_cpu)),
                    "enstrophy_gpu": float(enstrophy(omega_gpu)),
                }
            )
        if args.progress_every and (step % args.progress_every == 0):
            elapsed = time.perf_counter() - t0_wall
            done = step
            total = max(args.steps, 1)
            rate = done / max(elapsed, 1e-9) if done > 0 else 0.0
            est_total = elapsed * (total / done) if done > 0 else 0.0
            eta = max(est_total - elapsed, 0.0)
            print(f"[validate] t={step}/{args.steps}  elapsed={elapsed:.1f}s  eta={eta:.1f}s  steps/s={rate:.1f}")
        if step == args.steps:
            break
        omega_cpu = _cpu_step(
            omega_cpu,
            nu0=args.nu0,
            Cs=args.Cs,
            dt=args.dt,
            KX=KX,
            KY=KY,
            K2=K2,
            dx=dx,
        )
        gpu.step()

    summary = {
        "run_ts": run_ts,
        "N": int(args.N),
        "steps": int(args.steps),
        "stride": int(args.stride),
        "dt": float(args.dt),
        "nu0": float(args.nu0),
        "Cs": float(args.Cs),
        "seed": int(args.seed),
        "fft_backend": args.fft_backend,
        "spectral_truncation": args.spectral_truncation,
        "trunc_alpha": float(args.trunc_alpha),
        "trunc_power": float(args.trunc_power),
        "snapshots": len(records),
    }
    payload = {"summary": summary, "records": records}
    rel_l2_vals = [r["rel_l2"] for r in records]
    corr_vals = [r["corr"] for r in records]
    mean_abs_vals = [r["mean_abs"] for r in records]
    if records:
        summary.update(
            {
                "rel_l2_mean": float(np.mean(rel_l2_vals)),
                "rel_l2_max": float(np.max(rel_l2_vals)),
                "corr_mean": float(np.mean(corr_vals)),
                "corr_min": float(np.min(corr_vals)),
                "mean_abs_mean": float(np.mean(mean_abs_vals)),
                "mean_abs_max": float(np.max(mean_abs_vals)),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[validate] wrote {out_path}")
    if args.timing_detail:
        wall = time.perf_counter() - t0_wall
        cpu = time.process_time() - t0_cpu
        print(f"[validate-timing] wall={wall:.3f}s cpu={cpu:.3f}s snapshots={len(records)}")

    thresholds = []
    passed = True
    if args.rel_l2_max is not None and records:
        max_rel = summary.get("rel_l2_max", 0.0)
        if max_rel > args.rel_l2_max:
            passed = False
        thresholds.append(f"rel_l2_max={max_rel:.4g} (limit {args.rel_l2_max})")
    if args.corr_min is not None and records:
        min_corr = summary.get("corr_min", 0.0)
        if min_corr < args.corr_min:
            passed = False
        thresholds.append(f"corr_min={min_corr:.4g} (limit {args.corr_min})")
    if args.mean_abs_max is not None and records:
        max_mean_abs = summary.get("mean_abs_max", 0.0)
        if max_mean_abs > args.mean_abs_max:
            passed = False
        thresholds.append(f"mean_abs_max={max_mean_abs:.4g} (limit {args.mean_abs_max})")

    if thresholds:
        status = "PASS" if passed else "FAIL"
        print(f"[validate] {status} " + " | ".join(thresholds))
        if not passed:
            sys.exit(2)

    if args.plot and records:
        import matplotlib.pyplot as plt

        ts = [r["t"] for r in records]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax = axes[0, 0]
        ax.plot(ts, rel_l2_vals, label="rel_l2")
        ax.set_title("rel_l2 vs t")
        ax.set_xlabel("t")
        ax.set_ylabel("rel_l2")
        ax = axes[0, 1]
        ax.plot(ts, corr_vals, label="corr")
        ax.set_title("corr vs t")
        ax.set_xlabel("t")
        ax.set_ylabel("corr")
        ax = axes[1, 0]
        ax.plot(ts, [r["energy_cpu"] for r in records], label="energy_cpu")
        ax.plot(ts, [r["energy_gpu"] for r in records], label="energy_gpu")
        ax.set_title("energy")
        ax.set_xlabel("t")
        ax.legend()
        ax = axes[1, 1]
        ax.plot(ts, [r["enstrophy_cpu"] for r in records], label="enstrophy_cpu")
        ax.plot(ts, [r["enstrophy_gpu"] for r in records], label="enstrophy_gpu")
        ax.set_title("enstrophy")
        ax.set_xlabel("t")
        ax.legend()
        fig.tight_layout()
        plot_path = Path(str(out_path).replace(".json", "_plot.png"))
        fig.savefig(plot_path, dpi=150)
        print(f"[validate] wrote {plot_path}")


if __name__ == "__main__":
    main()
