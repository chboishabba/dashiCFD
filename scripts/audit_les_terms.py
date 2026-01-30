#!/usr/bin/env python3
"""
Audit GPU vs CPU LES term scaling on a single state.

Example:
  python scripts/audit_les_terms.py --N 64 --fft-backend vkfft-vulkan --out outputs/audit_terms
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from dashi_cfd_operator_v4 import (
    deriv_x,
    deriv_y,
    laplacian,
    make_grid,
    poisson_solve_minus_lap,
    smagorinsky_nu,
    velocity_from_psi,
)
from vulkan_les_backend import VulkanLESBackend, init_random_omega


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=64, help="grid size (default: 64)")
    p.add_argument("--dt", type=float, default=0.01, help="time step (default: 0.01)")
    p.add_argument("--nu0", type=float, default=1e-4, help="base viscosity (default: 1e-4)")
    p.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant (default: 0.17)")
    p.add_argument("--seed", type=int, default=0, help="initial condition seed (default: 0)")
    p.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="GPU FFT backend (default: vkfft-vulkan)")
    p.add_argument("--spectral-truncation", type=str, default="none", help="GPU spectral truncation (default: none)")
    p.add_argument("--trunc-alpha", type=float, default=36.0, help="GPU truncation alpha (default: 36.0)")
    p.add_argument("--trunc-power", type=float, default=8.0, help="GPU truncation power (default: 8.0)")
    p.add_argument("--out", type=Path, default=Path("outputs/audit_terms"), help="output prefix")
    p.add_argument("--save-npz", action="store_true", help="save term arrays to NPZ")
    return p.parse_args()


def _stats(a: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + 1e-12
    return float(num / den)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.size == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _compare(name: str, gpu: np.ndarray, cpu: np.ndarray) -> Dict[str, object]:
    return {
        "rel_l2": _rel_l2(gpu, cpu),
        "corr": _corr(gpu, cpu),
        "gpu": _stats(gpu),
        "cpu": _stats(cpu),
    }


def main() -> None:
    args = parse_args()
    run_ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_prefix = args.out
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    base = Path(str(out_prefix) + f"_{run_ts}")
    json_path = base.with_suffix(".json")
    npz_path = base.with_suffix(".npz")

    dx, KX, KY, K2 = make_grid(args.N)
    omega0 = init_random_omega(args.N, seed=args.seed).astype(np.float64)

    psi = poisson_solve_minus_lap(omega0, K2)
    u, v = velocity_from_psi(psi, KX, KY)
    dwdx = deriv_x(omega0, KX)
    dwdy = deriv_y(omega0, KY)
    adv = u * dwdx + v * dwdy
    lap = laplacian(omega0, K2)
    nu_t = np.maximum(0.0, smagorinsky_nu(u, v, KX, KY, args.Cs, dx))
    rhs = -adv + (args.nu0 + nu_t) * lap

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
    gpu_terms = gpu.debug_rhs_terms()

    report: Dict[str, object] = {
        "run_ts": run_ts,
        "N": int(args.N),
        "dt": float(args.dt),
        "nu0": float(args.nu0),
        "Cs": float(args.Cs),
        "seed": int(args.seed),
        "fft_backend": args.fft_backend,
        "spectral_truncation": args.spectral_truncation,
        "trunc_alpha": float(args.trunc_alpha),
        "trunc_power": float(args.trunc_power),
        "terms": {
            "adv": _compare("adv", gpu_terms["adv"], adv),
            "lap": _compare("lap", gpu_terms["lap"], lap),
            "nu_t": _compare("nu_t", gpu_terms["nu_t"], nu_t),
            "rhs": _compare("rhs", gpu_terms["rhs"], rhs),
            "dwdx": _compare("dwdx", gpu_terms["dwdx"], dwdx),
            "dwdy": _compare("dwdy", gpu_terms["dwdy"], dwdy),
            "ux": _compare("ux", gpu_terms["ux"], u),
            "uy": _compare("uy", gpu_terms["uy"], v),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[audit] wrote {json_path}")
    term_order = ["dwdx", "dwdy", "adv", "lap", "nu_t", "rhs", "ux", "uy"]
    print("term,rel_l2,corr")
    for name in term_order:
        entry = report["terms"].get(name, {})
        rel_l2 = entry.get("rel_l2", 0.0)
        corr = entry.get("corr", 0.0)
        print(f"{name},{rel_l2:.6g},{corr:.6g}")

    if args.save_npz:
        np.savez(
            npz_path,
            omega0=omega0.astype(np.float32),
            adv_cpu=adv.astype(np.float32),
            lap_cpu=lap.astype(np.float32),
            nu_t_cpu=nu_t.astype(np.float32),
            rhs_cpu=rhs.astype(np.float32),
            dwdx_cpu=dwdx.astype(np.float32),
            dwdy_cpu=dwdy.astype(np.float32),
            ux_cpu=u.astype(np.float32),
            uy_cpu=v.astype(np.float32),
            adv_gpu=gpu_terms["adv"],
            lap_gpu=gpu_terms["lap"],
            nu_t_gpu=gpu_terms["nu_t"],
            rhs_gpu=gpu_terms["rhs"],
            dwdx_gpu=gpu_terms["dwdx"],
            dwdy_gpu=gpu_terms["dwdy"],
            ux_gpu=gpu_terms["ux"],
            uy_gpu=gpu_terms["uy"],
        )
        print(f"[audit] wrote {npz_path}")


if __name__ == "__main__":
    main()
