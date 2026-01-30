#!/usr/bin/env python3
"""
Generate a ground-truth LES trajectory artifact (snapshots + metrics).

Example:
  python scripts/make_truth.py --backend cpu --N 128 --steps 2000 --stride 200 --out outputs/truth
  python scripts/make_truth.py --backend gpu --N 256 --steps 10000 --stride 500 --out outputs/truth --fft-backend vkfft-vulkan
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import time

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
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu", help="truth backend (default: cpu)")
    p.add_argument("--truth-tag", type=str, default=None, help="optional tag for output name (default: backend)")
    p.add_argument("--N", type=int, default=128, help="grid size (default: 128)")
    p.add_argument("--steps", type=int, default=2000, help="number of LES steps (default: 2000)")
    p.add_argument("--stride", type=int, default=200, help="snapshot stride (default: 200)")
    p.add_argument("--dt", type=float, default=0.01, help="time step (default: 0.01)")
    p.add_argument("--nu0", type=float, default=1e-4, help="base viscosity (default: 1e-4)")
    p.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant (default: 0.17)")
    p.add_argument("--seed", type=int, default=0, help="initial condition seed (default: 0)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="storage dtype for snapshots")
    p.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="GPU FFT backend (default: vkfft-vulkan)")
    p.add_argument("--spectral-truncation", type=str, default="none", help="GPU spectral truncation (default: none)")
    p.add_argument("--trunc-alpha", type=float, default=36.0, help="GPU truncation alpha (default: 36.0)")
    p.add_argument("--trunc-power", type=float, default=8.0, help="GPU truncation power (default: 8.0)")
    p.add_argument("--out", type=Path, default=Path("outputs/truth"), help="output prefix (default: outputs/truth)")
    p.add_argument("--progress-every", type=int, default=0, help="print progress every K steps (default: 0)")
    p.add_argument("--timing-detail", action="store_true", help="print timing summary")
    p.add_argument("--meta-only", action="store_true", help="only write metadata JSON, no NPZ")
    p.add_argument("--update-manifest", action="store_true", help="update outputs/truth_manifest.json")
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
    truth_tag = args.truth_tag if args.truth_tag is not None else args.backend
    out_prefix = args.out
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    base = Path(str(out_prefix) + f"_{truth_tag}_{run_ts}")
    npz_path = base.with_suffix(".npz")
    meta_path = base.with_suffix(".json")

    snapshot_steps = list(range(0, args.steps + 1, max(int(args.stride), 1)))
    snapshot_set = set(snapshot_steps)

    dx, KX, KY, K2 = make_grid(args.N)
    omega0 = init_random_omega(args.N, seed=args.seed)

    snapshots: List[np.ndarray] = []
    energies: List[float] = []
    enstrophies: List[float] = []
    max_abs: List[float] = []
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    if args.backend == "cpu":
        omega = omega0.astype(np.float64, copy=True)
        for step in range(args.steps + 1):
            if step in snapshot_set:
                snapshots.append(omega.astype(np.float32 if args.dtype == "float32" else np.float64, copy=True))
                energies.append(float(energy_from_omega(omega, KX, KY, K2)))
                enstrophies.append(float(enstrophy(omega)))
                max_abs.append(float(np.max(np.abs(omega))))
            if args.progress_every and (step % args.progress_every == 0):
                elapsed = time.perf_counter() - t0_wall
                done = step
                total = max(args.steps, 1)
                rate = done / max(elapsed, 1e-9) if done > 0 else 0.0
                est_total = elapsed * (total / done) if done > 0 else 0.0
                eta = max(est_total - elapsed, 0.0)
                print(f"[truth] t={step}/{args.steps}  elapsed={elapsed:.1f}s  eta={eta:.1f}s  steps/s={rate:.1f}")
            if step == args.steps:
                break
            omega = _cpu_step(
                omega,
                nu0=args.nu0,
                Cs=args.Cs,
                dt=args.dt,
                KX=KX,
                KY=KY,
                K2=K2,
                dx=dx,
            )
    else:
        backend = VulkanLESBackend(
            args.N,
            dt=args.dt,
            nu0=args.nu0,
            Cs=args.Cs,
            fft_backend=args.fft_backend,
            spectral_truncation=args.spectral_truncation,
            trunc_alpha=args.trunc_alpha,
            trunc_power=args.trunc_power,
        )
        backend.set_initial_omega(omega0)
        for step in range(args.steps + 1):
            if step in snapshot_set:
                omega = backend.read_omega()
                snapshots.append(omega.astype(np.float32 if args.dtype == "float32" else np.float64, copy=False))
                energies.append(float(energy_from_omega(omega.astype(np.float64), KX, KY, K2)))
                enstrophies.append(float(enstrophy(omega.astype(np.float64))))
                max_abs.append(float(np.max(np.abs(omega))))
            if args.progress_every and (step % args.progress_every == 0):
                elapsed = time.perf_counter() - t0_wall
                done = step
                total = max(args.steps, 1)
                rate = done / max(elapsed, 1e-9) if done > 0 else 0.0
                est_total = elapsed * (total / done) if done > 0 else 0.0
                eta = max(est_total - elapsed, 0.0)
                print(f"[truth] t={step}/{args.steps}  elapsed={elapsed:.1f}s  eta={eta:.1f}s  steps/s={rate:.1f}")
            if step == args.steps:
                break
            backend.step()

    meta: Dict[str, object] = {
        "run_ts": run_ts,
        "backend": args.backend,
        "truth_tag": truth_tag,
        "N": int(args.N),
        "steps": int(args.steps),
        "stride": int(args.stride),
        "dt": float(args.dt),
        "nu0": float(args.nu0),
        "Cs": float(args.Cs),
        "seed": int(args.seed),
        "dtype": args.dtype,
        "fft_backend": args.fft_backend,
        "spectral_truncation": args.spectral_truncation,
        "trunc_alpha": float(args.trunc_alpha),
        "trunc_power": float(args.trunc_power),
        "snapshots": len(snapshots),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.meta_only:
        print(f"[truth] wrote {meta_path}")
        return

    np.savez(
        npz_path,
        omega_snapshots=np.stack(snapshots, axis=0),
        steps=np.asarray(snapshot_steps, dtype=np.int64),
        energy=np.asarray(energies, dtype=np.float64),
        enstrophy=np.asarray(enstrophies, dtype=np.float64),
        max_abs=np.asarray(max_abs, dtype=np.float64),
        omega0=omega0.astype(np.float32),
        meta_json=json.dumps(meta),
    )
    print(f"[truth] wrote {npz_path} and {meta_path}")
    if args.timing_detail:
        wall = time.perf_counter() - t0_wall
        cpu = time.process_time() - t0_cpu
        print(f"[truth-timing] wall={wall:.3f}s cpu={cpu:.3f}s snapshots={len(snapshots)}")
    if args.update_manifest:
        manifest_path = out_prefix.parent / "truth_manifest.json"
        manifest: Dict[str, object] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = {}
        latest = dict(manifest.get("latest", {}))
        entry = {
            "npz": str(npz_path),
            "meta": str(meta_path),
            "backend": args.backend,
            "truth_tag": truth_tag,
            "run_ts": run_ts,
        }
        latest[truth_tag] = entry
        manifest["latest"] = latest
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[truth] updated {manifest_path}")


if __name__ == "__main__":
    main()
