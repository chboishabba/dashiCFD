#!/usr/bin/env python3
"""Synthetic NS theta-profile sweep for the DASHI EV5 seam diagnostic.

This script is evidence-only.  It generates three deterministic shell traces
and computes the fixed-cutoff profile

    theta(k,t) = |Flux_{>k}(t)| / Diss_{>k}(t)

from the observed tail balance

    dE_tail/dt = Flux_tail - Diss_tail.

The traces are deliberately simple:

* ``forced_taylor_green`` keeps injecting energy near a selected shell.
* ``unforced_smooth_decay`` is smooth and dissipative.
* ``near_critical_tail`` has a shallow high-frequency tail.

The CSV always writes ``promotion_status=NO_PROMOTION``.  A theta margin is a
runtime locator, not a proof of Navier-Stokes regularity or a Clay claim.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


PROMOTION_STATUS = "NO_PROMOTION"


@dataclass(frozen=True)
class ThetaRow:
    trace_type: str
    transition_index: int
    time: float
    k: int
    theta_k: float
    theta: float
    danger_shell: int
    margin: float
    tail_energy_before: float
    tail_energy_after: float
    tail_energy_derivative: float
    dissipation: float
    flux: float
    promotion_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/ns_theta_sweep.csv"))
    parser.add_argument("--k-max", type=int, default=48)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--nu", type=float, default=1.0e-3)
    parser.add_argument("--k-star", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def k_star_from_nu(nu: float, k_max: int) -> int:
    if nu <= 0.0 or not math.isfinite(nu):
        raise ValueError("nu must be positive and finite")
    raw = int(math.floor(0.5 * math.log2(1.0 / nu)))
    return max(1, min(k_max - 2, raw))


def shell_grid(k_max: int) -> np.ndarray:
    if k_max < 8:
        raise ValueError("k_max must be at least 8")
    return np.arange(1, k_max + 1, dtype=np.float64)


def unforced_smooth_decay(shells: np.ndarray, step: int, dt: float, nu: float, rng: np.random.Generator) -> np.ndarray:
    base = np.exp(-shells / 4.0)
    decay = np.exp(-nu * (2.0 ** (2.0 * shells)) * step * dt)
    return np.square(base * decay)


def forced_taylor_green(shells: np.ndarray, step: int, dt: float, nu: float, rng: np.random.Generator) -> np.ndarray:
    base = np.power(shells, -5.0 / 3.0)
    decay = np.exp(-0.25 * nu * (2.0 ** (2.0 * shells)) * step * dt)
    forcing_center = 8.0 + 0.25 * math.sin(step * dt)
    forcing = 0.18 * np.exp(-0.5 * np.square((shells - forcing_center) / 1.5))
    phase = 1.0 + 0.04 * np.sin(0.7 * step + shells)
    return np.maximum(0.0, np.square(base * decay * phase) + forcing)


def near_critical_tail(shells: np.ndarray, step: int, dt: float, nu: float, rng: np.random.Generator) -> np.ndarray:
    base = np.power(shells, -1.0)
    weak_decay = np.exp(-0.04 * nu * (2.0 ** (2.0 * shells)) * step * dt)
    ridge_center = 0.58 * float(shells[-1])
    ridge = 0.025 * np.exp(-0.5 * np.square((shells - ridge_center) / 3.0))
    return np.maximum(0.0, np.square(base * weak_decay) + ridge)


TRACE_BUILDERS: dict[str, Callable[[np.ndarray, int, float, float, np.random.Generator], np.ndarray]] = {
    "forced_taylor_green": forced_taylor_green,
    "unforced_smooth_decay": unforced_smooth_decay,
    "near_critical_tail": near_critical_tail,
}


def tail_energy(energy: np.ndarray, k: int) -> float:
    return float(np.sum(energy[k:]))


def tail_dissipation(energy: np.ndarray, shells: np.ndarray, k: int, nu: float) -> float:
    tail_shells = shells[k:]
    tail_energy_values = energy[k:]
    weights = np.power(2.0, 2.0 * tail_shells)
    return float(2.0 * nu * np.sum(weights * tail_energy_values))


def theta_profile_for_transition(
    before: np.ndarray,
    after: np.ndarray,
    shells: np.ndarray,
    *,
    trace_type: str,
    transition_index: int,
    time: float,
    dt: float,
    nu: float,
    k_start: int,
) -> list[ThetaRow]:
    provisional: list[dict[str, float | int | str]] = []
    for k in range(k_start, len(shells) - 1):
        e_before = tail_energy(before, k)
        e_after = tail_energy(after, k)
        derivative = (e_after - e_before) / dt
        diss = tail_dissipation(before, shells, k, nu)
        if diss <= 0.0 or not math.isfinite(diss):
            theta_k = math.inf
            flux = math.inf
        else:
            flux = derivative + diss
            theta_k = abs(flux) / diss
        provisional.append(
            {
                "k": k,
                "theta_k": float(theta_k),
                "tail_energy_before": e_before,
                "tail_energy_after": e_after,
                "tail_energy_derivative": derivative,
                "dissipation": diss,
                "flux": float(flux),
            }
        )

    finite = [row for row in provisional if math.isfinite(float(row["theta_k"]))]
    if finite:
        danger = max(finite, key=lambda row: float(row["theta_k"]))
        theta = float(danger["theta_k"])
        danger_shell = int(danger["k"])
    else:
        theta = math.inf
        danger_shell = -1
    margin = 1.0 - theta if math.isfinite(theta) else -math.inf

    rows: list[ThetaRow] = []
    for row in provisional:
        rows.append(
            ThetaRow(
                trace_type=trace_type,
                transition_index=transition_index,
                time=time,
                k=int(row["k"]),
                theta_k=float(row["theta_k"]),
                theta=theta,
                danger_shell=danger_shell,
                margin=margin,
                tail_energy_before=float(row["tail_energy_before"]),
                tail_energy_after=float(row["tail_energy_after"]),
                tail_energy_derivative=float(row["tail_energy_derivative"]),
                dissipation=float(row["dissipation"]),
                flux=float(row["flux"]),
                promotion_status=PROMOTION_STATUS,
            )
        )
    return rows


def run_sweep(k_max: int, steps: int, dt: float, nu: float, k_star: int, seed: int) -> list[ThetaRow]:
    shells = shell_grid(k_max)
    rng = np.random.default_rng(seed)
    rows: list[ThetaRow] = []
    for trace_type, builder in TRACE_BUILDERS.items():
        trace = [builder(shells, step, dt, nu, rng) for step in range(steps + 1)]
        for i, (before, after) in enumerate(zip(trace[:-1], trace[1:])):
            rows.extend(
                theta_profile_for_transition(
                    before,
                    after,
                    shells,
                    trace_type=trace_type,
                    transition_index=i,
                    time=(i + 1) * dt,
                    dt=dt,
                    nu=nu,
                    k_start=k_star,
                )
            )
    return rows


def write_csv(path: Path, rows: Iterable[ThetaRow]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ThetaRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in materialized:
            writer.writerow(row.__dict__)


def main() -> None:
    args = parse_args()
    k_star = args.k_star
    if k_star is None:
        k_star = k_star_from_nu(args.nu, args.k_max)
    rows = run_sweep(
        k_max=args.k_max,
        steps=args.steps,
        dt=args.dt,
        nu=args.nu,
        k_star=k_star,
        seed=args.seed,
    )
    write_csv(args.out, rows)
    summary: dict[str, tuple[float, int]] = {}
    for trace_type in TRACE_BUILDERS:
        trace_rows = [row for row in rows if row.trace_type == trace_type]
        if not trace_rows:
            continue
        worst = max(trace_rows, key=lambda row: row.theta if math.isfinite(row.theta) else math.inf)
        summary[trace_type] = (worst.theta, worst.danger_shell)
    print(f"wrote {args.out} rows={len(rows)} k_star={k_star} promotion_status={PROMOTION_STATUS}")
    for trace_type, (theta, danger_shell) in summary.items():
        print(f"{trace_type}: Theta={theta:.6g} danger_shell={danger_shell}")


if __name__ == "__main__":
    main()
