#!/usr/bin/env python3
"""Full deterministic NS theta-profile sweep for the DASHI seam diagnostic.

The diagnostic is evidence-only.  It evaluates five deterministic synthetic
shell-energy regimes and writes the cutoff profile

    theta(k,t) = |Flux_{>k}(t)| / Diss_{>k}(t)

using the balance convention

    dE_{>k}/dt = Flux_{>k} - Diss_{>k}.

The output rows are profile rows, one per ``trace, nu, k`` for every
``1 <= k <= K_max``.  ``theta_k`` is the supremum over observed transition
times at that cutoff, ``K_star`` is the first cutoff after which all finite
``theta_k`` values stay below one, and ``Theta`` is the supremum over cutoffs
at or above ``K_star`` for the same trace and viscosity.  Promotion is
intentionally disabled for every row.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


PROMOTION_STATUS = "NO_PROMOTION"
DEFAULT_NU_VALUES = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
THETA_BOUNDARY_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class ThetaProfileRow:
    trace: str
    nu: float
    k: int
    theta_k: float
    K_star: int
    Theta: float
    margin: float
    K_kolmogorov: int
    K_star_le_K_nu: bool
    promotion_status: str
    edge_leakage_ratio: float
    combined_ratio: float
    barrier_pass: bool


TraceBuilder = Callable[[np.ndarray, int, float, float], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ns_theta_full_sweep.csv"))
    parser.add_argument("--k-max", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--nu-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_NU_VALUES),
        help="positive viscosity values to sweep",
    )
    return parser.parse_args()


def validate_args(k_max: int, steps: int, dt: float, nu_values: Sequence[float]) -> None:
    if k_max < 8:
        raise ValueError("k-max must be at least 8")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be positive and finite")
    if not nu_values:
        raise ValueError("at least one nu value is required")
    for nu in nu_values:
        if nu <= 0.0 or not math.isfinite(nu):
            raise ValueError(f"nu must be positive and finite, got {nu!r}")


def shell_grid(k_max: int) -> np.ndarray:
    return np.arange(1, k_max + 1, dtype=np.float64)


def k_kolmogorov_from_nu(nu: float, k_max: int) -> int:
    raw = int(math.floor(0.75 * math.log2(1.0 / nu)))
    return max(1, min(k_max, raw))


def smooth_trace(shells: np.ndarray, step: int, dt: float, nu: float) -> np.ndarray:
    base = np.exp(-shells / 4.0)
    decay = np.exp(-nu * np.power(2.0, 2.0 * shells) * step * dt)
    return np.square(base * decay)


def kolmogorov_trace(shells: np.ndarray, step: int, dt: float, nu: float) -> np.ndarray:
    base = np.power(shells, -5.0 / 3.0)
    decay = np.exp(-0.25 * nu * np.power(2.0, 2.0 * shells) * step * dt)
    center = 8.0 + 0.25 * math.sin(step * dt)
    forcing = 0.18 * np.exp(-0.5 * np.square((shells - center) / 1.5))
    phase = 1.0 + 0.04 * np.sin(0.7 * step + shells)
    return np.maximum(0.0, np.square(base * decay * phase) + forcing)


def near_critical_trace(shells: np.ndarray, step: int, dt: float, nu: float) -> np.ndarray:
    base = np.power(shells, -1.0)
    weak_decay = np.exp(-0.04 * nu * np.power(2.0, 2.0 * shells) * step * dt)
    ridge_center = 0.58 * float(shells[-1])
    ridge = 0.025 * np.exp(-0.5 * np.square((shells - ridge_center) / 3.0))
    return np.maximum(0.0, np.square(base * weak_decay) + ridge)


def inviscid_trace(shells: np.ndarray, step: int, dt: float, nu: float) -> np.ndarray:
    drift = max(3.0, 0.74 * float(shells[-1]) - 0.35 * step)
    packet = 0.050 * np.exp(-0.5 * np.square((shells - drift) / 2.8))
    background = 0.016 * np.power(shells, -1.25)
    oscillation = 1.0 + 0.05 * np.sin(0.45 * step + 0.31 * shells)
    weak_decay = np.exp(-0.01 * nu * np.power(2.0, 2.0 * shells) * step * dt)
    return np.maximum(0.0, (background + packet) * oscillation * weak_decay)


def rough_trace(shells: np.ndarray, step: int, dt: float, nu: float) -> np.ndarray:
    base = 0.012 * np.power(shells, -1.15)
    burst_time = math.exp(-0.5 * ((step - 0.55 / dt) / max(1.0, 0.14 / dt)) ** 2)
    burst_center = 0.76 * float(shells[-1]) + 1.2 * math.sin(0.6 * step)
    burst = 0.035 * burst_time * np.exp(-0.5 * np.square((shells - burst_center) / 2.0))
    ripple = 1.0 + 0.03 * np.sin(1.7 * shells + 0.9 * step)
    decay = np.exp(-0.07 * nu * np.power(2.0, 2.0 * shells) * step * dt)
    return np.maximum(0.0, (base + burst) * ripple * decay)


TRACE_BUILDERS: dict[str, TraceBuilder] = {
    "kolmogorov": kolmogorov_trace,
    "smooth": smooth_trace,
    "near-critical": near_critical_trace,
    "inviscid": inviscid_trace,
    "rough": rough_trace,
}


def tail_energy(energy: np.ndarray, shells: np.ndarray, cutoff: int) -> float:
    return float(np.sum(energy[shells >= float(cutoff)]))


def tail_dissipation(energy: np.ndarray, shells: np.ndarray, cutoff: int, nu: float) -> float:
    mask = shells >= float(cutoff)
    tail_shells = shells[mask]
    tail_energy_values = energy[mask]
    weights = np.power(2.0, 2.0 * tail_shells)
    return float(2.0 * nu * np.dot(weights, tail_energy_values))


def theta_for_cutoff_transitions(
    trace: Sequence[np.ndarray], shells: np.ndarray, cutoff: int, dt: float, nu: float
) -> float:
    values: list[float] = []
    for before, after in zip(trace, trace[1:]):
        e_before = tail_energy(before, shells, cutoff)
        e_after = tail_energy(after, shells, cutoff)
        derivative = (e_after - e_before) / dt
        dissipation = tail_dissipation(before, shells, cutoff, nu)
        if dissipation <= 0.0 or not math.isfinite(dissipation):
            values.append(math.inf)
            continue
        flux = derivative + dissipation
        values.append(abs(flux) / dissipation)
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.inf


def edge_leakage_ratio_for_cutoff(
    trace: Sequence[np.ndarray], shells: np.ndarray, cutoff: int
) -> float:
    edge_mask = shells == float(cutoff)
    if not bool(np.any(edge_mask)):
        return 0.0
    ratios: list[float] = []
    for energy in trace:
        tail = tail_energy(energy, shells, cutoff)
        if tail <= 0.0 or not math.isfinite(tail):
            continue
        edge_energy = float(np.sum(energy[edge_mask]))
        ratios.append(edge_energy / tail)
    if not ratios:
        return 0.0
    return max(ratios) / float(cutoff + 1)


def build_trace(
    builder: TraceBuilder,
    shells: np.ndarray,
    steps: int,
    dt: float,
    nu: float,
) -> list[np.ndarray]:
    return [builder(shells, step, dt, nu) for step in range(steps + 1)]


def select_k_star(theta_by_cutoff: dict[int, float], k_max: int) -> int:
    for cutoff in range(1, k_max + 1):
        tail = [theta_by_cutoff[k] for k in range(cutoff, k_max + 1)]
        if tail and all(
            math.isfinite(theta) and theta <= 1.0 + THETA_BOUNDARY_TOLERANCE
            for theta in tail
        ):
            return cutoff
    return k_max


def profile_rows_for_trace_nu(
    trace_name: str,
    trace: Sequence[np.ndarray],
    shells: np.ndarray,
    nu: float,
    dt: float,
    k_kolmogorov: int,
) -> list[ThetaProfileRow]:
    k_max = int(shells[-1])
    cutoffs = list(range(1, k_max + 1))
    theta_by_cutoff = {
        cutoff: theta_for_cutoff_transitions(trace, shells, cutoff, dt, nu) for cutoff in cutoffs
    }
    edge_leakage_by_cutoff = {
        cutoff: edge_leakage_ratio_for_cutoff(trace, shells, cutoff) for cutoff in cutoffs
    }
    k_star = select_k_star(theta_by_cutoff, k_max)
    finite_theta = [
        theta for cutoff, theta in theta_by_cutoff.items() if cutoff >= k_star and math.isfinite(theta)
    ]
    theta_sup = max(finite_theta) if finite_theta else math.inf
    margin = 1.0 - theta_sup if math.isfinite(theta_sup) else -math.inf
    rows: list[ThetaProfileRow] = []
    for cutoff, theta_k in theta_by_cutoff.items():
        edge_leakage_ratio = edge_leakage_by_cutoff[cutoff]
        combined_ratio = (
            float(theta_k) + edge_leakage_ratio
            if math.isfinite(theta_k)
            else math.inf
        )
        rows.append(
            ThetaProfileRow(
                trace=trace_name,
                nu=float(nu),
                k=int(cutoff),
                theta_k=float(theta_k),
                K_star=int(k_star),
                Theta=float(theta_sup),
                margin=float(margin),
                K_kolmogorov=int(k_kolmogorov),
                K_star_le_K_nu=bool(k_star <= k_kolmogorov),
                promotion_status=PROMOTION_STATUS,
                edge_leakage_ratio=float(edge_leakage_ratio),
                combined_ratio=float(combined_ratio),
                barrier_pass=bool(math.isfinite(combined_ratio) and combined_ratio < 1.0),
            )
        )
    return rows


def run_sweep(
    *,
    k_max: int,
    steps: int,
    dt: float,
    nu_values: Sequence[float],
) -> list[ThetaProfileRow]:
    validate_args(k_max, steps, dt, nu_values)
    shells = shell_grid(k_max)
    rows: list[ThetaProfileRow] = []
    for nu in nu_values:
        k_kolmogorov = k_kolmogorov_from_nu(float(nu), k_max)
        for trace_name, builder in TRACE_BUILDERS.items():
            trace = build_trace(builder, shells, steps, dt, float(nu))
            rows.extend(
                profile_rows_for_trace_nu(
                    trace_name=trace_name,
                    trace=trace,
                    shells=shells,
                    nu=float(nu),
                    dt=dt,
                    k_kolmogorov=k_kolmogorov,
                )
            )
    return rows


def write_csv(path: Path, rows: Iterable[ThetaProfileRow]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ThetaProfileRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in materialized:
            writer.writerow(row.__dict__)


def summarize(rows: Sequence[ThetaProfileRow]) -> list[str]:
    lines: list[str] = []
    keys = sorted({(row.trace, row.nu) for row in rows})
    for trace_name, nu in keys:
        group = [row for row in rows if row.trace == trace_name and row.nu == nu]
        finite = [row for row in group if math.isfinite(row.theta_k)]
        worst = max(finite, key=lambda row: row.theta_k) if finite else group[0]
        lines.append(
            f"{trace_name} nu={nu:g}: "
            f"Theta={worst.Theta:.6g} danger_k={worst.k} "
            f"K_star={worst.K_star} K_kolmogorov={worst.K_kolmogorov} "
            f"K_star_le_K_nu={worst.K_star_le_K_nu}"
        )
    return lines


def main() -> None:
    args = parse_args()
    rows = run_sweep(
        k_max=args.k_max,
        steps=args.steps,
        dt=args.dt,
        nu_values=args.nu_values,
    )
    write_csv(args.out, rows)
    print(f"wrote {args.out} rows={len(rows)} promotion_status={PROMOTION_STATUS}")
    for line in summarize(rows):
        print(line)


if __name__ == "__main__":
    main()
