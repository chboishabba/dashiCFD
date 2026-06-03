#!/usr/bin/env python3
"""
Extended empirical Gate 3 atom-frame sweep.

Builds deterministic synthetic anisotropic atom dictionaries on N x N grids and
reports frame bounds from the positive eigenvalues of the Gram operator, maximum
off-diagonal crossterm, and finite trend diagnostics.  This is a finite
diagnostic only; every row carries NO_PROMOTION status.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


NS = (8, 16, 32, 64, 128)
PROMOTION_STATUS = "NO_PROMOTION"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "gate3_frame_extended.csv"


@dataclass(frozen=True)
class AtomSpec:
    center_y: float
    center_x: float
    amplitude: float
    orientation: float
    anisotropy: float
    phase: float | None
    twist: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV output path",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="only run grid sizes N <= MAX_N",
    )
    return parser.parse_args()


def _selected_sizes(max_n: int | None) -> list[int]:
    sizes = [n for n in NS if max_n is None or n <= max_n]
    if not sizes:
        raise SystemExit(f"--max-n must be at least {min(NS)}")
    return sizes


def _atom_specs(*, phase_complete: bool) -> list[AtomSpec]:
    centers = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75))
    amplitudes = (0.75, 1.25)
    orientations = (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0)
    anisotropies = (1.0, 2.5)
    phases: Sequence[float | None] = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    twists: Sequence[float | None] = (-0.65, 0.65)
    if not phase_complete:
        phases = (None,)
        twists = (None,)

    return [
        AtomSpec(
            center_y=cy,
            center_x=cx,
            amplitude=amplitude,
            orientation=orientation,
            anisotropy=anisotropy,
            phase=phase,
            twist=twist,
        )
        for cy, cx in centers
        for amplitude in amplitudes
        for orientation in orientations
        for anisotropy in anisotropies
        for phase in phases
        for twist in twists
    ]


@dataclass(frozen=True)
class FrameDiagnostics:
    lower: float
    upper: float
    actual_lambda_min: float
    positive_rank: int
    atom_count: int
    max_crossterm: float
    net_quality: float


def _positive_bounds(eigenvalues: Sequence[float]) -> tuple[float, float, int]:
    vals = sorted(float(v) for v in eigenvalues if float(v) > 1e-10)
    if not vals:
        return 0.0, 0.0, 0
    return vals[0], vals[-1], len(vals)


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _net_quality(lower: float, upper: float, max_crossterm: float) -> float:
    if lower <= 0.0 or upper <= 0.0:
        return 0.0
    return lower / (upper * (1.0 + max_crossterm))


def _compute_frame_diagnostics(n: int, specs: Sequence[AtomSpec]) -> FrameDiagnostics:
    yy, xx = np.indices((n, n), dtype=np.float64)
    columns = []
    sigma_major = max(n / 5.5, 1.0)

    for spec in specs:
        y0 = spec.center_y * n
        x0 = spec.center_x * n
        dy = ((yy - y0 + n / 2.0) % n) - n / 2.0
        dx = ((xx - x0 + n / 2.0) % n) - n / 2.0
        c = math.cos(spec.orientation)
        s = math.sin(spec.orientation)
        major = c * dx + s * dy
        minor = -s * dx + c * dy
        sigma_minor = sigma_major / spec.anisotropy
        envelope = np.exp(-0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2))
        if spec.phase is None or spec.twist is None:
            atom = envelope
        else:
            carrier = np.cos((2.0 * math.pi / n) * major + spec.twist * minor / n + spec.phase)
            atom = envelope * carrier
        atom = spec.amplitude * (atom - float(np.mean(atom)))
        norm = float(np.linalg.norm(atom))
        if norm > 1e-12:
            columns.append((atom / norm).reshape(-1))

    frame_matrix = np.stack(columns, axis=1)
    gram = frame_matrix.T @ frame_matrix
    eigenvalues = np.linalg.eigvalsh(gram)
    offdiag = gram - np.diag(np.diag(gram))
    max_crossterm = float(np.max(np.abs(offdiag))) if offdiag.size else 0.0
    lower, upper, positive_rank = _positive_bounds([float(v) for v in eigenvalues])
    return FrameDiagnostics(
        lower=lower,
        upper=upper,
        actual_lambda_min=float(eigenvalues[0]) if eigenvalues.size else 0.0,
        positive_rank=positive_rank,
        atom_count=len(columns),
        max_crossterm=max_crossterm,
        net_quality=_net_quality(lower, upper, max_crossterm),
    )


def compute_diagnostics(n: int, *, phase_complete: bool) -> FrameDiagnostics:
    specs = _atom_specs(phase_complete=phase_complete)
    return _compute_frame_diagnostics(n, specs)


def _trend(current: float, previous: float | None, *, higher_is_better: bool) -> str:
    if previous is None:
        return "initial"
    tolerance = 1e-12 * max(1.0, abs(previous), abs(current))
    if abs(current - previous) <= tolerance:
        return "flat"
    improved = current > previous if higher_is_better else current < previous
    return "improved" if improved else "declined"


def sweep(sizes: Sequence[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    previous_by_dictionary: dict[str, FrameDiagnostics] = {}
    for n in sizes:
        for phase_complete in (True, False):
            diagnostics = compute_diagnostics(n, phase_complete=phase_complete)
            ratio = math.inf if diagnostics.lower <= 0.0 else diagnostics.upper / diagnostics.lower
            dictionary = "phase_complete" if phase_complete else "phase_blind"
            previous = previous_by_dictionary.get(dictionary)
            mu_n = diagnostics.max_crossterm
            gershgorin_radius = float(max(diagnostics.atom_count - 1, 0)) * mu_n
            rows.append(
                {
                    "N": str(n),
                    "grid_dimension": str(n * n),
                    "dictionary": dictionary,
                    "atom_count": str(diagnostics.atom_count),
                    "gram_positive_eigenvalues": str(diagnostics.positive_rank),
                    "A_N": _format_float(diagnostics.lower),
                    "B_N": _format_float(diagnostics.upper),
                    "frame_ratio": _format_float(ratio),
                    "max_crossterm": _format_float(diagnostics.max_crossterm),
                    "net_quality": _format_float(diagnostics.net_quality),
                    "A_N_trend": _trend(
                        diagnostics.lower,
                        None if previous is None else previous.lower,
                        higher_is_better=True,
                    ),
                    "B_N_trend": _trend(
                        diagnostics.upper,
                        None if previous is None else previous.upper,
                        higher_is_better=False,
                    ),
                    "max_crossterm_trend": _trend(
                        diagnostics.max_crossterm,
                        None if previous is None else previous.max_crossterm,
                        higher_is_better=False,
                    ),
                    "net_quality_trend": _trend(
                        diagnostics.net_quality,
                        None if previous is None else previous.net_quality,
                        higher_is_better=True,
                    ),
                    "phase_included": "TRUE" if phase_complete else "FALSE",
                    "twist_included": "TRUE" if phase_complete else "FALSE",
                    "promotion_status": PROMOTION_STATUS,
                    "mu_N": _format_float(mu_n),
                    "(N-1)*mu_N": _format_float(gershgorin_radius),
                    "gershgorin_lower": _format_float(1.0 - gershgorin_radius),
                    "actual_lambda_min": _format_float(diagnostics.actual_lambda_min),
                }
            )
            previous_by_dictionary[dictionary] = diagnostics
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "N",
        "grid_dimension",
        "dictionary",
        "atom_count",
        "gram_positive_eigenvalues",
        "A_N",
        "B_N",
        "frame_ratio",
        "max_crossterm",
        "net_quality",
        "A_N_trend",
        "B_N_trend",
        "max_crossterm_trend",
        "net_quality_trend",
        "phase_included",
        "twist_included",
        "promotion_status",
        "mu_N",
        "(N-1)*mu_N",
        "gershgorin_lower",
        "actual_lambda_min",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = sweep(_selected_sizes(args.max_n))
    write_csv(args.output, rows)
    min_lower = min(float(row["A_N"]) for row in rows)
    max_upper = max(float(row["B_N"]) for row in rows)
    max_crossterm = max(float(row["max_crossterm"]) for row in rows)
    max_quality = max(float(row["net_quality"]) for row in rows)
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(
        "summary "
        f"N={','.join(str(n) for n in _selected_sizes(args.max_n))}; "
        f"dictionaries=phase_complete,phase_blind; "
        f"min_A_N={_format_float(min_lower)}; "
        f"max_B_N={_format_float(max_upper)}; "
        f"max_crossterm={_format_float(max_crossterm)}; "
        f"max_net_quality={_format_float(max_quality)}; "
        f"promotion_status={PROMOTION_STATUS}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
