#!/usr/bin/env python3
"""
Empirical Gate 3 atom-frame sweep.

This script builds deterministic anisotropic atom dictionaries on N x N grids
and reports empirical frame bounds from the nonzero eigenvalues of the Gram
operator.  The numbers are smoke-test diagnostics for the finite dictionaries;
they are not promoted to a formal uniform frame proof.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - exercised only on minimal Python installs.
    np = None  # type: ignore[assignment]


NS = (8, 16, 32, 64)
PROMOTION_STATUS = "NO_PROMOTION"


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
        default=Path("gate3_atom_frame.csv"),
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


def _positive_bounds(eigenvalues: Iterable[float]) -> tuple[float, float]:
    vals = sorted(float(v) for v in eigenvalues if float(v) > 1e-10)
    if not vals:
        return 0.0, 0.0
    return vals[0], vals[-1]


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _compute_bounds_numpy(n: int, specs: Sequence[AtomSpec]) -> tuple[float, float]:
    yy, xx = np.indices((n, n), dtype=np.float64)  # type: ignore[union-attr]
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
    return _positive_bounds(float(v) for v in eigenvalues)


def _periodic_delta(index: int, center: float, n: int) -> float:
    return ((float(index) - center + n / 2.0) % float(n)) - n / 2.0


def _compute_bounds_pure_python(n: int, specs: Sequence[AtomSpec]) -> tuple[float, float]:
    columns: list[list[float]] = []
    sigma_major = max(n / 5.5, 1.0)
    for spec in specs:
        y0 = spec.center_y * n
        x0 = spec.center_x * n
        c = math.cos(spec.orientation)
        s = math.sin(spec.orientation)
        sigma_minor = sigma_major / spec.anisotropy
        values: list[float] = []
        total = 0.0
        for y in range(n):
            dy = _periodic_delta(y, y0, n)
            for x in range(n):
                dx = _periodic_delta(x, x0, n)
                major = c * dx + s * dy
                minor = -s * dx + c * dy
                envelope = math.exp(-0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2))
                if spec.phase is None or spec.twist is None:
                    value = envelope
                else:
                    value = envelope * math.cos((2.0 * math.pi / n) * major + spec.twist * minor / n + spec.phase)
                value *= spec.amplitude
                values.append(value)
                total += value
        mean = total / len(values)
        centered = [v - mean for v in values]
        norm = math.sqrt(sum(v * v for v in centered))
        if norm > 1e-12:
            columns.append([v / norm for v in centered])

    gram = [[0.0 for _ in columns] for _ in columns]
    for i, col_i in enumerate(columns):
        gram[i][i] = 1.0
        for j in range(i + 1, len(columns)):
            value = sum(a * b for a, b in zip(col_i, columns[j]))
            gram[i][j] = value
            gram[j][i] = value
    return _positive_bounds(_jacobi_eigenvalues(gram))


def _jacobi_eigenvalues(matrix: list[list[float]]) -> list[float]:
    """Small symmetric eigensolver fallback for environments without numpy."""
    n = len(matrix)
    if n == 0:
        return []
    max_sweeps = max(25, 3 * n)
    tolerance = 1e-12
    for _ in range(max_sweeps):
        p = 0
        q = 1 if n > 1 else 0
        largest = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                value = abs(matrix[i][j])
                if value > largest:
                    largest = value
                    p = i
                    q = j
        if largest < tolerance or n == 1:
            break

        app = matrix[p][p]
        aqq = matrix[q][q]
        apq = matrix[p][q]
        tau = (aqq - app) / (2.0 * apq)
        sign = 1.0 if tau >= 0.0 else -1.0
        t = sign / (abs(tau) + math.sqrt(1.0 + tau * tau))
        cosine = 1.0 / math.sqrt(1.0 + t * t)
        sine = t * cosine

        for k in range(n):
            if k == p or k == q:
                continue
            mkp = matrix[k][p]
            mkq = matrix[k][q]
            matrix[k][p] = cosine * mkp - sine * mkq
            matrix[p][k] = matrix[k][p]
            matrix[k][q] = sine * mkp + cosine * mkq
            matrix[q][k] = matrix[k][q]
        matrix[p][p] = app - t * apq
        matrix[q][q] = aqq + t * apq
        matrix[p][q] = 0.0
        matrix[q][p] = 0.0
    return [matrix[i][i] for i in range(n)]


def compute_bounds(n: int, *, phase_complete: bool) -> tuple[float, float]:
    specs = _atom_specs(phase_complete=phase_complete)
    if np is not None:
        return _compute_bounds_numpy(n, specs)
    return _compute_bounds_pure_python(n, specs)


def sweep(sizes: Sequence[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for n in sizes:
        for phase_complete in (True, False):
            lower, upper = compute_bounds(n, phase_complete=phase_complete)
            ratio = math.inf if lower <= 0.0 else upper / lower
            rows.append(
                {
                    "N": str(n),
                    "A_N": _format_float(lower),
                    "B_N": _format_float(upper),
                    "frame_ratio": _format_float(ratio),
                    "phase_included": "TRUE" if phase_complete else "FALSE",
                    "twist_included": "TRUE" if phase_complete else "FALSE",
                    "promotion_status": PROMOTION_STATUS,
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "N",
        "A_N",
        "B_N",
        "frame_ratio",
        "phase_included",
        "twist_included",
        "promotion_status",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = sweep(_selected_sizes(args.max_n))
    write_csv(args.output, rows)
    print(f"wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
