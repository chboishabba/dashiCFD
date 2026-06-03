#!/usr/bin/env python3
"""Deterministic p=7 polymer bound table through diameter 3."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

P = 7
DIAMETERS = (1, 2, 3)
BETAS = (6.0, 7.69, 10.13, 13.64, 16.7)
C_MIN = 0.242
A_WEIGHT = 0.5
PROMOTION_STATUS = "NO_PROMOTION"
DEFAULT_OUTPUT = Path("ym_p7_polymer_d3.csv")


@dataclass(frozen=True)
class PolymerBoundRow:
    beta: float
    diameter: int
    count: int
    action_lower_bound: float
    activity_bound: float
    entropy_weight: float
    weighted_sum: float
    cumulative_ratio: float
    normalised_threshold: float
    strict_absorption_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV path to write; default: ym_p7_polymer_d3.csv",
    )
    parser.add_argument(
        "--C0",
        "--c0",
        dest="c0",
        type=float,
        default=1.0,
        help="multiplicative constant C0 in absorption thresholds; default: 1",
    )
    return parser.parse_args()


def polymer_count(diameter: int) -> int:
    """Diameter-d p=7 deterministic branching count."""
    if diameter not in DIAMETERS:
        raise ValueError(f"unsupported diameter {diameter}; expected 1, 2, or 3")
    return P**diameter


def action_lower_bound(diameter: int) -> float:
    return C_MIN * diameter


def activity_bound(beta: float, diameter: int) -> float:
    return math.exp(-beta * action_lower_bound(diameter))


def entropy_weight(diameter: int) -> float:
    return math.exp(A_WEIGHT * diameter)


def diameter_weighted_sum(beta: float, diameter: int) -> float:
    return polymer_count(diameter) * entropy_weight(diameter) * activity_bound(beta, diameter)


def normalised_threshold(c0: float) -> float:
    return (A_WEIGHT + math.log(P) + math.log(c0)) / C_MIN


def strict_absorption_threshold(c0: float) -> float:
    return (A_WEIGHT + math.log(2.0 * P) + math.log(c0)) / C_MIN


def enumerate_rows(c0: float) -> list[PolymerBoundRow]:
    if c0 <= 0.0 or not math.isfinite(c0):
        raise ValueError(f"C0 must be positive and finite, got {c0!r}")
    normalised = normalised_threshold(c0)
    strict = strict_absorption_threshold(c0)
    rows: list[PolymerBoundRow] = []
    for beta in BETAS:
        cumulative = 0.0
        for diameter in DIAMETERS:
            weighted = diameter_weighted_sum(beta, diameter)
            cumulative += weighted
            rows.append(
                PolymerBoundRow(
                    beta=beta,
                    diameter=diameter,
                    count=polymer_count(diameter),
                    action_lower_bound=action_lower_bound(diameter),
                    activity_bound=activity_bound(beta, diameter),
                    entropy_weight=entropy_weight(diameter),
                    weighted_sum=weighted,
                    cumulative_ratio=cumulative,
                    normalised_threshold=normalised,
                    strict_absorption_threshold=strict,
                )
            )
    return rows


def write_csv(path: Path, rows: list[PolymerBoundRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "p",
        "diameter",
        "beta",
        "c_min",
        "a",
        "count_p_to_d",
        "action_lower_bound",
        "activity_bound",
        "entropy_weight",
        "weighted_sum",
        "cumulative_ratio",
        "promotion_status",
        "normalised_threshold",
        "strict_absorption_threshold",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "p": P,
                    "diameter": row.diameter,
                    "beta": f"{row.beta:.12g}",
                    "c_min": f"{C_MIN:.12g}",
                    "a": f"{A_WEIGHT:.12g}",
                    "count_p_to_d": row.count,
                    "action_lower_bound": f"{row.action_lower_bound:.12g}",
                    "activity_bound": f"{row.activity_bound:.12g}",
                    "entropy_weight": f"{row.entropy_weight:.12g}",
                    "weighted_sum": f"{row.weighted_sum:.12g}",
                    "cumulative_ratio": f"{row.cumulative_ratio:.12g}",
                    "promotion_status": PROMOTION_STATUS,
                    "normalised_threshold": f"{row.normalised_threshold:.12g}",
                    "strict_absorption_threshold": f"{row.strict_absorption_threshold:.12g}",
                }
            )


def main() -> int:
    args = parse_args()
    rows = enumerate_rows(float(args.c0))
    write_csv(args.output, rows)
    print(
        f"wrote {args.output}; rows={len(rows)}; "
        f"promotion_status={PROMOTION_STATUS}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
