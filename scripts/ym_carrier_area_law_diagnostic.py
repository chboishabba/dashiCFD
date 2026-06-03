#!/usr/bin/env python3
"""Finite carrier area-law sigma diagnostic.

This script is evidence-only.  It estimates the carrier string-tension
bookkeeping value

    sigma_DASHI = (beta_carrier * c_min - log(p)) / area_normalization

for finite carrier lattice sizes and writes ``sigma_dashi.csv``.  It also
records whether the supplied beta is above the strict absorption threshold.
The output is deterministic and does not claim a continuum Yang-Mills area
law, a mass gap, or Clay promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for sigma_dashi.csv and checks JSON")
    parser.add_argument("--p", type=int, default=7, help="Bruhat-Tits lane prime")
    parser.add_argument("--beta-carrier", type=float, default=14.0, help="carrier beta used in sigma_DASHI")
    parser.add_argument("--beta-absorption", type=float, default=13.64, help="strict KP absorption beta threshold")
    parser.add_argument("--c-min", type=float, default=1.0, help="carrier c_min lower activity constant")
    parser.add_argument(
        "--area-normalization",
        type=float,
        default=100.0,
        help="positive area normalization in sigma_DASHI denominator",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="4,8,16,32",
        help="comma-separated finite carrier lattice sizes",
    )
    return parser.parse_args()


def parse_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise SystemExit(f"lattice sizes must be positive, got {value!r}")
        sizes.append(value)
    if not sizes:
        raise SystemExit("--sizes must contain at least one positive integer")
    return sizes


def validate_inputs(p: int, beta_carrier: float, c_min: float, area_normalization: float) -> None:
    if p <= 1:
        raise SystemExit(f"--p must be an integer prime lane greater than 1, got {p!r}")
    for name, value in (
        ("--beta-carrier", beta_carrier),
        ("--c-min", c_min),
        ("--area-normalization", area_normalization),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SystemExit(f"{name} must be finite and positive, got {value!r}")


def sigma_dashi(p: int, beta_carrier: float, c_min: float, area_normalization: float) -> float:
    return (float(beta_carrier) * float(c_min) - math.log(float(p))) / float(area_normalization)


def rows_for_sizes(
    sizes: list[int],
    *,
    p: int,
    beta_carrier: float,
    beta_absorption: float,
    c_min: float,
    area_normalization: float,
) -> list[dict[str, object]]:
    sigma = sigma_dashi(p, beta_carrier, c_min, area_normalization)
    beta_margin = float(beta_carrier) - float(beta_absorption)
    sigma_positive_from_absorption = beta_margin > 0.0 and sigma > 0.0
    rows = []
    for size in sizes:
        plaquette_area = 1.0 / float(size * size)
        full_lattice_area = float(size * size) * plaquette_area
        wilson_bound_unit_area = math.exp(-sigma)
        wilson_bound_full_lattice = math.exp(-sigma * full_lattice_area)
        rows.append(
            {
                "lattice_size": int(size),
                "p": int(p),
                "beta_carrier": float(beta_carrier),
                "beta_absorption": float(beta_absorption),
                "beta_margin": beta_margin,
                "c_min": float(c_min),
                "area_normalization": float(area_normalization),
                "sigma_DASHI": sigma,
                "sigma_positive_from_absorption": bool(sigma_positive_from_absorption),
                "plaquette_area": plaquette_area,
                "full_lattice_area": full_lattice_area,
                "wilson_bound_unit_area": wilson_bound_unit_area,
                "wilson_bound_full_lattice": wilson_bound_full_lattice,
                "carrier_area_law_diagnostic": bool(sigma_positive_from_absorption),
                "continuum_area_law_requires_gate3": True,
                "clay_ym_promoted": False,
                "evidence_only": True,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "lattice_size",
        "p",
        "beta_carrier",
        "beta_absorption",
        "beta_margin",
        "c_min",
        "area_normalization",
        "sigma_DASHI",
        "sigma_positive_from_absorption",
        "plaquette_area",
        "full_lattice_area",
        "wilson_bound_unit_area",
        "wilson_bound_full_lattice",
        "carrier_area_law_diagnostic",
        "continuum_area_law_requires_gate3",
        "clay_ym_promoted",
        "evidence_only",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args.sizes)
    validate_inputs(args.p, args.beta_carrier, args.c_min, args.area_normalization)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = rows_for_sizes(
        sizes,
        p=args.p,
        beta_carrier=args.beta_carrier,
        beta_absorption=args.beta_absorption,
        c_min=args.c_min,
        area_normalization=args.area_normalization,
    )
    csv_path = args.out_dir / "sigma_dashi.csv"
    checks_path = args.out_dir / "sigma_dashi_checks.json"
    write_csv(csv_path, rows)

    sigma = float(rows[0]["sigma_DASHI"])
    checks = {
        "sigma_DASHI": sigma,
        "sigma_formula": "(beta_carrier * c_min - log(p)) / area_normalization",
        "beta_carrier_above_absorption": bool(args.beta_carrier > args.beta_absorption),
        "sigma_positive": bool(sigma > 0.0),
        "carrier_area_law_diagnostic": bool(args.beta_carrier > args.beta_absorption and sigma > 0.0),
        "continuum_area_law_requires_gate3": True,
        "clay_ym_promoted": False,
        "evidence_only": True,
        "promotion_boundary": (
            "Finite carrier diagnostic only; not a continuum area-law theorem, "
            "not a Yang-Mills mass-gap theorem, and not Clay evidence."
        ),
        "rows": len(rows),
        "outputs": {"sigma_dashi_csv": str(csv_path)},
    }
    checks_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    print(f"[ym-area-law] wrote {csv_path}, {checks_path}")
    print(
        "[ym-area-law] "
        f"sigma_DASHI={sigma:.12g} "
        f"beta_margin={args.beta_carrier - args.beta_absorption:.12g} "
        f"carrier_area_law_diagnostic={checks['carrier_area_law_diagnostic']} "
        "continuum_area_law_requires_gate3=True clay_ym_promoted=False"
    )


if __name__ == "__main__":
    main()
