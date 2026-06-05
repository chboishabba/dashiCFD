#!/usr/bin/env python3
"""Sprint 62 direction-coherence robustness audit.

Sprint 61 found the selected high raw-red packets direction-incoherent under
one proxy configuration.  Sprint 62 checks whether that conclusion is stable
under post-hoc threshold and top-population sensitivity, and whether the result
is consistent across available N/seed/run groups.

This script consumes Sprint 61's packet anatomy CSV.  It does not recompute
truth fields and it cannot substitute for missing dense-cadence or N128 runs.
Those unavailable robustness surfaces are reported explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import ns_sprint53_no2cycle_physical_amplitude_audit as sprint53


SENSITIVITY_FIELDS = [
    "top_fraction",
    "coherence_threshold",
    "lipschitz_threshold",
    "selected_count",
    "incoherent_count",
    "incoherent_fraction",
    "direction_coherence_mean",
    "direction_lipschitz_proxy_mean",
    "robustness_label",
]

GROUP_FIELDS = [
    "run",
    "N",
    "seed",
    "selected_count",
    "incoherent_count",
    "incoherent_fraction",
    "direction_coherence_mean",
    "direction_lipschitz_proxy_mean",
    "group_route_label",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sprint61-csv", type=Path, required=True, help="ns_raw_red_direction_coherence.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    p.add_argument("--coherence-thresholds", type=float, nargs="+", default=[0.60, 0.70, 0.80, 0.90])
    p.add_argument("--lipschitz-thresholds", type=float, nargs="+", default=[2.0])
    p.add_argument("--incoherent-fraction-threshold", type=float, default=0.20)
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"run", "weighted_A_raw_positive", "direction_coherence_mean", "direction_lipschitz_proxy"}
    missing = sorted(required.difference(rows[0].keys() if rows else []))
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    return rows


def _run_meta(run: str) -> tuple[str, str]:
    n_match = re.search(r"_N(\d+)_", run)
    seed_match = re.search(r"_seed(\d+)", run)
    return (n_match.group(1) if n_match else "unknown", seed_match.group(1) if seed_match else "unknown")


def _label(frac: float, threshold: float) -> str:
    if frac > threshold:
        return "direction_incoherent_concentration_blocked"
    return "direction_coherent_cfm_route_alive_diagnostic"


def _selected(rows: list[dict[str, str]], top_fraction: float) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: float(row.get("weighted_A_raw_positive") or 0.0), reverse=True)
    if not ordered:
        return []
    count = max(1, int(round(float(top_fraction) * len(ordered))))
    count = min(count, len(ordered))
    return ordered[:count]


def _stats(rows: list[dict[str, str]], coherence_threshold: float, lipschitz_threshold: float) -> dict[str, float]:
    count = len(rows)
    if count == 0:
        return {"count": 0.0, "incoherent": 0.0, "frac": 0.0, "coh": 0.0, "lip": 0.0}
    incoherent = 0
    coh_total = 0.0
    lip_total = 0.0
    for row in rows:
        coh = float(row.get("direction_coherence_mean") or 0.0)
        lip = float(row.get("direction_lipschitz_proxy") or 0.0)
        coh_total += coh
        lip_total += lip
        if coh < coherence_threshold or lip > lipschitz_threshold:
            incoherent += 1
    return {
        "count": float(count),
        "incoherent": float(incoherent),
        "frac": float(incoherent) / max(float(count), 1.0),
        "coh": coh_total / max(float(count), 1.0),
        "lip": lip_total / max(float(count), 1.0),
    }


def _sensitivity_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for top_fraction in args.top_fractions:
        selected = _selected(rows, top_fraction)
        for coherence_threshold in args.coherence_thresholds:
            for lipschitz_threshold in args.lipschitz_thresholds:
                stats = _stats(selected, coherence_threshold, lipschitz_threshold)
                out.append(
                    {
                        "top_fraction": _fmt(top_fraction),
                        "coherence_threshold": _fmt(coherence_threshold),
                        "lipschitz_threshold": _fmt(lipschitz_threshold),
                        "selected_count": str(int(stats["count"])),
                        "incoherent_count": str(int(stats["incoherent"])),
                        "incoherent_fraction": _fmt(stats["frac"]),
                        "direction_coherence_mean": _fmt(stats["coh"]),
                        "direction_lipschitz_proxy_mean": _fmt(stats["lip"]),
                        "robustness_label": _label(stats["frac"], float(args.incoherent_fraction_threshold)),
                    }
                )
    return out


def _group_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run"])].append(row)
    out: list[dict[str, Any]] = []
    coherence_threshold = float(args.coherence_thresholds[min(len(args.coherence_thresholds) - 1, 2)])
    lipschitz_threshold = float(args.lipschitz_thresholds[0])
    for run, group in sorted(by_run.items()):
        n, seed = _run_meta(run)
        stats = _stats(group, coherence_threshold, lipschitz_threshold)
        out.append(
            {
                "run": run,
                "N": n,
                "seed": seed,
                "selected_count": str(int(stats["count"])),
                "incoherent_count": str(int(stats["incoherent"])),
                "incoherent_fraction": _fmt(stats["frac"]),
                "direction_coherence_mean": _fmt(stats["coh"]),
                "direction_lipschitz_proxy_mean": _fmt(stats["lip"]),
                "group_route_label": _label(stats["frac"], float(args.incoherent_fraction_threshold)),
            }
        )
    return out


def _route(sensitivity: list[dict[str, Any]], groups: list[dict[str, Any]], args: argparse.Namespace) -> str:
    if not sensitivity or not groups:
        return "DIRECTION_COHERENCE_ROBUSTNESS_INCONCLUSIVE"
    blocked = [row for row in sensitivity if str(row["robustness_label"]) == "direction_incoherent_concentration_blocked"]
    blocked_frac = len(blocked) / max(len(sensitivity), 1)
    group_blocked = [row for row in groups if str(row["group_route_label"]) == "direction_incoherent_concentration_blocked"]
    if blocked_frac >= 0.75 and len(group_blocked) == len(groups):
        return "DIRECTION_COHERENCE_INCOHERENCE_ROBUST_ON_AVAILABLE_DATA"
    if blocked_frac <= 0.25:
        return "DIRECTION_COHERENCE_CFM_ROUTE_SENSITIVE_ALIVE"
    return "DIRECTION_COHERENCE_ROBUSTNESS_MIXED"


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    rows = _read_rows(args.sprint61_csv)
    sensitivity = _sensitivity_rows(rows, args)
    groups = _group_rows(rows, args)
    route = _route(sensitivity, groups, args)
    n_values = sorted({_run_meta(str(row["run"]))[0] for row in rows})
    dense_available = any("dense" in str(row["run"]).lower() or "save" in str(row["run"]).lower() for row in rows)
    summary: dict[str, Any] = {
        "contract": "ns_sprint62_direction_coherence_robustness_artifact",
        "diagnostic_mode": "sprint62_direction_coherence_threshold_group_robustness",
        "sprint61_csv": str(args.sprint61_csv),
        "packet_row_count": len(rows),
        "sensitivity_row_count": len(sensitivity),
        "group_row_count": len(groups),
        "top_fractions": [float(x) for x in args.top_fractions],
        "coherence_thresholds": [float(x) for x in args.coherence_thresholds],
        "lipschitz_thresholds": [float(x) for x in args.lipschitz_thresholds],
        "incoherent_fraction_threshold": float(args.incoherent_fraction_threshold),
        "available_N_values": n_values,
        "n128_available": "128" in n_values,
        "dense_cadence_available": dense_available,
        "robustness_status": route,
        "route_decision": route,
        "cfm_direction_regularity_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT62_DIRECTION_COHERENCE_ROBUSTNESS_DIAGNOSTIC",
        "boundary": (
            "Sprint 62 is a post-hoc robustness audit over Sprint 61 packet "
            "anatomy. It does not replace dense-cadence or higher-resolution "
            "truth runs and does not prove continuum CFM direction regularity."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_path = args.out_dir / "ns_direction_coherence_sensitivity.csv"
    group_path = args.out_dir / "ns_direction_coherence_by_run.csv"
    summary_path = args.out_dir / "ns_sprint62_direction_coherence_robustness_summary.json"
    summary["ns_direction_coherence_sensitivity_path"] = str(sensitivity_path)
    summary["ns_direction_coherence_by_run_path"] = str(group_path)
    _write_csv(sensitivity_path, SENSITIVITY_FIELDS, sensitivity)
    _write_csv(group_path, GROUP_FIELDS, groups)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ns_sprint62_direction_coherence_robustness_audit] wrote {sensitivity_path}")
    print(f"[ns_sprint62_direction_coherence_robustness_audit] wrote {group_path}")
    print(f"[ns_sprint62_direction_coherence_robustness_audit] wrote {summary_path}")
    print(
        "[ns_sprint62_direction_coherence_robustness_audit] "
        f"route={route} n128={summary['n128_available']} dense={summary['dense_cadence_available']}"
    )


if __name__ == "__main__":
    main()
