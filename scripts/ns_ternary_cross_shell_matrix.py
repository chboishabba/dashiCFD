#!/usr/bin/env python3
"""Sprint 50 full ternary cross-shell matrix producer.

This consumes Sprint 49 ``ns_material_parent_table.csv`` artifacts and derives
the observed ternary parent-state to child-state transition matrix without
rerunning packet matching or truth diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EPS = 1e-30
STATES = ("minus", "zero", "plus")
SOURCE_KINDS = (
    "same_shell",
    "adjacent_shell",
    "cross_shell",
    "low_shell_injection",
    "tracking_uncertain",
    "true_new",
)
MATRIX_FIELDS = [
    "source_kind",
    "child_state",
    "parent_state",
    "transition_mass",
    "weighted_child_mass",
    "row_count",
]
DECOMPOSITION_FIELDS = [
    "source_kind",
    "parent_state",
    "child_state",
    "transition_mass",
    "weighted_child_mass",
    "bt_distance_proxy_min",
    "bt_distance_proxy_max",
    "bt_distance_proxy_mass_weighted_mean",
    "row_count",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Sprint 49 output directories containing ns_material_parent_table.csv and ns_material_parent_summary.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--subcritical-threshold",
        type=float,
        default=0.0,
        help="diagnostic route threshold for total classified plus cross/low weighted mass",
    )
    return parser.parse_args()


def _num(value: str | float | int | None) -> float:
    try:
        out = float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _int(value: str | int | None) -> int:
    try:
        return int(float(value)) if value not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


def _source_kind(parent_relation: str, k_parent: int, k_child: int) -> str:
    delta = abs(int(k_child) - int(k_parent))
    relation = str(parent_relation)
    if relation == "tracking_uncertain":
        return "tracking_uncertain"
    if relation == "true_new":
        return "true_new"
    if relation == "low_shell_parent":
        return "low_shell_injection"
    if relation == "cross_shell_parent" or delta > 1:
        return "cross_shell"
    if relation in {"advected_parent", "split_parent", "merge_parent"} and delta == 0:
        return "same_shell"
    if delta == 1 and relation not in {"true_new", "tracking_uncertain"}:
        return "adjacent_shell"
    if delta == 0:
        return "same_shell"
    return "cross_shell"


def _require_input_dir(path: Path) -> tuple[Path, dict[str, Any]]:
    table = path / "ns_material_parent_table.csv"
    summary = path / "ns_material_parent_summary.json"
    if not table.exists():
        raise SystemExit(f"{path} lacks ns_material_parent_table.csv")
    if not summary.exists():
        raise SystemExit(f"{path} lacks ns_material_parent_summary.json")
    try:
        meta = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{summary} is not valid JSON: {exc}") from exc
    return table, meta


def _read_rows(inputs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    required = {
        "time",
        "K_parent",
        "K_child",
        "child_state",
        "parent_state",
        "credited_mass",
        "parent_relation",
    }
    for input_dir in inputs:
        table, meta = _require_input_dir(input_dir)
        manifests.append(
            {
                "input_dir": str(input_dir),
                "summary_contract": meta.get("contract"),
                "summary_row_count": meta.get("summary_row_count"),
                "table_row_count": meta.get("table_row_count"),
                "material_parent_route_status": meta.get("material_parent_route_status"),
            }
        )
        with table.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(required.difference(reader.fieldnames or []))
            if missing:
                raise SystemExit(f"{table} is missing material parent table columns: {', '.join(missing)}")
            for raw in reader:
                k_parent = _int(raw.get("K_parent"))
                k_child = _int(raw.get("K_child"))
                mass = _num(raw.get("credited_mass"))
                child_state = str(raw.get("child_state", ""))
                parent_state = str(raw.get("parent_state", ""))
                kind = _source_kind(str(raw.get("parent_relation", "")), k_parent, k_child)
                rows.append(
                    {
                        **raw,
                        "input_dir": str(input_dir),
                        "K_parent_int": k_parent,
                        "K_child_int": k_child,
                        "child_state_norm": child_state,
                        "parent_state_norm": parent_state,
                        "credited_mass_float": mass,
                        "weighted_child_mass_float": (2.0 ** (0.5 * float(k_child))) * mass,
                        "source_kind": kind,
                        "BT_distance_proxy": abs(k_child - k_parent),
                    }
                )
    return rows, manifests


def _empty_stats() -> dict[str, float | int]:
    return {
        "transition_mass": 0.0,
        "weighted_child_mass": 0.0,
        "row_count": 0,
        "bt_mass_distance": 0.0,
        "bt_min": math.inf,
        "bt_max": -math.inf,
    }


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    matrix: dict[tuple[str, str, str], dict[str, float | int]] = {}
    decomp: dict[tuple[str, str, str], dict[str, float | int]] = {}
    plus_kind_parent: dict[tuple[str, str], float] = {}
    plus_kind_parent_weighted: dict[tuple[str, str], float] = {}
    child_state_weighted: dict[str, float] = {state: 0.0 for state in STATES}
    bt_values: list[int] = []

    for kind in ("all", *SOURCE_KINDS):
        for child_state in STATES:
            for parent_state in STATES:
                matrix[(kind, child_state, parent_state)] = _empty_stats()
    for kind in SOURCE_KINDS:
        for child_state in STATES:
            for parent_state in STATES:
                decomp[(kind, parent_state, child_state)] = _empty_stats()

    for row in rows:
        child_state = row["child_state_norm"]
        parent_state = row["parent_state_norm"]
        if child_state not in STATES or parent_state not in STATES:
            continue
        kind = row["source_kind"]
        mass = float(row["credited_mass_float"])
        weighted = float(row["weighted_child_mass_float"])
        bt = int(row["BT_distance_proxy"])
        bt_values.append(bt)
        child_state_weighted[child_state] += weighted
        for source_kind in ("all", kind):
            stats = matrix[(source_kind, child_state, parent_state)]
            stats["transition_mass"] = float(stats["transition_mass"]) + mass
            stats["weighted_child_mass"] = float(stats["weighted_child_mass"]) + weighted
            stats["row_count"] = int(stats["row_count"]) + 1
        dstats = decomp[(kind, parent_state, child_state)]
        dstats["transition_mass"] = float(dstats["transition_mass"]) + mass
        dstats["weighted_child_mass"] = float(dstats["weighted_child_mass"]) + weighted
        dstats["row_count"] = int(dstats["row_count"]) + 1
        dstats["bt_mass_distance"] = float(dstats["bt_mass_distance"]) + mass * bt
        dstats["bt_min"] = min(float(dstats["bt_min"]), float(bt))
        dstats["bt_max"] = max(float(dstats["bt_max"]), float(bt))
        if child_state == "plus":
            plus_kind_parent[(kind, parent_state)] = plus_kind_parent.get((kind, parent_state), 0.0) + mass
            plus_kind_parent_weighted[(kind, parent_state)] = plus_kind_parent_weighted.get((kind, parent_state), 0.0) + weighted

    matrix_rows: list[dict[str, Any]] = []
    for kind in ("all", *SOURCE_KINDS):
        for child_state in STATES:
            for parent_state in STATES:
                stats = matrix[(kind, child_state, parent_state)]
                matrix_rows.append(
                    {
                        "source_kind": kind,
                        "child_state": child_state,
                        "parent_state": parent_state,
                        "transition_mass": f"{float(stats['transition_mass']):.17g}",
                        "weighted_child_mass": f"{float(stats['weighted_child_mass']):.17g}",
                        "row_count": str(int(stats["row_count"])),
                    }
                )

    decomp_rows: list[dict[str, Any]] = []
    for kind in SOURCE_KINDS:
        for parent_state in STATES:
            for child_state in STATES:
                stats = decomp[(kind, parent_state, child_state)]
                mass = float(stats["transition_mass"])
                row_count = int(stats["row_count"])
                mean = float(stats["bt_mass_distance"]) / max(mass, EPS)
                decomp_rows.append(
                    {
                        "source_kind": kind,
                        "parent_state": parent_state,
                        "child_state": child_state,
                        "transition_mass": f"{mass:.17g}",
                        "weighted_child_mass": f"{float(stats['weighted_child_mass']):.17g}",
                        "bt_distance_proxy_min": "" if row_count == 0 else f"{float(stats['bt_min']):.17g}",
                        "bt_distance_proxy_max": "" if row_count == 0 else f"{float(stats['bt_max']):.17g}",
                        "bt_distance_proxy_mass_weighted_mean": "" if row_count == 0 else f"{mean:.17g}",
                        "row_count": str(row_count),
                    }
                )

    cross_weighted = {
        state: plus_kind_parent_weighted.get(("cross_shell", state), 0.0)
        for state in STATES
    }
    low_weighted = {
        state: plus_kind_parent_weighted.get(("low_shell_injection", state), 0.0)
        for state in STATES
    }
    uncertain_weighted = sum(plus_kind_parent_weighted.get(("tracking_uncertain", state), 0.0) for state in STATES)
    cross_low_total = sum(cross_weighted.values()) + sum(low_weighted.values())
    cross_plus_total = sum(cross_weighted.values())
    dominant_red_source_state = max(STATES, key=lambda state: cross_weighted[state])
    source_kind_totals = {
        kind: sum(plus_kind_parent_weighted.get((kind, state), 0.0) for state in STATES)
        for kind in SOURCE_KINDS
    }
    dominant_red_source_kind = max(source_kind_totals, key=source_kind_totals.get)
    sigma_cross = {
        state: cross_weighted[state] / max(cross_plus_total, EPS)
        for state in STATES
    }
    summary = {
        "contract": "ns_ternary_cross_shell_artifact",
        "diagnostic_mode": "sprint50_full_ternary_cross_shell_from_material_parent_table",
        "matrix_entry_count": len(matrix_rows),
        "source_decomposition_entry_count": len(decomp_rows),
        "input_table_row_count": len(rows),
        "ternary_states": list(STATES),
        "source_kinds": list(SOURCE_KINDS),
        "BT_distance_proxy": "abs(K_child - K_parent)",
        "BT_distance_proxy_boundary": "diagnostic proxy only; not theorem-grade BT metric",
        "classification_field_used_for_source_kind": False,
        "weighted_cross_plus_from_minus": cross_weighted["minus"],
        "weighted_cross_plus_from_zero": cross_weighted["zero"],
        "weighted_cross_plus_from_plus": cross_weighted["plus"],
        "weighted_low_shell_plus_from_minus": low_weighted["minus"],
        "weighted_low_shell_plus_from_zero": low_weighted["zero"],
        "weighted_low_shell_plus_from_plus": low_weighted["plus"],
        "weighted_tracking_uncertain_plus": uncertain_weighted,
        "weighted_cross_low_plus_total": cross_low_total,
        "sigma_cross_from_minus": sigma_cross["minus"],
        "sigma_cross_from_zero": sigma_cross["zero"],
        "sigma_cross_from_plus": sigma_cross["plus"],
        "dominant_red_source_state": dominant_red_source_state,
        "dominant_red_source_kind": dominant_red_source_kind,
        "weighted_child_plus_total": child_state_weighted["plus"],
        "weighted_child_zero_total": child_state_weighted["zero"],
        "weighted_child_minus_total": child_state_weighted["minus"],
        "BT_distance_proxy_min": min(bt_values) if bt_values else None,
        "BT_distance_proxy_max": max(bt_values) if bt_values else None,
        "full_ternary_matrix_available": True,
        "adjacent_only_theorem_sufficient": False,
        "cross_shell_summability_proved": False,
        "bt_distance_decay_theorem_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "promotion_status": "NO_PROMOTION_TERNARY_CROSS_SHELL_DIAGNOSTIC",
    }
    return matrix_rows, decomp_rows, summary


def _route_decision(summary: dict[str, Any], threshold: float) -> str:
    if float(summary["weighted_tracking_uncertain_plus"]) > float(summary["weighted_cross_low_plus_total"]):
        return "TRACKING_UNCERTAIN_DOMINATES"
    low_total = (
        float(summary["weighted_low_shell_plus_from_minus"])
        + float(summary["weighted_low_shell_plus_from_zero"])
        + float(summary["weighted_low_shell_plus_from_plus"])
    )
    cross_total = (
        float(summary["weighted_cross_plus_from_minus"])
        + float(summary["weighted_cross_plus_from_zero"])
        + float(summary["weighted_cross_plus_from_plus"])
    )
    if low_total > cross_total:
        return "LOW_SHELL_PLUS_INJECTION_DOMINATES"
    if cross_total <= threshold:
        return "TERNARY_CROSS_SHELL_SUBCRITICAL_DIAGNOSTIC"
    candidates = {
        "CROSS_PLUS_FROM_MINUS_DOMINATES": float(summary["weighted_cross_plus_from_minus"]),
        "CROSS_PLUS_FROM_ZERO_DOMINATES": float(summary["weighted_cross_plus_from_zero"]),
        "CROSS_PLUS_FROM_PLUS_DOMINATES": float(summary["weighted_cross_plus_from_plus"]),
    }
    return max(candidates, key=candidates.get)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    rows, input_manifests = _read_rows(args.inputs)
    matrix_rows, decomposition_rows, summary = _aggregate(rows)
    summary.update(
        {
            "inputs": [str(path) for path in args.inputs],
            "input_manifest_summaries": input_manifests,
            "subcritical_threshold": float(args.subcritical_threshold),
        }
    )
    summary["route_decision"] = _route_decision(summary, float(args.subcritical_threshold))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.out_dir / "ns_full_ternary_transition_matrix.csv"
    decomp_path = args.out_dir / "ns_cross_shell_source_decomposition.csv"
    summary_path = args.out_dir / "ns_ternary_cross_shell_summary.json"
    summary["ns_full_ternary_transition_matrix_path"] = str(matrix_path)
    summary["ns_cross_shell_source_decomposition_path"] = str(decomp_path)
    summary["receipt_alignment"] = "DASHI.Physics.Closure.ClaySprintFiftyFullTernaryCrossShellAuditReceipt"

    _write_csv(matrix_path, MATRIX_FIELDS, matrix_rows)
    _write_csv(decomp_path, DECOMPOSITION_FIELDS, decomposition_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_ternary_cross_shell_matrix] wrote {matrix_path}")
    print(f"[ns_ternary_cross_shell_matrix] wrote {decomp_path}")
    print(f"[ns_ternary_cross_shell_matrix] wrote {summary_path}")
    print(
        "[ns_ternary_cross_shell_matrix] "
        f"route={summary['route_decision']} "
        f"dominant_state={summary['dominant_red_source_state']} "
        f"dominant_kind={summary['dominant_red_source_kind']}"
    )


if __name__ == "__main__":
    main()
