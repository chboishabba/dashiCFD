#!/usr/bin/env python3
"""Sprint 63 cross-shell replenishment contractivity audit.

Sprint 60 left raw positive stretch action nearly flat under Euclidean,
smoothed, and provisional BT shell assignments. Sprint 61/62 then blocked the
immediate CFM coherent-tube rescue on available data. Sprint 63 tests the
remaining DASHI-native diagnostic fork:

    can adjacent/cross-shell parent credit replenish raw-red action
    without amplifying the parent's available raw-action budget?

The producer joins Sprint 49 material-parent edges with Sprint 59 raw
packet-action rows. For each adjacent/cross-shell edge, it compares the child
packet raw positive action with the credited fraction of the parent packet raw
positive action. This is a diagnostic contractivity surface only; it does not
prove support non-creation, defect monotonicity, stretch absorption, no blowup,
or any Clay/NS promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import ns_sprint53_no2cycle_physical_amplitude_audit as sprint53
import ns_ternary_cross_shell_matrix as sprint50


EPS = 1e-30

EDGE_FIELDS = [
    "run",
    "time",
    "dt",
    "parent_time",
    "parent_lookup_status",
    "K_parent",
    "K_child",
    "shell_delta",
    "source_kind",
    "parent_relation",
    "parent_state",
    "child_state",
    "parent_packet_id",
    "child_packet_id",
    "parent_mass",
    "child_mass",
    "credited_mass",
    "credit_fraction",
    "parent_confidence",
    "child_A_raw_positive",
    "parent_A_raw_positive",
    "parent_action_budget",
    "child_weighted_A_raw_positive",
    "weighted_parent_action_budget",
    "contractivity_ratio",
    "weighted_contractivity_ratio",
    "amplification_excess",
    "support_created_fraction_proxy",
    "contractive",
    "contractivity_route_label",
]

BY_K_FIELDS = [
    "K_child",
    "edge_count",
    "available_parent_action_edge_count",
    "contractive_edge_count",
    "noncontractive_edge_count",
    "contractive_edge_fraction",
    "child_A_raw_positive_total",
    "parent_action_budget_total",
    "contractivity_ratio_total",
    "weighted_child_A_raw_positive_total",
    "weighted_parent_action_budget_total",
    "weighted_contractivity_ratio_total",
    "support_created_fraction_proxy_mean",
    "dominant_route_label",
]

BY_TRANSITION_FIELDS = [
    "source_kind",
    "parent_state",
    "child_state",
    "edge_count",
    "available_parent_action_edge_count",
    "contractive_edge_count",
    "contractive_edge_fraction",
    "child_A_raw_positive_total",
    "parent_action_budget_total",
    "contractivity_ratio_total",
    "weighted_child_A_raw_positive_total",
    "weighted_parent_action_budget_total",
    "weighted_contractivity_ratio_total",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Sprint 49 material-parent output directories")
    p.add_argument("--raw-action-csv", type=Path, required=True, help="Sprint 59 ns_raw_packet_stretch_action.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--contractivity-threshold", type=float, default=1.0)
    p.add_argument("--mixed-noncontractive-fraction", type=float, default=0.05)
    p.add_argument("--include-same-shell", action="store_true", help="include same-shell rows as controls")
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _num(value: str | float | int | None) -> float:
    return sprint50._num(value)


def _source_kind(row: dict[str, Any]) -> str:
    return sprint50._source_kind(
        str(row.get("parent_relation", "")),
        int(row.get("K_parent_int") or 0),
        int(row.get("K_child_int") or 0),
    )


def _time_key(value: float) -> str:
    return _fmt(round(float(value), 12))


def _read_raw_action(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "run",
            "time",
            "K",
            "packet_id",
            "A_raw_positive",
            "weighted_A_raw_positive",
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
        out: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in reader:
            key = (str(row.get("run", "")), _time_key(_num(row.get("time"))), str(row.get("packet_id", "")))
            out[key] = row
    return out


def _lookup_raw(
    raw_by_key: dict[tuple[str, str, str], dict[str, str]],
    run: str,
    time: float,
    packet_id: str,
) -> tuple[dict[str, str] | None, str]:
    exact = raw_by_key.get((run, _time_key(time), packet_id))
    if exact is not None:
        return exact, "parent_previous_time"
    return None, "parent_action_unavailable"


def _lookup_parent_raw(
    raw_by_key: dict[tuple[str, str, str], dict[str, str]],
    run: str,
    child_time: float,
    dt: float,
    parent_packet_id: str,
) -> tuple[dict[str, str] | None, float, str]:
    parent_time = child_time - dt
    row, status = _lookup_raw(raw_by_key, run, parent_time, parent_packet_id)
    if row is not None:
        return row, parent_time, status
    same_time = raw_by_key.get((run, _time_key(child_time), parent_packet_id))
    if same_time is not None:
        return same_time, child_time, "parent_same_time_fallback"
    return None, parent_time, "parent_action_unavailable"


def _edge_label(row: dict[str, Any], threshold: float) -> str:
    if row["parent_lookup_status"] == "parent_action_unavailable":
        return "PARENT_ACTION_UNAVAILABLE"
    if str(row["contractive"]) == "true":
        return "CONTRACTIVE"
    if float(row["support_created_fraction_proxy"]) > 0.0:
        return "NONCONTRACTIVE_WITH_SUPPORT_GAP"
    return "NONCONTRACTIVE"


def _build_edges(
    table_rows: list[dict[str, Any]],
    raw_by_key: dict[tuple[str, str, str], dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in table_rows:
        source_kind = _source_kind(row)
        if source_kind == "same_shell" and not bool(args.include_same_shell):
            continue
        if source_kind in {"true_new", "tracking_uncertain"}:
            continue
        if source_kind not in {"adjacent_shell", "cross_shell", "low_shell_injection", "same_shell"}:
            continue

        run = str(row["run"])
        child_time = float(row["time_float"])
        dt = float(row["dt_float"])
        child_id = str(row.get("child_packet_id", ""))
        parent_id = str(row.get("parent_packet_id", ""))
        child_raw = raw_by_key.get((run, _time_key(child_time), child_id))
        if child_raw is None:
            continue
        parent_raw, parent_time, lookup_status = _lookup_parent_raw(raw_by_key, run, child_time, dt, parent_id)

        child_positive = _num(child_raw.get("A_raw_positive"))
        child_weighted = _num(child_raw.get("weighted_A_raw_positive"))
        parent_positive = _num(parent_raw.get("A_raw_positive")) if parent_raw is not None else 0.0
        parent_weighted = _num(parent_raw.get("weighted_A_raw_positive")) if parent_raw is not None else 0.0
        parent_mass = _num(row.get("parent_mass"))
        child_mass = _num(row.get("child_mass"))
        credited_mass = _num(row.get("credited_mass"))
        credit_fraction = min(max(credited_mass / (parent_mass + EPS), 0.0), 1.0)
        parent_budget = parent_positive * credit_fraction
        weighted_parent_budget = parent_weighted * credit_fraction
        ratio = child_positive / (parent_budget + EPS)
        weighted_ratio = child_weighted / (weighted_parent_budget + EPS)
        contractive = lookup_status != "parent_action_unavailable" and ratio <= float(args.contractivity_threshold)
        support_created = max(child_mass - credited_mass, 0.0) / (child_mass + EPS)
        out = {
            "run": run,
            "time": _fmt(child_time),
            "dt": _fmt(dt),
            "parent_time": _fmt(parent_time),
            "parent_lookup_status": lookup_status,
            "K_parent": str(int(row.get("K_parent_int") or 0)),
            "K_child": str(int(row.get("K_child_int") or 0)),
            "shell_delta": str(abs(int(row.get("K_child_int") or 0) - int(row.get("K_parent_int") or 0))),
            "source_kind": source_kind,
            "parent_relation": str(row.get("parent_relation", "")),
            "parent_state": str(row.get("parent_state", "")),
            "child_state": str(row.get("child_state", "")),
            "parent_packet_id": parent_id,
            "child_packet_id": child_id,
            "parent_mass": _fmt(parent_mass),
            "child_mass": _fmt(child_mass),
            "credited_mass": _fmt(credited_mass),
            "credit_fraction": _fmt(credit_fraction),
            "parent_confidence": _fmt(float(row.get("parent_confidence_float") or 0.0)),
            "child_A_raw_positive": _fmt(child_positive),
            "parent_A_raw_positive": _fmt(parent_positive),
            "parent_action_budget": _fmt(parent_budget),
            "child_weighted_A_raw_positive": _fmt(child_weighted),
            "weighted_parent_action_budget": _fmt(weighted_parent_budget),
            "contractivity_ratio": _fmt(ratio),
            "weighted_contractivity_ratio": _fmt(weighted_ratio),
            "amplification_excess": _fmt(max(child_positive - parent_budget, 0.0)),
            "support_created_fraction_proxy": _fmt(support_created),
            "contractive": "true" if contractive else "false",
        }
        out["contractivity_route_label"] = _edge_label(out, float(args.contractivity_threshold))
        rows.append(out)
    return rows


def _empty_stats() -> dict[str, float]:
    return {
        "edge_count": 0.0,
        "available": 0.0,
        "contractive": 0.0,
        "child": 0.0,
        "parent_budget": 0.0,
        "weighted_child": 0.0,
        "weighted_parent_budget": 0.0,
        "support_gap": 0.0,
    }


def _add(stats: dict[str, float], row: dict[str, Any]) -> None:
    stats["edge_count"] += 1.0
    available = row["parent_lookup_status"] != "parent_action_unavailable"
    stats["available"] += 1.0 if available else 0.0
    stats["contractive"] += 1.0 if str(row["contractive"]) == "true" else 0.0
    stats["child"] += float(row["child_A_raw_positive"])
    stats["parent_budget"] += float(row["parent_action_budget"])
    stats["weighted_child"] += float(row["child_weighted_A_raw_positive"])
    stats["weighted_parent_budget"] += float(row["weighted_parent_action_budget"])
    stats["support_gap"] += float(row["support_created_fraction_proxy"])


def _stats_route(stats: dict[str, float], threshold: float, mixed_fraction: float) -> str:
    if stats["edge_count"] <= 0.0:
        return "NO_EDGES"
    if stats["available"] <= 0.0:
        return "PARENT_ACTION_UNAVAILABLE"
    noncontractive = max(stats["available"] - stats["contractive"], 0.0)
    frac = noncontractive / max(stats["available"], 1.0)
    ratio = stats["child"] / (stats["parent_budget"] + EPS)
    if frac <= mixed_fraction and ratio <= threshold:
        return "CONTRACTIVE"
    if frac < 1.0:
        return "MIXED"
    return "NONCONTRACTIVE"


def _build_by_k(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_k: dict[int, dict[str, float]] = defaultdict(_empty_stats)
    for row in rows:
        _add(by_k[int(row["K_child"])], row)
    out: list[dict[str, Any]] = []
    for k, stats in sorted(by_k.items()):
        available = int(stats["available"])
        contractive = int(stats["contractive"])
        out.append(
            {
                "K_child": str(k),
                "edge_count": str(int(stats["edge_count"])),
                "available_parent_action_edge_count": str(available),
                "contractive_edge_count": str(contractive),
                "noncontractive_edge_count": str(max(available - contractive, 0)),
                "contractive_edge_fraction": _fmt(contractive / max(available, 1)),
                "child_A_raw_positive_total": _fmt(stats["child"]),
                "parent_action_budget_total": _fmt(stats["parent_budget"]),
                "contractivity_ratio_total": _fmt(stats["child"] / (stats["parent_budget"] + EPS)),
                "weighted_child_A_raw_positive_total": _fmt(stats["weighted_child"]),
                "weighted_parent_action_budget_total": _fmt(stats["weighted_parent_budget"]),
                "weighted_contractivity_ratio_total": _fmt(stats["weighted_child"] / (stats["weighted_parent_budget"] + EPS)),
                "support_created_fraction_proxy_mean": _fmt(stats["support_gap"] / max(stats["edge_count"], 1.0)),
                "dominant_route_label": _stats_route(
                    stats,
                    float(args.contractivity_threshold),
                    float(args.mixed_noncontractive_fraction),
                ),
            }
        )
    return out


def _build_by_transition(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_t: dict[tuple[str, str, str], dict[str, float]] = defaultdict(_empty_stats)
    for row in rows:
        key = (str(row["source_kind"]), str(row["parent_state"]), str(row["child_state"]))
        _add(by_t[key], row)
    out: list[dict[str, Any]] = []
    for key, stats in sorted(by_t.items()):
        source_kind, parent_state, child_state = key
        available = int(stats["available"])
        contractive = int(stats["contractive"])
        out.append(
            {
                "source_kind": source_kind,
                "parent_state": parent_state,
                "child_state": child_state,
                "edge_count": str(int(stats["edge_count"])),
                "available_parent_action_edge_count": str(available),
                "contractive_edge_count": str(contractive),
                "contractive_edge_fraction": _fmt(contractive / max(available, 1)),
                "child_A_raw_positive_total": _fmt(stats["child"]),
                "parent_action_budget_total": _fmt(stats["parent_budget"]),
                "contractivity_ratio_total": _fmt(stats["child"] / (stats["parent_budget"] + EPS)),
                "weighted_child_A_raw_positive_total": _fmt(stats["weighted_child"]),
                "weighted_parent_action_budget_total": _fmt(stats["weighted_parent_budget"]),
                "weighted_contractivity_ratio_total": _fmt(stats["weighted_child"] / (stats["weighted_parent_budget"] + EPS)),
            }
        )
    return out


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows)


def _route(summary: dict[str, Any], args: argparse.Namespace) -> str:
    if int(summary["edge_count"]) == 0:
        return "CROSS_SHELL_REPLENISHMENT_NO_EDGES"
    if int(summary["available_parent_action_edge_count"]) == 0:
        return "CROSS_SHELL_REPLENISHMENT_SOURCE_UNAVAILABLE"
    nonfrac = float(summary["noncontractive_edge_fraction"])
    ratio = float(summary["contractivity_ratio_total"])
    threshold = float(args.contractivity_threshold)
    mixed = float(args.mixed_noncontractive_fraction)
    if nonfrac <= mixed and ratio <= threshold:
        return "CROSS_SHELL_REPLENISHMENT_CONTRACTIVE_ON_AVAILABLE_DATA"
    if nonfrac < 1.0:
        return "CROSS_SHELL_REPLENISHMENT_MIXED"
    return "CROSS_SHELL_REPLENISHMENT_NONCONTRACTIVE_BLOCKED"


def _build_summary(
    edge_rows: list[dict[str, Any]],
    by_k_rows: list[dict[str, Any]],
    by_transition_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    available = sum(1 for row in edge_rows if row["parent_lookup_status"] != "parent_action_unavailable")
    contractive = sum(1 for row in edge_rows if str(row["contractive"]) == "true")
    edge_count = len(edge_rows)
    child_total = _sum(edge_rows, "child_A_raw_positive")
    parent_total = _sum(edge_rows, "parent_action_budget")
    weighted_child_total = _sum(edge_rows, "child_weighted_A_raw_positive")
    weighted_parent_total = _sum(edge_rows, "weighted_parent_action_budget")
    summary: dict[str, Any] = {
        "contract": "ns_sprint63_cross_shell_replenishment_contractivity_artifact",
        "diagnostic_mode": "sprint63_cross_shell_parent_credit_contractivity",
        "edge_count": edge_count,
        "by_k_row_count": len(by_k_rows),
        "by_transition_row_count": len(by_transition_rows),
        "available_parent_action_edge_count": available,
        "missing_parent_action_edge_count": edge_count - available,
        "contractive_edge_count": contractive,
        "noncontractive_edge_count": max(available - contractive, 0),
        "contractive_edge_fraction": contractive / max(available, 1),
        "noncontractive_edge_fraction": max(available - contractive, 0) / max(available, 1),
        "child_A_raw_positive_total": child_total,
        "parent_action_budget_total": parent_total,
        "contractivity_ratio_total": child_total / (parent_total + EPS),
        "weighted_child_A_raw_positive_total": weighted_child_total,
        "weighted_parent_action_budget_total": weighted_parent_total,
        "weighted_contractivity_ratio_total": weighted_child_total / (weighted_parent_total + EPS),
        "support_created_fraction_proxy_mean": _sum(edge_rows, "support_created_fraction_proxy") / max(edge_count, 1),
        "contractivity_threshold": float(args.contractivity_threshold),
        "mixed_noncontractive_fraction": float(args.mixed_noncontractive_fraction),
        "include_same_shell": bool(args.include_same_shell),
        "contractivity_proved": False,
        "support_non_creation_proved": False,
        "defect_monotonicity_proved": False,
        "adjacent_cross_shell_replenishment_summability_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT63_CROSS_SHELL_REPLENISHMENT_DIAGNOSTIC",
        "formal_target": "AdjacentCrossShellReplenishmentSummable",
        "boundary": (
            "Sprint 63 compares child raw positive action with credited parent "
            "raw-action budgets on available Sprint 49/59 artifacts. It is an "
            "empirical contractivity surface, not a proof of support "
            "non-creation, defect monotonicity, summability, or no blowup."
        ),
    }
    summary["route_decision"] = _route(summary, args)
    return summary


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    table_rows, _summary_rows, manifests = sprint53._read_inputs(args.inputs)
    raw_by_key = _read_raw_action(args.raw_action_csv)
    edge_rows = _build_edges(table_rows, raw_by_key, args)
    by_k_rows = _build_by_k(edge_rows, args)
    by_transition_rows = _build_by_transition(edge_rows, args)
    summary = _build_summary(edge_rows, by_k_rows, by_transition_rows, args)
    summary["inputs"] = [str(path) for path in args.inputs]
    summary["raw_action_csv"] = str(args.raw_action_csv)
    summary["input_manifest_summaries"] = manifests

    args.out_dir.mkdir(parents=True, exist_ok=True)
    edge_path = args.out_dir / "ns_cross_shell_replenishment_contractivity.csv"
    by_k_path = args.out_dir / "ns_cross_shell_replenishment_contractivity_by_k.csv"
    by_transition_path = args.out_dir / "ns_cross_shell_replenishment_contractivity_by_transition.csv"
    summary_path = args.out_dir / "ns_sprint63_cross_shell_replenishment_contractivity_summary.json"
    summary["ns_cross_shell_replenishment_contractivity_path"] = str(edge_path)
    summary["ns_cross_shell_replenishment_contractivity_by_k_path"] = str(by_k_path)
    summary["ns_cross_shell_replenishment_contractivity_by_transition_path"] = str(by_transition_path)

    _write_csv(edge_path, EDGE_FIELDS, edge_rows)
    _write_csv(by_k_path, BY_K_FIELDS, by_k_rows)
    _write_csv(by_transition_path, BY_TRANSITION_FIELDS, by_transition_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint63_cross_shell_replenishment_contractivity_audit] wrote {edge_path}")
    print(f"[ns_sprint63_cross_shell_replenishment_contractivity_audit] wrote {by_k_path}")
    print(f"[ns_sprint63_cross_shell_replenishment_contractivity_audit] wrote {by_transition_path}")
    print(f"[ns_sprint63_cross_shell_replenishment_contractivity_audit] wrote {summary_path}")
    print(
        "[ns_sprint63_cross_shell_replenishment_contractivity_audit] "
        f"route={summary['route_decision']} "
        f"ratio={summary['contractivity_ratio_total']} "
        f"noncontractive_fraction={summary['noncontractive_edge_fraction']} "
        f"promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
