#!/usr/bin/env python3
"""Sprint 51 signed ternary flip audit.

This consumes Sprint 49 ``ns_material_parent_table.csv`` artifacts and audits
cross-shell minus/plus flow as an involutive signed-flip channel rather than a
raw plus-source channel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


EPS = 1e-30
STATES = ("minus", "zero", "plus")
TRANSITIONS = (
    ("minus", "plus"),
    ("plus", "minus"),
    ("minus", "zero"),
    ("plus", "zero"),
    ("zero", "plus"),
    ("zero", "minus"),
)

FLIP_BALANCE_FIELDS = [
    "K_parent",
    "K_child",
    "delta_K",
    "cross_minus_to_plus",
    "cross_plus_to_minus",
    "cross_minus_to_zero",
    "cross_plus_to_zero",
    "cross_zero_to_plus",
    "cross_zero_to_minus",
    "signed_flip_imbalance",
    "absolute_flip_imbalance",
    "net_residue_N",
    "BT_distance_proxy",
    "parent_confidence_mass_weighted_mean",
    "row_count",
]

NET_LYAPUNOV_FIELDS = [
    "time",
    "K",
    "K_next",
    "net_residue_N",
    "net_residue_N_next",
    "N_next_minus_N",
    "q_proxy",
    "source_proxy",
    "weighted_plus",
    "weighted_minus",
    "does_net_residue_decay",
]

BT_DECAY_FIELDS = [
    "BT_distance_proxy",
    "cross_minus_to_plus",
    "cross_plus_to_minus",
    "signed_flip_imbalance",
    "absolute_flip_imbalance",
    "paired_flip_flow",
    "imbalance_fraction_of_paired_flow",
    "eta_signed_flip_by_p",
    "does_signed_flip_decay",
]

NO2CYCLE_FIELDS = [
    "first_time",
    "second_time",
    "first_child_packet_id",
    "second_child_packet_id",
    "first_transition",
    "second_transition",
    "first_weighted_mass",
    "second_weighted_mass",
    "cycle_mass_ratio",
    "does_no2cycle_hold",
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
    parser.add_argument("--p", type=float, default=3.0, help="ultrametric prime lane for eta threshold")
    parser.add_argument(
        "--balance-fraction-threshold",
        type=float,
        default=0.10,
        help="max abs imbalance / paired flip flow for SIGNED_FLIP_BALANCED_ROUTE_ALIVE",
    )
    parser.add_argument(
        "--cycle-damping-threshold",
        type=float,
        default=1.0 / math.sqrt(2.0),
        help="max second/first weighted flip mass for no-2-cycle damping proxy",
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


def _fmt(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


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
        "child_packet_id",
        "parent_packet_id",
        "child_state",
        "parent_state",
        "credited_mass",
        "parent_confidence",
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
                weighted = (2.0 ** (0.5 * float(k_child))) * mass
                child_state = str(raw.get("child_state", ""))
                parent_state = str(raw.get("parent_state", ""))
                kind = _source_kind(str(raw.get("parent_relation", "")), k_parent, k_child)
                rows.append(
                    {
                        **raw,
                        "input_dir": str(input_dir),
                        "time_float": _num(raw.get("time")),
                        "K_parent_int": k_parent,
                        "K_child_int": k_child,
                        "delta_K": k_child - k_parent,
                        "child_state_norm": child_state,
                        "parent_state_norm": parent_state,
                        "credited_mass_float": mass,
                        "weighted_mass_float": weighted,
                        "source_kind": kind,
                        "BT_distance_proxy": abs(k_child - k_parent),
                        "parent_confidence_float": _num(raw.get("parent_confidence")),
                    }
                )
    return rows, manifests


def _transition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["parent_state_norm"]), str(row["child_state_norm"])


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= EPS:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def _aggregate_flip_balance(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    totals = {
        "cross_minus_to_plus": 0.0,
        "cross_plus_to_minus": 0.0,
        "cross_minus_to_zero": 0.0,
        "cross_plus_to_zero": 0.0,
        "cross_zero_to_plus": 0.0,
        "cross_zero_to_minus": 0.0,
        "weighted_plus": 0.0,
        "weighted_minus": 0.0,
    }
    for row in rows:
        if row["source_kind"] != "cross_shell":
            continue
        parent = row["parent_state_norm"]
        child = row["child_state_norm"]
        if parent not in STATES or child not in STATES:
            continue
        key = (int(row["K_parent_int"]), int(row["K_child_int"]))
        item = grouped.setdefault(
            key,
            {
                "K_parent": int(row["K_parent_int"]),
                "K_child": int(row["K_child_int"]),
                "delta_K": int(row["delta_K"]),
                "BT_distance_proxy": int(row["BT_distance_proxy"]),
                "row_count": 0,
                "confidence_mass": 0.0,
                "mass_for_confidence": 0.0,
                **{f"cross_{src}_to_{dst}": 0.0 for src, dst in TRANSITIONS},
                "weighted_plus": 0.0,
                "weighted_minus": 0.0,
            },
        )
        weighted = float(row["weighted_mass_float"])
        item["row_count"] += 1
        item["confidence_mass"] += weighted * float(row["parent_confidence_float"])
        item["mass_for_confidence"] += weighted
        if (parent, child) in TRANSITIONS:
            field = f"cross_{parent}_to_{child}"
            item[field] += weighted
            totals[field] += weighted
        if child == "plus":
            item["weighted_plus"] += weighted
            totals["weighted_plus"] += weighted
        elif child == "minus":
            item["weighted_minus"] += weighted
            totals["weighted_minus"] += weighted

    out: list[dict[str, Any]] = []
    for _, item in sorted(grouped.items()):
        m2p = float(item["cross_minus_to_plus"])
        p2m = float(item["cross_plus_to_minus"])
        signed = m2p - p2m
        absolute = abs(signed)
        net = float(item["weighted_plus"]) - float(item["weighted_minus"])
        confidence = float(item["confidence_mass"]) / max(float(item["mass_for_confidence"]), EPS)
        out.append(
            {
                "K_parent": str(item["K_parent"]),
                "K_child": str(item["K_child"]),
                "delta_K": str(item["delta_K"]),
                "cross_minus_to_plus": _fmt(m2p),
                "cross_plus_to_minus": _fmt(p2m),
                "cross_minus_to_zero": _fmt(float(item["cross_minus_to_zero"])),
                "cross_plus_to_zero": _fmt(float(item["cross_plus_to_zero"])),
                "cross_zero_to_plus": _fmt(float(item["cross_zero_to_plus"])),
                "cross_zero_to_minus": _fmt(float(item["cross_zero_to_minus"])),
                "signed_flip_imbalance": _fmt(signed),
                "absolute_flip_imbalance": _fmt(absolute),
                "net_residue_N": _fmt(net),
                "BT_distance_proxy": str(item["BT_distance_proxy"]),
                "parent_confidence_mass_weighted_mean": _fmt(confidence),
                "row_count": str(item["row_count"]),
            }
        )
    return out, totals


def _aggregate_bt_decay(rows: list[dict[str, Any]], p: float) -> tuple[list[dict[str, Any]], dict[str, float | bool]]:
    grouped: dict[int, dict[str, float]] = defaultdict(
        lambda: {"minus_to_plus": 0.0, "plus_to_minus": 0.0, "signed": 0.0, "absolute": 0.0}
    )
    for row in rows:
        if row["source_kind"] != "cross_shell":
            continue
        parent, child = _transition_key(row)
        if (parent, child) not in {("minus", "plus"), ("plus", "minus")}:
            continue
        bt = int(row["BT_distance_proxy"])
        weighted = float(row["weighted_mass_float"])
        if parent == "minus":
            grouped[bt]["minus_to_plus"] += weighted
        else:
            grouped[bt]["plus_to_minus"] += weighted

    for item in grouped.values():
        item["signed"] = item["minus_to_plus"] - item["plus_to_minus"]
        item["absolute"] = abs(item["signed"])

    xs = [float(bt) for bt, item in sorted(grouped.items()) if item["absolute"] > 0.0]
    ys = [math.log(item["absolute"]) for _, item in sorted(grouped.items()) if item["absolute"] > 0.0]
    slope = _linear_slope(xs, ys)
    eta = max(0.0, -slope / math.log(max(p, 1.000001)))
    threshold = math.log(math.sqrt(2.0)) / math.log(max(p, 1.000001))
    does_decay = eta > threshold

    out: list[dict[str, Any]] = []
    for bt, item in sorted(grouped.items()):
        paired = item["minus_to_plus"] + item["plus_to_minus"]
        out.append(
            {
                "BT_distance_proxy": str(bt),
                "cross_minus_to_plus": _fmt(item["minus_to_plus"]),
                "cross_plus_to_minus": _fmt(item["plus_to_minus"]),
                "signed_flip_imbalance": _fmt(item["signed"]),
                "absolute_flip_imbalance": _fmt(item["absolute"]),
                "paired_flip_flow": _fmt(paired),
                "imbalance_fraction_of_paired_flow": _fmt(item["absolute"] / max(paired, EPS)),
                "eta_signed_flip_by_p": _fmt(eta),
                "does_signed_flip_decay": _fmt(does_decay),
            }
        )
    return out, {
        "eta_signed_flip_by_p": eta,
        "eta_signed_flip_threshold_by_p": threshold,
        "does_signed_flip_decay": does_decay,
    }


def _aggregate_net_lyapunov(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float | bool]]:
    grouped: dict[tuple[float, int], dict[str, float]] = defaultdict(lambda: {"plus": 0.0, "minus": 0.0})
    for row in rows:
        if row["source_kind"] != "cross_shell":
            continue
        child = row["child_state_norm"]
        if child not in {"plus", "minus"}:
            continue
        key = (float(row["time_float"]), int(row["K_child_int"]))
        grouped[key][child] += float(row["weighted_mass_float"])

    by_time: dict[float, list[tuple[int, float, float, float]]] = defaultdict(list)
    for (time, shell), item in grouped.items():
        net = item["plus"] - item["minus"]
        by_time[time].append((shell, net, item["plus"], item["minus"]))

    ratios: list[float] = []
    out: list[dict[str, Any]] = []
    for time, items in sorted(by_time.items()):
        items.sort()
        for (shell, net, plus, minus), (next_shell, next_net, _, _) in zip(items, items[1:]):
            ratio = next_net / net if net > EPS else 0.0
            if net > EPS and math.isfinite(ratio):
                ratios.append(max(0.0, ratio))
            source = max(0.0, next_net - max(0.0, ratio) * net)
            does_decay = ratio * math.sqrt(2.0) < 1.0 if net > EPS else next_net <= 0.0
            out.append(
                {
                    "time": _fmt(time),
                    "K": str(shell),
                    "K_next": str(next_shell),
                    "net_residue_N": _fmt(net),
                    "net_residue_N_next": _fmt(next_net),
                    "N_next_minus_N": _fmt(next_net - net),
                    "q_proxy": _fmt(max(0.0, ratio)),
                    "source_proxy": _fmt(source),
                    "weighted_plus": _fmt(plus),
                    "weighted_minus": _fmt(minus),
                    "does_net_residue_decay": _fmt(does_decay),
                }
            )
    q_proxy = max(ratios) if ratios else 0.0
    return out, {
        "q_net_residue_proxy": q_proxy,
        "q_net_residue_times_sqrt2": q_proxy * math.sqrt(2.0),
        "does_net_residue_decay": bool(out) and q_proxy * math.sqrt(2.0) < 1.0,
    }


def _aggregate_no2cycle(rows: list[dict[str, Any]], damping_threshold: float) -> tuple[list[dict[str, Any]], dict[str, float | bool]]:
    flip_rows = [
        row
        for row in rows
        if row["source_kind"] == "cross_shell" and _transition_key(row) in {("minus", "plus"), ("plus", "minus")}
    ]
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in flip_rows:
        by_parent[(str(row["input_dir"]), str(row.get("parent_packet_id", "")))].append(row)

    out: list[dict[str, Any]] = []
    for first in flip_rows:
        first_child_key = (str(first["input_dir"]), str(first.get("child_packet_id", "")))
        parent, child = _transition_key(first)
        opposite = (child, parent)
        candidates = [
            second
            for second in by_parent.get(first_child_key, [])
            if _transition_key(second) == opposite and float(second["time_float"]) > float(first["time_float"])
        ]
        if not candidates:
            continue
        second = min(candidates, key=lambda row: float(row["time_float"]))
        first_weight = float(first["weighted_mass_float"])
        second_weight = float(second["weighted_mass_float"])
        ratio = second_weight / max(first_weight, EPS)
        holds = ratio <= damping_threshold
        out.append(
            {
                "first_time": _fmt(float(first["time_float"])),
                "second_time": _fmt(float(second["time_float"])),
                "first_child_packet_id": str(first.get("child_packet_id", "")),
                "second_child_packet_id": str(second.get("child_packet_id", "")),
                "first_transition": f"{parent}_to_{child}",
                "second_transition": f"{opposite[0]}_to_{opposite[1]}",
                "first_weighted_mass": _fmt(first_weight),
                "second_weighted_mass": _fmt(second_weight),
                "cycle_mass_ratio": _fmt(ratio),
                "does_no2cycle_hold": _fmt(holds),
            }
        )

    max_ratio = max((_num(row["cycle_mass_ratio"]) for row in out), default=0.0)
    failures = sum(1 for row in out if row["does_no2cycle_hold"] == "false")
    return out, {
        "no2cycle_candidate_count": len(out),
        "no2cycle_failure_count": failures,
        "max_no2cycle_mass_ratio": max_ratio,
        "does_no2cycle_hold": failures == 0,
    }


def _route_decision(summary: dict[str, Any]) -> str:
    if not summary["does_no2cycle_hold"]:
        return "NO2CYCLE_FAILS"
    if summary["does_signed_flip_balance"]:
        return "SIGNED_FLIP_BALANCED_ROUTE_ALIVE"
    if summary["does_net_residue_decay"]:
        return "NET_RESIDUE_LYAPUNOV_ROUTE_ALIVE"
    if summary["does_signed_flip_decay"]:
        return "BT_SIGNED_DECAY_ROUTE_ALIVE"
    if summary["raw_minus_to_plus_exceeds_plus_to_minus"]:
        return "RAW_MINUS_TO_PLUS_UNBALANCED_ROUTE_BLOCKED"
    return "SIGNED_TERNARY_FLIP_UNRESOLVED"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    rows, input_manifests = _read_rows(args.inputs)
    flip_rows, totals = _aggregate_flip_balance(rows)
    bt_rows, bt_summary = _aggregate_bt_decay(rows, float(args.p))
    net_rows, net_summary = _aggregate_net_lyapunov(rows)
    no2cycle_rows, no2cycle_summary = _aggregate_no2cycle(rows, float(args.cycle_damping_threshold))

    m2p = totals["cross_minus_to_plus"]
    p2m = totals["cross_plus_to_minus"]
    paired = m2p + p2m
    imbalance = m2p - p2m
    abs_imbalance = abs(imbalance)
    imbalance_fraction = abs_imbalance / max(paired, EPS)
    summary: dict[str, Any] = {
        "contract": "ns_signed_ternary_flip_artifact",
        "diagnostic_mode": "sprint51_signed_ternary_flip_from_material_parent_table",
        "input_table_row_count": len(rows),
        "cross_shell_flip_balance_row_count": len(flip_rows),
        "bt_signed_flip_decay_row_count": len(bt_rows),
        "net_ternary_lyapunov_row_count": len(net_rows),
        "no2cycle_diagnostic_row_count": len(no2cycle_rows),
        "ternary_involution_semantics": "minus and plus are treated as an involutive pair; raw minus-to-plus is audited against plus-to-minus counter-flow",
        "BT_distance_proxy": "abs(K_child - K_parent)",
        "BT_distance_proxy_boundary": "diagnostic proxy only; not theorem-grade BT metric",
        "classification_field_used_for_source_kind": False,
        "weighted_cross_minus_to_plus": m2p,
        "weighted_cross_plus_to_minus": p2m,
        "weighted_cross_minus_to_zero": totals["cross_minus_to_zero"],
        "weighted_cross_plus_to_zero": totals["cross_plus_to_zero"],
        "weighted_cross_zero_to_plus": totals["cross_zero_to_plus"],
        "weighted_cross_zero_to_minus": totals["cross_zero_to_minus"],
        "signed_flip_imbalance": imbalance,
        "absolute_flip_imbalance": abs_imbalance,
        "paired_cross_flip_flow": paired,
        "signed_flip_imbalance_fraction_of_paired_flow": imbalance_fraction,
        "balance_fraction_threshold": float(args.balance_fraction_threshold),
        "does_signed_flip_balance": imbalance_fraction <= float(args.balance_fraction_threshold),
        "raw_minus_to_plus_exceeds_plus_to_minus": m2p > p2m,
        "net_cross_residue_total": totals["weighted_plus"] - totals["weighted_minus"],
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "signed_source_summability_proved": False,
        "bt_signed_decay_theorem_proved": False,
        "net_residue_lyapunov_proved": False,
        "no_persistent_two_cycle_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "promotion_status": "NO_PROMOTION_SIGNED_TERNARY_FLIP_DIAGNOSTIC",
        "inputs": [str(path) for path in args.inputs],
        "input_manifest_summaries": input_manifests,
    }
    summary.update(bt_summary)
    summary.update(net_summary)
    summary.update(no2cycle_summary)
    summary["route_decision"] = _route_decision(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    flip_path = args.out_dir / "ns_cross_shell_flip_balance.csv"
    net_path = args.out_dir / "ns_net_ternary_lyapunov.csv"
    bt_path = args.out_dir / "ns_bt_signed_flip_decay.csv"
    no2cycle_path = args.out_dir / "ns_no2cycle_diagnostic.csv"
    summary_path = args.out_dir / "ns_signed_ternary_flip_summary.json"
    summary["ns_cross_shell_flip_balance_path"] = str(flip_path)
    summary["ns_net_ternary_lyapunov_path"] = str(net_path)
    summary["ns_bt_signed_flip_decay_path"] = str(bt_path)
    summary["ns_no2cycle_diagnostic_path"] = str(no2cycle_path)
    summary["receipt_alignment"] = "DASHI.Physics.Closure.ClaySprintFiftyOneSignedTernaryFlipAuditReceipt"

    _write_csv(flip_path, FLIP_BALANCE_FIELDS, flip_rows)
    _write_csv(net_path, NET_LYAPUNOV_FIELDS, net_rows)
    _write_csv(bt_path, BT_DECAY_FIELDS, bt_rows)
    _write_csv(no2cycle_path, NO2CYCLE_FIELDS, no2cycle_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_signed_ternary_flip_audit] wrote {flip_path}")
    print(f"[ns_signed_ternary_flip_audit] wrote {net_path}")
    print(f"[ns_signed_ternary_flip_audit] wrote {bt_path}")
    print(f"[ns_signed_ternary_flip_audit] wrote {no2cycle_path}")
    print(f"[ns_signed_ternary_flip_audit] wrote {summary_path}")
    print(
        "[ns_signed_ternary_flip_audit] "
        f"route={summary['route_decision']} "
        f"imbalance_fraction={summary['signed_flip_imbalance_fraction_of_paired_flow']:.6g} "
        f"m2p={m2p:.6g} p2m={p2m:.6g}"
    )


if __name__ == "__main__":
    main()
