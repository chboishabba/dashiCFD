#!/usr/bin/env python3
"""Sprint 52 material-source and no-2-cycle amplitude audit.

This consumes Sprint 49 material-parent artifacts and decides the two narrow
post-Sprint-51 gates:

* whether material-parent matching leaves any true-new positive source;
* whether v1 no-2-cycle proxy failures are small in material-packet amplitude.

The script does not rerun GPU truth or packet matching.  It treats Sprint 49
material-parent tables as the authoritative advected-parent evidence.
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
SOURCE_FIELDS = [
    "K",
    "t",
    "material_parent_id",
    "parent_relation",
    "child_state",
    "parent_state",
    "source_true_new_material",
    "source_tracking_uncertain",
    "source_cross_shell",
    "source_low_shell",
    "sigma_true_new_material",
    "weighted_true_new_material",
    "material_M_plus_plus",
]
AMPLITUDE_FIELDS = [
    "cycle_id",
    "K",
    "t",
    "minus_to_plus_mass",
    "plus_to_minus_mass",
    "signed_imbalance",
    "cycle_amplitude",
    "cycle_persistence",
    "weighted_cycle_amplitude",
    "plus_shell_weighted_mass",
    "amplitude_fraction_of_plus_shell",
    "no2cycle_proxy_fail",
    "no2cycle_amplitude_small",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Sprint 49 output directories containing material-parent table and summary artifacts",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sigma-threshold", type=float, default=0.5)
    p.add_argument("--cycle-damping-threshold", type=float, default=1.0 / math.sqrt(2.0))
    p.add_argument(
        "--amplitude-small-fraction",
        type=float,
        default=0.05,
        help="cycle amplitude is small if <= this fraction of plus-shell weighted mass",
    )
    p.add_argument(
        "--amplitude-small-majority",
        type=float,
        default=0.90,
        help="fraction of proxy-failing cycles that must be amplitude-small to clear 52B",
    )
    return p.parse_args()


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


def _require_input_dir(path: Path) -> tuple[Path, Path, dict[str, Any]]:
    table = path / "ns_material_parent_table.csv"
    summary_csv = path / "ns_material_parent_summary.csv"
    summary_json = path / "ns_material_parent_summary.json"
    if not table.exists():
        raise SystemExit(f"{path} lacks ns_material_parent_table.csv")
    if not summary_csv.exists():
        raise SystemExit(f"{path} lacks ns_material_parent_summary.csv")
    if not summary_json.exists():
        raise SystemExit(f"{path} lacks ns_material_parent_summary.json")
    try:
        meta = json.loads(summary_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{summary_json} is not valid JSON: {exc}") from exc
    return table, summary_csv, meta


def _read_inputs(inputs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    table_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    table_required = {
        "time",
        "K_parent",
        "K_child",
        "child_packet_id",
        "parent_packet_id",
        "child_state",
        "parent_state",
        "child_mass",
        "parent_mass",
        "credited_mass",
        "source_true_new",
        "source_tracking_uncertain",
        "source_cross_shell",
        "source_low_shell_injection",
        "parent_relation",
    }
    summary_required = {
        "time",
        "K_child",
        "M_plus_plus_material",
        "source_true_new",
        "weighted_true_new",
        "sigma_true_new_fit",
    }
    for input_dir in inputs:
        table, summary_csv, meta = _require_input_dir(input_dir)
        manifests.append(
            {
                "input_dir": str(input_dir),
                "summary_contract": meta.get("contract"),
                "summary_row_count": meta.get("summary_row_count"),
                "table_row_count": meta.get("table_row_count"),
                "material_parent_route_status": meta.get("material_parent_route_status"),
                "source_truth": meta.get("source_truth"),
            }
        )
        with table.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(table_required.difference(reader.fieldnames or []))
            if missing:
                raise SystemExit(f"{table} is missing columns: {', '.join(missing)}")
            for raw in reader:
                k_parent = _int(raw.get("K_parent"))
                k_child = _int(raw.get("K_child"))
                weighted = (2.0 ** (0.5 * float(k_child))) * _num(raw.get("credited_mass"))
                row = {
                    **raw,
                    "input_dir": str(input_dir),
                    "time_float": _num(raw.get("time")),
                    "K_parent_int": k_parent,
                    "K_child_int": k_child,
                    "credited_mass_float": _num(raw.get("credited_mass")),
                    "child_mass_float": _num(raw.get("child_mass")),
                    "parent_mass_float": _num(raw.get("parent_mass")),
                    "weighted_mass_float": weighted,
                    "source_true_new_float": _num(raw.get("source_true_new")),
                    "source_tracking_uncertain_float": _num(raw.get("source_tracking_uncertain")),
                    "source_cross_shell_float": _num(raw.get("source_cross_shell")),
                    "source_low_shell_float": _num(raw.get("source_low_shell_injection")),
                }
                table_rows.append(row)
        with summary_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(summary_required.difference(reader.fieldnames or []))
            if missing:
                raise SystemExit(f"{summary_csv} is missing columns: {', '.join(missing)}")
            for raw in reader:
                summary_rows.append({**raw, "input_dir": str(input_dir)})
    return table_rows, summary_rows, manifests


def _material_source_rows(table_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    mpp: dict[tuple[str, float, int], float] = {}
    for row in summary_rows:
        key = (str(row["input_dir"]), _num(row.get("time")), _int(row.get("K_child")))
        mpp[key] = _num(row.get("M_plus_plus_material"))

    plus_mass_by_time_k: dict[tuple[str, float, int], float] = defaultdict(float)
    true_new_by_time_k: dict[tuple[str, float, int], float] = defaultdict(float)
    for row in table_rows:
        key = (str(row["input_dir"]), float(row["time_float"]), int(row["K_child_int"]))
        if row.get("child_state") == "plus":
            plus_mass_by_time_k[key] += float(row["child_mass_float"])
        true_new_by_time_k[key] += float(row["source_true_new_float"])

    out: list[dict[str, Any]] = []
    weighted_true_total = 0.0
    true_total = 0.0
    max_sigma = 0.0
    for row in table_rows:
        k = int(row["K_child_int"])
        time = float(row["time_float"])
        key = (str(row["input_dir"]), time, k)
        sigma = true_new_by_time_k[key] / max(plus_mass_by_time_k[key], EPS)
        weighted_true = (2.0 ** (0.5 * float(k))) * float(row["source_true_new_float"])
        weighted_true_total += weighted_true
        true_total += float(row["source_true_new_float"])
        max_sigma = max(max_sigma, sigma)
        if (
            row.get("child_state") == "plus"
            or float(row["source_true_new_float"]) > 0.0
            or float(row["source_tracking_uncertain_float"]) > 0.0
            or float(row["source_cross_shell_float"]) > 0.0
            or float(row["source_low_shell_float"]) > 0.0
        ):
            out.append(
                {
                    "K": str(k),
                    "t": _fmt(time),
                    "material_parent_id": str(row.get("parent_packet_id", "")),
                    "parent_relation": str(row.get("parent_relation", "")),
                    "child_state": str(row.get("child_state", "")),
                    "parent_state": str(row.get("parent_state", "")),
                    "source_true_new_material": _fmt(float(row["source_true_new_float"])),
                    "source_tracking_uncertain": _fmt(float(row["source_tracking_uncertain_float"])),
                    "source_cross_shell": _fmt(float(row["source_cross_shell_float"])),
                    "source_low_shell": _fmt(float(row["source_low_shell_float"])),
                    "sigma_true_new_material": _fmt(sigma),
                    "weighted_true_new_material": _fmt(weighted_true),
                    "material_M_plus_plus": _fmt(mpp.get(key, 0.0)),
                }
            )
    return out, {
        "source_true_new_material_total": true_total,
        "weighted_true_new_material_total": weighted_true_total,
        "sigma_true_new_material_max": max_sigma,
    }


def _transition(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("parent_state", "")), str(row.get("child_state", ""))


def _no2cycle_amplitude_rows(
    table_rows: list[dict[str, Any]],
    damping_threshold: float,
    amplitude_small_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    plus_shell: dict[tuple[str, float, int], float] = defaultdict(float)
    for row in table_rows:
        if row.get("child_state") == "plus":
            plus_shell[(str(row["input_dir"]), float(row["time_float"]), int(row["K_child_int"]))] += float(
                row["weighted_mass_float"]
            )

    flips = [
        row
        for row in table_rows
        if _transition(row) in {("minus", "plus"), ("plus", "minus")} and float(row["credited_mass_float"]) > 0.0
    ]
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in flips:
        by_parent[(str(row["input_dir"]), str(row.get("parent_packet_id", "")))].append(row)

    out: list[dict[str, Any]] = []
    by_k_amplitude: dict[int, float] = defaultdict(float)
    cycle_idx = 0
    for first in flips:
        first_child_key = (str(first["input_dir"]), str(first.get("child_packet_id", "")))
        parent, child = _transition(first)
        opposite = (child, parent)
        candidates = [
            second
            for second in by_parent.get(first_child_key, [])
            if _transition(second) == opposite and float(second["time_float"]) > float(first["time_float"])
        ]
        if not candidates:
            continue
        second = min(candidates, key=lambda item: float(item["time_float"]))
        first_weight = float(first["weighted_mass_float"])
        second_weight = float(second["weighted_mass_float"])
        ratio = second_weight / max(first_weight, EPS)
        proxy_fail = ratio > damping_threshold
        if parent == "minus":
            m2p = first_weight
            p2m = second_weight
        else:
            m2p = second_weight
            p2m = first_weight
        signed = m2p - p2m
        persistence = min(first_weight, second_weight) / max(first_weight, second_weight, EPS)
        amplitude = min(m2p, p2m) * persistence
        shell_key = (str(second["input_dir"]), float(second["time_float"]), int(second["K_child_int"]))
        shell_plus = plus_shell.get(shell_key, 0.0)
        amplitude_denominator = shell_plus if shell_plus > EPS else (m2p + p2m)
        amp_fraction = amplitude / max(amplitude_denominator, EPS)
        amplitude_small = amp_fraction <= amplitude_small_fraction
        by_k_amplitude[int(second["K_child_int"])] += amplitude
        cycle_idx += 1
        out.append(
            {
                "cycle_id": f"cycle_{cycle_idx}",
                "K": str(int(second["K_child_int"])),
                "t": _fmt(float(second["time_float"])),
                "minus_to_plus_mass": _fmt(m2p),
                "plus_to_minus_mass": _fmt(p2m),
                "signed_imbalance": _fmt(signed),
                "cycle_amplitude": _fmt(amplitude),
                "cycle_persistence": _fmt(persistence),
                "weighted_cycle_amplitude": _fmt(amplitude),
                "plus_shell_weighted_mass": _fmt(shell_plus),
                "amplitude_fraction_of_plus_shell": _fmt(amp_fraction),
                "no2cycle_proxy_fail": _fmt(proxy_fail),
                "no2cycle_amplitude_small": _fmt(amplitude_small),
            }
        )

    fail_rows = [row for row in out if row["no2cycle_proxy_fail"] == "true"]
    small_failures = [row for row in fail_rows if row["no2cycle_amplitude_small"] == "true"]
    max_fraction = max((_num(row["amplitude_fraction_of_plus_shell"]) for row in out), default=0.0)
    total_amp = sum(_num(row["weighted_cycle_amplitude"]) for row in out)
    return out, {
        "no2cycle_candidate_count": len(out),
        "no2cycle_proxy_failure_count": len(fail_rows),
        "no2cycle_amplitude_small_failure_count": len(small_failures),
        "no2cycle_amplitude_small_failure_fraction": len(small_failures) / max(len(fail_rows), 1),
        "weighted_no2cycle_amplitude_total": total_amp,
        "max_no2cycle_amplitude_fraction_of_plus_shell": max_fraction,
    }


def _route(summary: dict[str, Any]) -> str:
    material = bool(summary["does_material_source_gate_close"])
    amp = bool(summary["does_no2cycle_amplitude_gate_close"])
    if material and amp:
        return "NS_SOURCE_BUDGET_ROUTE_ALIVE_DIAGNOSTIC"
    if not material and not amp:
        return "NS_SOURCE_BUDGET_ROUTE_FALSIFIED_DIAGNOSTIC"
    if material:
        return "MATERIAL_SOURCE_GATE_CLOSED_NO2CYCLE_AMPLITUDE_BLOCKED"
    return "MATERIAL_SOURCE_GATE_BLOCKED_NO2CYCLE_AMPLITUDE_SMALL"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    table_rows, summary_rows, manifests = _read_inputs(args.inputs)
    source_rows, source_summary = _material_source_rows(table_rows, summary_rows)
    amp_rows, amp_summary = _no2cycle_amplitude_rows(
        table_rows,
        damping_threshold=float(args.cycle_damping_threshold),
        amplitude_small_fraction=float(args.amplitude_small_fraction),
    )
    true_absent = float(source_summary["weighted_true_new_material_total"]) <= EPS
    sigma_beats = float(source_summary["sigma_true_new_material_max"]) > float(args.sigma_threshold)
    amp_fraction = float(amp_summary["no2cycle_amplitude_small_failure_fraction"])
    amp_closes = amp_fraction >= float(args.amplitude_small_majority)
    summary: dict[str, Any] = {
        "contract": "ns_sprint52_material_no2cycle_artifact",
        "diagnostic_mode": "sprint52_material_parent_exponent_and_no2cycle_amplitude",
        "input_table_row_count": len(table_rows),
        "material_source_resolution_row_count": len(source_rows),
        "no2cycle_amplitude_row_count": len(amp_rows),
        "material_parent_relation_boundary": "reuses Sprint 49 advected material-parent table; does not rerun packet matching",
        "no2cycle_amplitude_boundary": "material-packet weighted amplitude proxy; not theorem-grade physical stretch oscillation",
        "source_true_new_material_total": source_summary["source_true_new_material_total"],
        "weighted_true_new_material_total": source_summary["weighted_true_new_material_total"],
        "sigma_true_new_material": source_summary["sigma_true_new_material_max"],
        "sigma_threshold": float(args.sigma_threshold),
        "material_true_new_source_absent": true_absent,
        "material_source_beats_half_derivative": sigma_beats,
        "does_material_source_gate_close": true_absent or sigma_beats,
        "cycle_damping_threshold": float(args.cycle_damping_threshold),
        "amplitude_small_fraction": float(args.amplitude_small_fraction),
        "amplitude_small_majority": float(args.amplitude_small_majority),
        **amp_summary,
        "does_no2cycle_amplitude_gate_close": amp_closes,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "material_source_exponent_proved": False,
        "weighted_no2cycle_amplitude_summability_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "promotion_status": "NO_PROMOTION_SPRINT52_MATERIAL_NO2CYCLE_DIAGNOSTIC",
        "inputs": [str(path) for path in args.inputs],
        "input_manifest_summaries": manifests,
    }
    summary["route_decision"] = _route(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.out_dir / "ns_material_source_resolution.csv"
    amp_path = args.out_dir / "ns_no2cycle_amplitude.csv"
    summary_path = args.out_dir / "ns_sprint52_material_no2cycle_summary.json"
    summary["ns_material_source_resolution_path"] = str(source_path)
    summary["ns_no2cycle_amplitude_path"] = str(amp_path)
    summary["receipt_alignment"] = "DASHI.Physics.Closure.ClaySprintFiftyTwoMaterialNo2CycleAuditReceipt"
    _write_csv(source_path, SOURCE_FIELDS, source_rows)
    _write_csv(amp_path, AMPLITUDE_FIELDS, amp_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint52_material_no2cycle_audit] wrote {source_path}")
    print(f"[ns_sprint52_material_no2cycle_audit] wrote {amp_path}")
    print(f"[ns_sprint52_material_no2cycle_audit] wrote {summary_path}")
    print(
        "[ns_sprint52_material_no2cycle_audit] "
        f"route={summary['route_decision']} "
        f"material_gate={summary['does_material_source_gate_close']} "
        f"amplitude_gate={summary['does_no2cycle_amplitude_gate_close']}"
    )


if __name__ == "__main__":
    main()
