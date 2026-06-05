#!/usr/bin/env python3
"""Sprint 58 normalized packet-action inflation audit.

Sprint 57 showed that raw packet stretch under-reconstructs vessel stretch,
while normalized packet action is much larger than normalized vessel action.
Sprint 58 decomposes that mismatch:

    sum_P max((stretch_P / enstrophy_P) dt, 0)

is compared with

    max((sum_P stretch_P / sum_P enstrophy_P) dt, 0)

on the same covered packet union.  The first is a sum of local ratios; the
second is a ratio of sums.  Their gap is the non-additive normalization
inflation measured here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import ns_sprint53_no2cycle_physical_amplitude_audit as sprint53
import ns_sprint55_lagrangian_stretch_action_audit as sprint55
import ns_sprint56_packet_local_stretch_action_audit as sprint56
import ns_sprint57_vessel_action_reconciliation_audit as sprint57


EPS = 1e-30

PACKET_FIELDS = [
    "run",
    "time",
    "dt",
    "K",
    "packet_id",
    "packet_raw_signed_stretch_action",
    "packet_raw_positive_stretch_action",
    "packet_enstrophy",
    "covered_enstrophy",
    "global_enstrophy",
    "packet_enstrophy_fraction_of_covered",
    "packet_enstrophy_fraction_of_global",
    "packet_normalized_signed_action",
    "packet_normalized_positive_action",
    "covered_ratio_positive_action",
    "global_ratio_positive_action",
    "packet_to_covered_ratio_inflation",
    "packet_to_global_ratio_inflation",
    "low_enstrophy_denominator",
    "inflation_candidate",
]

TIME_FIELDS = [
    "run",
    "time",
    "dt",
    "requested_packet_count",
    "available_packet_count",
    "packet_raw_positive_stretch_action",
    "covered_raw_positive_stretch_action",
    "global_raw_positive_stretch_action",
    "packet_enstrophy_sum",
    "covered_enstrophy",
    "global_enstrophy",
    "packet_normalized_positive_action_sum",
    "covered_ratio_positive_action",
    "global_ratio_positive_action",
    "sum_ratios_over_ratio_of_sums_covered",
    "sum_ratios_over_ratio_of_sums_global",
    "low_enstrophy_packet_count",
    "inflation_candidate_count",
    "normalization_gap_status",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Sprint 49 material-parent output directories")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--truth-root", type=Path, default=Path("."), help="root used to resolve source_truth paths")
    p.add_argument("--low-enstrophy-fraction", type=float, default=1e-3)
    p.add_argument("--inflation-factor-threshold", type=float, default=10.0)
    p.add_argument("--action-threshold", type=float, default=0.05)
    p.add_argument("--action-small-majority", type=float, default=0.90)
    p.add_argument("--sigma-threshold", type=float, default=0.5)
    p.add_argument("--redirection-threshold", type=float, default=0.25)
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _ratio(num: float, den: float) -> float:
    return num / (den + EPS)


def _status(ratio: float, threshold: float) -> str:
    if ratio >= threshold:
        return "SUM_OF_LOCAL_RATIOS_INFLATED"
    return "NORMALIZATION_GAP_BELOW_THRESHOLD"


def _build_rows(
    table_rows: list[dict[str, Any]],
    truth_meta: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    requested = sprint57._requested_packets_by_run_time(table_rows)
    packet_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    status_by_run: dict[str, str] = {}
    last_time_by_run: dict[str, float] = {}
    for meta in truth_meta.values():
        run = str(meta["run"])
        loaded = sprint57._load_truth_run(meta)
        if loaded is None:
            status_by_run[run] = "source_truth_unavailable_or_incompatible"
            continue
        omega, velocity, steps, truth_json = loaded
        n = int(omega.shape[1])
        L = float(truth_json.get("domain_length") or (2.0 * np.pi))
        dt0 = float(truth_json.get("dt") or meta.get("dt") or 0.0)
        packet_grid = int(meta.get("packet_grid") or 8)
        shell_map = sprint57.sprint54._build_shell_map(n, L)
        cell_map = sprint56._cell_map(n, packet_grid)
        for t_idx, frame in enumerate(omega):
            time = float(steps[t_idx] * dt0) if dt0 > 0.0 and t_idx < len(steps) else float(t_idx)
            packet_keys = requested.get((run, time), set())
            if not packet_keys:
                continue
            dt = sprint57._snapshot_dt(run, time, last_time_by_run, meta)
            grad_u = sprint57.sprint54._build_velocity_gradient(velocity[t_idx], L)
            stretch = np.einsum("...i,...ij,...j->...", frame, grad_u, frame)
            enstrophy = np.einsum("...i,...i->...", frame, frame)
            global_enstrophy = float(np.sum(enstrophy))
            global_positive = float(np.sum(np.maximum(stretch, 0.0))) * dt
            covered_mask = np.zeros(stretch.shape, dtype=bool)
            packet_metrics: list[dict[str, Any]] = []
            for k, packet_id in sorted(packet_keys):
                mask = sprint57._packet_mask(shell_map, cell_map, packet_id)
                if mask is None:
                    continue
                covered_mask |= mask
                signed = float(np.sum(stretch[mask]))
                local_enstrophy = float(np.sum(enstrophy[mask]))
                raw_signed = signed * dt
                raw_positive = max(raw_signed, 0.0)
                normalized_signed = _ratio(signed, local_enstrophy) * dt
                normalized_positive = max(normalized_signed, 0.0)
                packet_metrics.append(
                    {
                        "K": k,
                        "packet_id": packet_id,
                        "raw_signed": raw_signed,
                        "raw_positive": raw_positive,
                        "enstrophy": local_enstrophy,
                        "normalized_signed": normalized_signed,
                        "normalized_positive": normalized_positive,
                    }
                )
            covered_enstrophy = float(np.sum(enstrophy[covered_mask])) if bool(np.any(covered_mask)) else 0.0
            covered_positive = float(np.sum(np.maximum(stretch[covered_mask], 0.0))) * dt if bool(np.any(covered_mask)) else 0.0
            covered_signed = float(np.sum(stretch[covered_mask])) if bool(np.any(covered_mask)) else 0.0
            covered_ratio_positive = max(_ratio(covered_signed, covered_enstrophy) * dt, 0.0)
            global_signed = float(np.sum(stretch))
            global_ratio_positive = max(_ratio(global_signed, global_enstrophy) * dt, 0.0)
            packet_norm_sum = sum(float(m["normalized_positive"]) for m in packet_metrics)
            packet_raw_sum = sum(float(m["raw_positive"]) for m in packet_metrics)
            packet_enstrophy_sum = sum(float(m["enstrophy"]) for m in packet_metrics)
            ratio_covered = _ratio(packet_norm_sum, covered_ratio_positive)
            ratio_global = _ratio(packet_norm_sum, global_ratio_positive)
            low_count = 0
            inflation_count = 0
            for metric in packet_metrics:
                e_frac_cov = _ratio(float(metric["enstrophy"]), covered_enstrophy)
                e_frac_global = _ratio(float(metric["enstrophy"]), global_enstrophy)
                packet_to_covered = _ratio(float(metric["normalized_positive"]), covered_ratio_positive)
                packet_to_global = _ratio(float(metric["normalized_positive"]), global_ratio_positive)
                low = e_frac_cov <= float(args.low_enstrophy_fraction)
                inflates = packet_to_covered >= float(args.inflation_factor_threshold)
                low_count += 1 if low else 0
                inflation_count += 1 if inflates else 0
                packet_rows.append(
                    {
                        "run": run,
                        "time": _fmt(time),
                        "dt": _fmt(dt),
                        "K": str(int(metric["K"])),
                        "packet_id": str(metric["packet_id"]),
                        "packet_raw_signed_stretch_action": _fmt(float(metric["raw_signed"])),
                        "packet_raw_positive_stretch_action": _fmt(float(metric["raw_positive"])),
                        "packet_enstrophy": _fmt(float(metric["enstrophy"])),
                        "covered_enstrophy": _fmt(covered_enstrophy),
                        "global_enstrophy": _fmt(global_enstrophy),
                        "packet_enstrophy_fraction_of_covered": _fmt(e_frac_cov),
                        "packet_enstrophy_fraction_of_global": _fmt(e_frac_global),
                        "packet_normalized_signed_action": _fmt(float(metric["normalized_signed"])),
                        "packet_normalized_positive_action": _fmt(float(metric["normalized_positive"])),
                        "covered_ratio_positive_action": _fmt(covered_ratio_positive),
                        "global_ratio_positive_action": _fmt(global_ratio_positive),
                        "packet_to_covered_ratio_inflation": _fmt(packet_to_covered),
                        "packet_to_global_ratio_inflation": _fmt(packet_to_global),
                        "low_enstrophy_denominator": _fmt(low),
                        "inflation_candidate": _fmt(inflates),
                    }
                )
            time_rows.append(
                {
                    "run": run,
                    "time": _fmt(time),
                    "dt": _fmt(dt),
                    "requested_packet_count": str(len(packet_keys)),
                    "available_packet_count": str(len(packet_metrics)),
                    "packet_raw_positive_stretch_action": _fmt(packet_raw_sum),
                    "covered_raw_positive_stretch_action": _fmt(covered_positive),
                    "global_raw_positive_stretch_action": _fmt(global_positive),
                    "packet_enstrophy_sum": _fmt(packet_enstrophy_sum),
                    "covered_enstrophy": _fmt(covered_enstrophy),
                    "global_enstrophy": _fmt(global_enstrophy),
                    "packet_normalized_positive_action_sum": _fmt(packet_norm_sum),
                    "covered_ratio_positive_action": _fmt(covered_ratio_positive),
                    "global_ratio_positive_action": _fmt(global_ratio_positive),
                    "sum_ratios_over_ratio_of_sums_covered": _fmt(ratio_covered),
                    "sum_ratios_over_ratio_of_sums_global": _fmt(ratio_global),
                    "low_enstrophy_packet_count": str(low_count),
                    "inflation_candidate_count": str(inflation_count),
                    "normalization_gap_status": _status(ratio_covered, float(args.inflation_factor_threshold)),
                }
            )
        status_by_run[run] = "normalized_action_inflation_available"
    return packet_rows, time_rows, status_by_run


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows)


def _max(rows: list[dict[str, Any]], key: str) -> float:
    return max((float(row[key]) for row in rows), default=0.0)


def _route(summary: dict[str, Any]) -> str:
    if int(summary["time_window_count"]) == 0:
        return "NORMALIZED_ACTION_SOURCE_TRUTH_UNAVAILABLE"
    if float(summary["sum_ratios_over_ratio_of_sums_covered"]) >= float(summary["inflation_factor_threshold"]):
        if float(summary["low_enstrophy_denominator_fraction"]) >= 0.5:
            return "NORMALIZED_ACTION_DENOMINATOR_INFLATION_DOMINATES"
        return "NORMALIZED_ACTION_NONADDITIVE_RATIO_INFLATION"
    return "NORMALIZED_ACTION_INFLATION_BELOW_THRESHOLD"


def _build_summary(
    packet_rows: list[dict[str, Any]],
    time_rows: list[dict[str, Any]],
    status_by_run: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    low_count = sum(1 for row in packet_rows if row["low_enstrophy_denominator"] == "true")
    infl_count = sum(1 for row in packet_rows if row["inflation_candidate"] == "true")
    packet_norm = _sum(time_rows, "packet_normalized_positive_action_sum")
    covered_ratio = _sum(time_rows, "covered_ratio_positive_action")
    global_ratio = _sum(time_rows, "global_ratio_positive_action")
    summary: dict[str, Any] = {
        "contract": "ns_sprint58_normalized_action_inflation_artifact",
        "diagnostic_mode": "sprint58_normalized_packet_action_inflation",
        "time_window_count": len(time_rows),
        "packet_inflation_row_count": len(packet_rows),
        "packet_normalized_positive_action_total": packet_norm,
        "covered_ratio_positive_action_total": covered_ratio,
        "global_ratio_positive_action_total": global_ratio,
        "sum_ratios_over_ratio_of_sums_covered": _ratio(packet_norm, covered_ratio),
        "sum_ratios_over_ratio_of_sums_global": _ratio(packet_norm, global_ratio),
        "packet_raw_positive_stretch_action_total": _sum(time_rows, "packet_raw_positive_stretch_action"),
        "covered_raw_positive_stretch_action_total": _sum(time_rows, "covered_raw_positive_stretch_action"),
        "global_raw_positive_stretch_action_total": _sum(time_rows, "global_raw_positive_stretch_action"),
        "packet_enstrophy_total": _sum(time_rows, "packet_enstrophy_sum"),
        "covered_enstrophy_total": _sum(time_rows, "covered_enstrophy"),
        "global_enstrophy_total": _sum(time_rows, "global_enstrophy"),
        "low_enstrophy_denominator_count": low_count,
        "low_enstrophy_denominator_fraction": low_count / max(len(packet_rows), 1),
        "inflation_candidate_count": infl_count,
        "inflation_candidate_fraction": infl_count / max(len(packet_rows), 1),
        "max_packet_to_covered_ratio_inflation": _max(packet_rows, "packet_to_covered_ratio_inflation"),
        "max_packet_to_global_ratio_inflation": _max(packet_rows, "packet_to_global_ratio_inflation"),
        "low_enstrophy_fraction_threshold": float(args.low_enstrophy_fraction),
        "inflation_factor_threshold": float(args.inflation_factor_threshold),
        "packet_action_reconstruction_proved": False,
        "normalized_action_additivity_proved": False,
        "denominator_inflation_theorem_proved": False,
        "weighted_packet_action_summability_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT58_NORMALIZED_ACTION_INFLATION_DIAGNOSTIC",
        "receipt_alignment": "DASHI.Physics.Closure.ClaySprintFiftyEightNormalizedActionInflationReceipt",
        "status_by_run": status_by_run,
        "boundary": (
            "Sprint 58 decomposes normalized packet-action inflation as a "
            "sum-of-local-ratios versus ratio-of-sums diagnostic. It records "
            "denominator and non-additivity evidence only; weighted summability, "
            "physical bridge, stretch absorption, and no-blowup remain unproved."
        ),
    }
    summary["route_decision"] = _route(summary)
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
    truth_meta = sprint55._truth_meta_by_run(args.inputs, args.truth_root)
    for input_dir in args.inputs:
        run = input_dir.name
        summary = json.loads((input_dir / "ns_material_parent_summary.json").read_text(encoding="utf-8"))
        truth_meta[run]["packet_grid"] = int(summary.get("packet_grid") or 8)
        truth_meta[run]["packet_active_quantile"] = float(summary.get("packet_active_quantile") or 0.9)
    packet_rows, time_rows, status_by_run = _build_rows(table_rows, truth_meta, args)
    summary = _build_summary(packet_rows, time_rows, status_by_run, args)
    summary["inputs"] = [str(path) for path in args.inputs]
    summary["input_manifest_summaries"] = manifests
    summary["truth_metadata"] = list(truth_meta.values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.out_dir / "ns_normalized_action_inflation_packets.csv"
    time_path = args.out_dir / "ns_normalized_action_inflation_by_time.csv"
    summary_path = args.out_dir / "ns_sprint58_normalized_action_inflation_summary.json"
    summary["ns_normalized_action_inflation_packets_path"] = str(packet_path)
    summary["ns_normalized_action_inflation_by_time_path"] = str(time_path)
    _write_csv(packet_path, PACKET_FIELDS, packet_rows)
    _write_csv(time_path, TIME_FIELDS, time_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint58_normalized_action_inflation_audit] wrote {packet_path}")
    print(f"[ns_sprint58_normalized_action_inflation_audit] wrote {time_path}")
    print(f"[ns_sprint58_normalized_action_inflation_audit] wrote {summary_path}")
    print(
        "[ns_sprint58_normalized_action_inflation_audit] "
        f"route={summary['route_decision']} "
        f"ratio_covered={summary['sum_ratios_over_ratio_of_sums_covered']} "
        f"low_denominator_fraction={summary['low_enstrophy_denominator_fraction']} "
        f"promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
