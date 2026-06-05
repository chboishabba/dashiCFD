#!/usr/bin/env python3
"""Sprint 57 global vessel/action reconciliation audit.

Sprint 56 identified packet-local accumulated positive stretch action as the
right physical object, but the packet-local route stayed blocked under the
Euclidean ``K_cell`` packet ledger.  Sprint 57 asks whether that packet ledger
reconstructs the whole-domain vortex-stretching balance.

The script reports two reconciliation surfaces:

* normalized action, matching Sprint 56 ``A+`` units:
  ``alpha = sum(omega dot S omega) / (sum |omega|^2 + eps)``;
* raw stretch action, matching the global enstrophy production integrand:
  ``sum(omega dot S omega) * dt``.

Raw packet sums are compared both to the full domain and to the union of packet
masks touched by the Sprint 49 material-parent rows.  This keeps the audit
fail-closed: reducing a packet total is only meaningful if the physical stretch
action is still accounted for.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import ns_sprint53_no2cycle_physical_amplitude_audit as sprint53
import ns_sprint54_no2cycle_resolution_cadence_audit as sprint54
import ns_sprint55_lagrangian_stretch_action_audit as sprint55
import ns_sprint56_packet_local_stretch_action_audit as sprint56


EPS = 1e-30

TIME_FIELDS = [
    "run",
    "time",
    "dt",
    "N",
    "save_every",
    "requested_packet_count",
    "available_packet_count",
    "covered_voxel_count",
    "domain_voxel_count",
    "coverage_fraction",
    "double_count_voxel_count",
    "double_count_fraction",
    "global_raw_positive_stretch_action",
    "global_raw_net_stretch_action",
    "covered_raw_positive_stretch_action",
    "covered_raw_net_stretch_action",
    "packet_raw_positive_stretch_action",
    "packet_raw_net_stretch_action",
    "global_normalized_positive_action",
    "global_normalized_net_action",
    "packet_normalized_positive_action",
    "packet_normalized_net_action",
    "epsilon_raw_positive_vs_global",
    "epsilon_raw_net_vs_global",
    "epsilon_raw_positive_vs_covered",
    "epsilon_raw_net_vs_covered",
    "epsilon_normalized_positive_vs_global",
    "epsilon_normalized_net_vs_global",
    "partition_status",
]

SUMMARY_FIELDS = [
    "assignment_scheme",
    "time_window_count",
    "global_raw_positive_stretch_action_total",
    "global_raw_net_stretch_action_total",
    "covered_raw_positive_stretch_action_total",
    "covered_raw_net_stretch_action_total",
    "packet_raw_positive_stretch_action_total",
    "packet_raw_net_stretch_action_total",
    "global_normalized_positive_action_total",
    "global_normalized_net_action_total",
    "packet_normalized_positive_action_total",
    "packet_normalized_net_action_total",
    "epsilon_raw_positive_vs_global",
    "epsilon_raw_net_vs_global",
    "epsilon_raw_positive_vs_covered",
    "epsilon_raw_net_vs_covered",
    "epsilon_normalized_positive_vs_global",
    "epsilon_normalized_net_vs_global",
    "mean_coverage_fraction",
    "max_double_count_fraction",
    "dangerous_lineage_count",
    "sigma_packet_local_action_fit",
    "route_decision",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Sprint 49 material-parent output directories")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--truth-root", type=Path, default=Path("."), help="root used to resolve source_truth paths")
    p.add_argument("--action-threshold", type=float, default=0.05)
    p.add_argument("--action-small-majority", type=float, default=0.90)
    p.add_argument("--sigma-threshold", type=float, default=0.5)
    p.add_argument("--redirection-threshold", type=float, default=0.25)
    p.add_argument("--reconstruction-tolerance", type=float, default=0.25)
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _epsilon(observed: float, reference: float) -> float:
    return (observed - reference) / (abs(reference) + EPS)


def _snapshot_dt(run: str, time: float, last_time_by_run: dict[str, float], meta: dict[str, Any]) -> float:
    fallback = float(meta.get("save_every") or 0) * float(meta.get("dt") or 0.0)
    if fallback <= EPS:
        fallback = float(meta.get("dt") or 0.0)
    last = last_time_by_run.get(run)
    last_time_by_run[run] = time
    if last is None:
        return fallback if fallback > EPS else 0.0
    return max(time - last, fallback if fallback > EPS else 0.0)


def _requested_packets_by_run_time(table_rows: list[dict[str, Any]]) -> dict[tuple[str, float], set[tuple[int, str]]]:
    requested: dict[tuple[str, float], set[tuple[int, str]]] = defaultdict(set)
    for row in table_rows:
        packet_id = str(row.get("child_packet_id", ""))
        parsed = sprint56._parse_packet_id(packet_id)
        if parsed is None:
            continue
        k, _cell = parsed
        requested[(str(row["run"]), float(row["time_float"]))].add((k, packet_id))
    return requested


def _load_truth_run(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    path = Path(str(meta.get("truth_path") or ""))
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "omega_snapshots" not in data.files or "velocity_snapshots" not in data.files:
            return None
        omega = np.asarray(data["omega_snapshots"], dtype=np.float64)
        velocity = np.asarray(data["velocity_snapshots"], dtype=np.float64)
        steps = np.asarray(data["steps"], dtype=np.float64) if "steps" in data.files else np.arange(len(omega))
        meta_json = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
    if omega.shape != velocity.shape or omega.ndim != 5 or omega.shape[-1] != 3:
        return None
    return omega, velocity, steps, meta_json


def _packet_mask(shell_map: np.ndarray, cell_map: np.ndarray, packet_id: str) -> np.ndarray | None:
    parsed = sprint56._parse_packet_id(packet_id)
    if parsed is None:
        return None
    k, cell = parsed
    mask = (shell_map == k) & (cell_map == cell)
    return mask if bool(np.any(mask)) else None


def _build_global_rows(
    table_rows: list[dict[str, Any]],
    truth_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    requested = _requested_packets_by_run_time(table_rows)
    rows: list[dict[str, Any]] = []
    status_by_run: dict[str, str] = {}
    last_time_by_run: dict[str, float] = {}
    for meta in truth_meta.values():
        run = str(meta["run"])
        loaded = _load_truth_run(meta)
        if loaded is None:
            status_by_run[run] = "source_truth_unavailable_or_incompatible"
            continue
        omega, velocity, steps, truth_json = loaded
        n = int(omega.shape[1])
        L = float(truth_json.get("domain_length") or (2.0 * math.pi))
        dt0 = float(truth_json.get("dt") or meta.get("dt") or 0.0)
        packet_grid = int(meta.get("packet_grid") or 8)
        shell_map = sprint54._build_shell_map(n, L)
        cell_map = sprint56._cell_map(n, packet_grid)
        for t_idx, frame in enumerate(omega):
            time = float(steps[t_idx] * dt0) if dt0 > 0.0 and t_idx < len(steps) else float(t_idx)
            packet_keys = requested.get((run, time), set())
            if not packet_keys:
                continue
            dt = _snapshot_dt(run, time, last_time_by_run, meta)
            grad_u = sprint54._build_velocity_gradient(velocity[t_idx], L)
            stretch = np.einsum("...i,...ij,...j->...", frame, grad_u, frame)
            enstrophy = np.einsum("...i,...i->...", frame, frame)
            global_signed = float(np.sum(stretch))
            global_positive = float(np.sum(np.maximum(stretch, 0.0)))
            global_enstrophy = float(np.sum(enstrophy))
            covered_mask = np.zeros(stretch.shape, dtype=bool)
            coverage_count = np.zeros(stretch.shape, dtype=np.int16)
            packet_raw_pos = 0.0
            packet_raw_net = 0.0
            packet_norm_pos = 0.0
            packet_norm_net = 0.0
            available_packets = 0
            for _k, packet_id in sorted(packet_keys):
                mask = _packet_mask(shell_map, cell_map, packet_id)
                if mask is None:
                    continue
                coverage_count[mask] += 1
                covered_mask |= mask
                signed = float(np.sum(stretch[mask]))
                local_enstrophy = float(np.sum(enstrophy[mask]))
                alpha = signed / (local_enstrophy + EPS)
                inc = alpha * dt
                raw_inc = signed * dt
                packet_raw_pos += max(raw_inc, 0.0)
                packet_raw_net += raw_inc
                packet_norm_pos += max(inc, 0.0)
                packet_norm_net += inc
                available_packets += 1
            covered_signed = float(np.sum(stretch[covered_mask])) if bool(np.any(covered_mask)) else 0.0
            covered_positive = float(np.sum(np.maximum(stretch[covered_mask], 0.0))) if bool(np.any(covered_mask)) else 0.0
            double_count = int(np.count_nonzero(coverage_count > 1))
            covered_count = int(np.count_nonzero(covered_mask))
            domain_count = int(stretch.size)
            global_norm_net = (global_signed / (global_enstrophy + EPS)) * dt
            global_norm_pos = (global_positive / (global_enstrophy + EPS)) * dt
            global_raw_pos_action = global_positive * dt
            global_raw_net_action = global_signed * dt
            covered_raw_pos_action = covered_positive * dt
            covered_raw_net_action = covered_signed * dt
            rows.append(
                {
                    "run": run,
                    "time": _fmt(time),
                    "dt": _fmt(dt),
                    "N": str(n),
                    "save_every": str(int(meta.get("save_every") or truth_json.get("save_every") or 0)),
                    "requested_packet_count": str(len(packet_keys)),
                    "available_packet_count": str(available_packets),
                    "covered_voxel_count": str(covered_count),
                    "domain_voxel_count": str(domain_count),
                    "coverage_fraction": _fmt(covered_count / max(domain_count, 1)),
                    "double_count_voxel_count": str(double_count),
                    "double_count_fraction": _fmt(double_count / max(covered_count, 1)),
                    "global_raw_positive_stretch_action": _fmt(global_raw_pos_action),
                    "global_raw_net_stretch_action": _fmt(global_raw_net_action),
                    "covered_raw_positive_stretch_action": _fmt(covered_raw_pos_action),
                    "covered_raw_net_stretch_action": _fmt(covered_raw_net_action),
                    "packet_raw_positive_stretch_action": _fmt(packet_raw_pos),
                    "packet_raw_net_stretch_action": _fmt(packet_raw_net),
                    "global_normalized_positive_action": _fmt(global_norm_pos),
                    "global_normalized_net_action": _fmt(global_norm_net),
                    "packet_normalized_positive_action": _fmt(packet_norm_pos),
                    "packet_normalized_net_action": _fmt(packet_norm_net),
                    "epsilon_raw_positive_vs_global": _fmt(_epsilon(packet_raw_pos, global_raw_pos_action)),
                    "epsilon_raw_net_vs_global": _fmt(_epsilon(packet_raw_net, global_raw_net_action)),
                    "epsilon_raw_positive_vs_covered": _fmt(_epsilon(packet_raw_pos, covered_raw_pos_action)),
                    "epsilon_raw_net_vs_covered": _fmt(_epsilon(packet_raw_net, covered_raw_net_action)),
                    "epsilon_normalized_positive_vs_global": _fmt(_epsilon(packet_norm_pos, global_norm_pos)),
                    "epsilon_normalized_net_vs_global": _fmt(_epsilon(packet_norm_net, global_norm_net)),
                    "partition_status": "packet_union_reconstructed_from_K_cell_geometry",
                }
            )
        status_by_run[run] = "global_stretch_reconciliation_available"
    return rows, status_by_run


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return _sum(rows, key) / max(len(rows), 1)


def _max(rows: list[dict[str, Any]], key: str) -> float:
    return max((float(row[key]) for row in rows), default=0.0)


def _route(summary: dict[str, Any], tolerance: float) -> str:
    if int(summary["time_window_count"]) == 0:
        return "VESSEL_RECONCILIATION_SOURCE_TRUTH_UNAVAILABLE"
    eps_cov = abs(float(summary["epsilon_raw_positive_vs_covered"]))
    eps_global = abs(float(summary["epsilon_raw_positive_vs_global"]))
    max_double = float(summary["max_double_count_fraction"])
    if eps_cov <= tolerance and eps_global <= tolerance:
        return "PACKET_ACTION_RECONSTRUCTS_GLOBAL_STRETCH"
    if float(summary["epsilon_raw_positive_vs_covered"]) > tolerance:
        return "PACKET_ACTION_OVERCOUNTS_COVERED_STRETCH"
    if float(summary["epsilon_raw_positive_vs_covered"]) < -tolerance:
        return "PACKET_ACTION_UNDERCOUNTS_COVERED_STRETCH"
    if max_double > tolerance:
        return "PACKET_ACTION_DOUBLE_COUNTS_BOUNDARY_BOUNCES"
    if eps_global > tolerance:
        return "PACKET_ACTION_DOES_NOT_COVER_GLOBAL_STRETCH"
    return "NS_PACKET_ACTION_ROUTE_STILL_BLOCKED"


def _build_summary(
    time_rows: list[dict[str, Any]],
    sprint56_summary: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    totals = {
        "global_raw_positive_stretch_action_total": _sum(time_rows, "global_raw_positive_stretch_action"),
        "global_raw_net_stretch_action_total": _sum(time_rows, "global_raw_net_stretch_action"),
        "covered_raw_positive_stretch_action_total": _sum(time_rows, "covered_raw_positive_stretch_action"),
        "covered_raw_net_stretch_action_total": _sum(time_rows, "covered_raw_net_stretch_action"),
        "packet_raw_positive_stretch_action_total": _sum(time_rows, "packet_raw_positive_stretch_action"),
        "packet_raw_net_stretch_action_total": _sum(time_rows, "packet_raw_net_stretch_action"),
        "global_normalized_positive_action_total": _sum(time_rows, "global_normalized_positive_action"),
        "global_normalized_net_action_total": _sum(time_rows, "global_normalized_net_action"),
        "packet_normalized_positive_action_total": _sum(time_rows, "packet_normalized_positive_action"),
        "packet_normalized_net_action_total": _sum(time_rows, "packet_normalized_net_action"),
    }
    summary: dict[str, Any] = {
        "contract": "ns_sprint57_vessel_action_reconciliation_artifact",
        "diagnostic_mode": "sprint57_global_vessel_action_reconciliation",
        "assignment_scheme": "euclidean_K_cell_packet_union",
        "time_window_count": len(time_rows),
        **totals,
        "epsilon_raw_positive_vs_global": _epsilon(
            totals["packet_raw_positive_stretch_action_total"],
            totals["global_raw_positive_stretch_action_total"],
        ),
        "epsilon_raw_net_vs_global": _epsilon(
            totals["packet_raw_net_stretch_action_total"],
            totals["global_raw_net_stretch_action_total"],
        ),
        "epsilon_raw_positive_vs_covered": _epsilon(
            totals["packet_raw_positive_stretch_action_total"],
            totals["covered_raw_positive_stretch_action_total"],
        ),
        "epsilon_raw_net_vs_covered": _epsilon(
            totals["packet_raw_net_stretch_action_total"],
            totals["covered_raw_net_stretch_action_total"],
        ),
        "epsilon_normalized_positive_vs_global": _epsilon(
            totals["packet_normalized_positive_action_total"],
            totals["global_normalized_positive_action_total"],
        ),
        "epsilon_normalized_net_vs_global": _epsilon(
            totals["packet_normalized_net_action_total"],
            totals["global_normalized_net_action_total"],
        ),
        "mean_coverage_fraction": _mean(time_rows, "coverage_fraction"),
        "max_double_count_fraction": _max(time_rows, "double_count_fraction"),
        "reconstruction_tolerance": float(args.reconstruction_tolerance),
        "dangerous_lineage_count": int(sprint56_summary.get("dangerous_lineage_count") or 0),
        "sigma_packet_local_action_fit": float(sprint56_summary.get("sigma_packet_local_action_fit") or 0.0),
        "packet_local_action_gate_proved": False,
        "packet_action_reconstructs_global_stretch_proved": False,
        "ultrametric_reassignment_proved": False,
        "weighted_packet_local_action_summability_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT57_VESSEL_ACTION_RECONCILIATION_DIAGNOSTIC",
        "receipt_alignment": "DASHI.Physics.Closure.ClaySprintFiftySevenVesselActionReconciliationReceipt",
        "boundary": (
            "Sprint 57 compares Sprint 49/56 Euclidean K_cell packet-local "
            "stretch actions with whole-domain and covered-mask omega dot S omega "
            "actions. It records accounting evidence only; BT reassignment, "
            "weighted summability, physical bridge, stretch absorption, and "
            "no-blowup remain unproved."
        ),
    }
    summary["route_decision"] = _route(summary, float(args.reconstruction_tolerance))
    row = {
        "assignment_scheme": summary["assignment_scheme"],
        "time_window_count": str(summary["time_window_count"]),
        **{key: _fmt(summary[key]) for key in SUMMARY_FIELDS if key in summary and key not in {"assignment_scheme", "time_window_count", "route_decision"}},
        "route_decision": summary["route_decision"],
    }
    return [row], summary


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

    metrics, status_by_run = sprint56._load_packet_local_metrics(table_rows, truth_meta)
    action_rows, _hysteresis_rows, _direction_rows, _by_shell_rows, sprint56_summary = sprint56._build_outputs(
        table_rows, truth_meta, metrics, status_by_run, args
    )
    time_rows, global_status_by_run = _build_global_rows(table_rows, truth_meta)
    summary_rows, summary = _build_summary(time_rows, sprint56_summary, args)
    summary["input_table_row_count"] = len(table_rows)
    summary["packet_local_action_row_count"] = len(action_rows)
    summary["inputs"] = [str(path) for path in args.inputs]
    summary["input_manifest_summaries"] = manifests
    summary["truth_metadata"] = list(truth_meta.values())
    summary["global_status_by_run"] = global_status_by_run
    summary["sprint56_route_decision"] = sprint56_summary.get("route_decision")
    summary["sprint56_action_small_fraction"] = sprint56_summary.get("action_small_fraction")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    time_path = args.out_dir / "ns_vessel_action_reconciliation_by_time.csv"
    summary_csv_path = args.out_dir / "ns_vessel_action_reconciliation_summary.csv"
    summary_path = args.out_dir / "ns_sprint57_vessel_action_reconciliation_summary.json"
    summary["ns_vessel_action_reconciliation_by_time_path"] = str(time_path)
    summary["ns_vessel_action_reconciliation_summary_path"] = str(summary_csv_path)
    _write_csv(time_path, TIME_FIELDS, time_rows)
    _write_csv(summary_csv_path, SUMMARY_FIELDS, summary_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint57_vessel_action_reconciliation_audit] wrote {time_path}")
    print(f"[ns_sprint57_vessel_action_reconciliation_audit] wrote {summary_csv_path}")
    print(f"[ns_sprint57_vessel_action_reconciliation_audit] wrote {summary_path}")
    print(
        "[ns_sprint57_vessel_action_reconciliation_audit] "
        f"route={summary['route_decision']} "
        f"epsilon_raw_positive_vs_covered={summary['epsilon_raw_positive_vs_covered']} "
        f"epsilon_raw_positive_vs_global={summary['epsilon_raw_positive_vs_global']} "
        f"promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
