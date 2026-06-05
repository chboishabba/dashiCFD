#!/usr/bin/env python3
"""Sprint 56 packet-local accumulated stretch-action audit.

Sprint 55 established that instantaneous red/blue strings should be demoted to
labels derived after integrating physical stretching.  Sprint 56 tightens the
measurement by reconstructing Sprint 49 packet support masks from the
deterministic ``K{K}_cell{cell}`` geometry and computing packet-local

    alpha_P(t) = integral_P omega dot S omega / (integral_P |omega|^2 + eps).

It also records a packet-local direction-change integral to separate angular
redirection from accumulated positive stretching.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import ns_sprint53_no2cycle_physical_amplitude_audit as sprint53
import ns_sprint54_no2cycle_resolution_cadence_audit as sprint54
import ns_sprint55_lagrangian_stretch_action_audit as sprint55


EPS = 1e-30

PACKET_ACTION_FIELDS = [
    "run",
    "material_lineage_id",
    "packet_id",
    "K",
    "t_start",
    "t_end",
    "step_count",
    "A_signed",
    "A_positive",
    "A_negative",
    "A_net",
    "direction_change_integral",
    "sign_flip_count",
    "weighted_A_positive",
    "mass_cycle_amplitude",
    "direct_stretch_cycle_amplitude",
    "packet_local_mask_available",
    "lagrangian_trit_after_integration",
    "is_dangerous",
    "action_small",
    "cadence",
    "N",
    "packet_confidence_min",
    "packet_confidence_mean",
    "packet_local_status",
    "action_boundary",
]

HYSTERESIS_FIELDS = [
    "run",
    "material_lineage_id",
    "time",
    "dt",
    "K",
    "packet_id",
    "instant_state",
    "packet_local_alpha",
    "increment_signed_action",
    "cumulative_signed_action",
    "cumulative_positive_action",
    "cumulative_negative_action",
    "direction_change_increment",
    "direction_change_integral",
    "instant_color_flip_count",
    "hysteresis_color_state",
    "packet_local_mask_available",
    "packet_local_status",
]

DIRECTION_FIELDS = [
    "run",
    "material_lineage_id",
    "K",
    "t_start",
    "t_end",
    "sign_flip_count",
    "direction_change_integral",
    "A_positive",
    "weighted_A_positive",
    "redirection_without_overwhelm",
    "packet_local_mask_available",
]

SUMMARY_FIELDS = [
    "K",
    "lineage_count",
    "A_positive_total",
    "weighted_A_positive_total",
    "direction_change_integral_total",
    "dangerous_lineage_count",
    "action_small_fraction",
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
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _parse_packet_id(packet_id: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"K(-?\d+)_cell(\d+)", packet_id)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _cell_map(n: int, packet_grid: int) -> np.ndarray:
    stride = max(1, int(n / packet_grid))
    i, j, k = np.indices((n, n, n), dtype=np.int64)
    return (i // stride) * packet_grid * packet_grid + (j // stride) * packet_grid + (k // stride)


def _packet_direction(frame: np.ndarray, enstrophy: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    local_e = enstrophy[mask]
    local_w = frame[mask]
    denom = float(np.sum(local_e))
    if denom <= EPS or local_w.size == 0:
        return None
    norm = np.linalg.norm(local_w, axis=1)
    valid = norm > EPS
    if not bool(np.any(valid)):
        return None
    unit = local_w[valid] / norm[valid, None]
    weights = local_e[valid]
    vec = np.sum(unit * weights[:, None], axis=0)
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm <= EPS:
        return None
    return vec / vec_norm


def _direction_delta(prev: np.ndarray | None, cur: np.ndarray | None) -> float:
    if prev is None or cur is None:
        return 0.0
    dot = float(np.clip(np.dot(prev, cur), -1.0, 1.0))
    return float(math.acos(dot))


def _needed_packets(table_rows: list[dict[str, Any]]) -> dict[str, dict[float, set[tuple[int, str]]]]:
    out: dict[str, dict[float, set[tuple[int, str]]]] = defaultdict(lambda: defaultdict(set))
    for row in table_rows:
        packet_id = str(row.get("child_packet_id", ""))
        parsed = _parse_packet_id(packet_id)
        if parsed is None:
            continue
        k, _cell = parsed
        out[str(row["run"])][float(row["time_float"])].add((k, packet_id))
    return out


def _load_packet_local_metrics(
    table_rows: list[dict[str, Any]],
    truth_meta: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, float, int, str], dict[str, Any]], dict[str, str]]:
    needed = _needed_packets(table_rows)
    metrics: dict[tuple[str, float, int, str], dict[str, Any]] = {}
    status_by_run: dict[str, str] = {}
    for meta in truth_meta.values():
        run = str(meta["run"])
        path = Path(str(meta.get("truth_path") or ""))
        if not path.exists():
            status_by_run[run] = "source_truth_unavailable"
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if "omega_snapshots" not in data.files or "velocity_snapshots" not in data.files:
                    status_by_run[run] = "truth_velocity_or_omega_unavailable"
                    continue
                omega = np.asarray(data["omega_snapshots"], dtype=np.float64)
                velocity = np.asarray(data["velocity_snapshots"], dtype=np.float64)
                steps = np.asarray(data["steps"], dtype=np.float64) if "steps" in data.files else np.arange(len(omega))
                meta_json = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
        except Exception as exc:  # pragma: no cover - defensive truth boundary.
            status_by_run[run] = f"truth_load_failed:{exc}"
            continue
        if omega.shape != velocity.shape or omega.ndim != 5 or omega.shape[-1] != 3:
            status_by_run[run] = "truth_shape_incompatible"
            continue
        n = int(omega.shape[1])
        packet_grid = int(meta.get("packet_grid") or 8)
        L = float(meta_json.get("domain_length") or (2.0 * math.pi))
        dt = float(meta_json.get("dt") or meta.get("dt") or 0.0)
        shell_map = sprint54._build_shell_map(n, L)
        cells = _cell_map(n, packet_grid)
        requested_by_time = needed.get(run, {})
        for t_idx, frame in enumerate(omega):
            time = float(steps[t_idx] * dt) if dt > 0.0 and t_idx < len(steps) else float(t_idx)
            requested = requested_by_time.get(time)
            if not requested:
                continue
            grad_u = sprint54._build_velocity_gradient(velocity[t_idx], L)
            stretch = np.einsum("...i,...ij,...j->...", frame, grad_u, frame)
            enstrophy = np.einsum("...i,...i->...", frame, frame)
            for k, packet_id in requested:
                parsed = _parse_packet_id(packet_id)
                if parsed is None:
                    continue
                parsed_k, cell = parsed
                if parsed_k != k:
                    continue
                mask = (shell_map == k) & (cells == cell)
                if not bool(np.any(mask)):
                    continue
                signed = float(np.sum(stretch[mask]))
                local_enstrophy = float(np.sum(enstrophy[mask]))
                alpha = signed / (local_enstrophy + EPS)
                metrics[(run, time, k, packet_id)] = {
                    "alpha": alpha,
                    "signed_stretch": signed,
                    "stretch_amplitude": abs(signed),
                    "packet_enstrophy": local_enstrophy,
                    "voxel_count": int(np.count_nonzero(mask)),
                    "direction": _packet_direction(frame, enstrophy, mask),
                }
        status_by_run[run] = "packet_local_mask_reconstructed_from_sprint49_packet_id_geometry"
    return metrics, status_by_run


def _label(net: float, threshold: float) -> str:
    return sprint55._label(net, threshold)


def _build_outputs(
    table_rows: list[dict[str, Any]],
    truth_meta: dict[str, dict[str, Any]],
    metrics: dict[tuple[str, float, int, str], dict[str, Any]],
    status_by_run: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    threshold = float(args.action_threshold)
    redirection_threshold = float(args.redirection_threshold)
    boundary = (
        "Sprint 56 reconstructs packet-local support masks from Sprint 49 "
        "K_cell packet IDs and packet_grid, then accumulates packet-local "
        "omega dot S omega / |omega|^2 and direction-change increments along "
        "material lineages. This is still diagnostic and does not prove "
        "continuum summability, physical bridge, stretch absorption, or no-blowup."
    )
    lineage_for_packet: dict[tuple[str, str], str] = {}
    lineage_data: dict[str, dict[str, Any]] = {}
    hysteresis_rows: list[dict[str, Any]] = []
    counter = 0
    rows = sorted(table_rows, key=lambda r: (str(r["run"]), float(r["time_float"]), str(r.get("child_packet_id", ""))))
    for row in rows:
        run = str(row["run"])
        parent_key = (run, str(row.get("parent_packet_id", "")))
        child_id = str(row.get("child_packet_id", ""))
        child_key = (run, child_id)
        lineage = lineage_for_packet.get(parent_key)
        if lineage is None:
            counter += 1
            lineage = f"{run}_lineage_{counter}"
        lineage_for_packet[child_key] = lineage
        data = lineage_data.setdefault(
            lineage,
            {
                "run": run,
                "material_lineage_id": lineage,
                "packet_id": child_id,
                "K_values": [],
                "t_start": float(row["time_float"]),
                "t_end": float(row["time_float"]),
                "step_count": 0,
                "signed": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "mass_proxy": 0.0,
                "direct_amp": 0.0,
                "dir_total": 0.0,
                "last_dir": None,
                "last_time": None,
                "last_state": "",
                "sign_flip_count": 0,
                "conf": [],
                "available_steps": 0,
                "status": status_by_run.get(run, "source_truth_unavailable"),
            },
        )
        k = int(row["K_child_int"])
        time = float(row["time_float"])
        meta = truth_meta.get(run, {})
        fallback_dt = float(meta.get("save_every") or 0) * float(meta.get("dt") or 0.0)
        if fallback_dt <= EPS:
            fallback_dt = float(row["dt_float"])
        last_time = data.get("last_time")
        if last_time is None:
            dt = fallback_dt
        else:
            dt = max(time - float(last_time), fallback_dt if fallback_dt > EPS else float(row["dt_float"]))
        data["last_time"] = time
        metric = metrics.get((run, time, k, child_id))
        alpha = float(metric["alpha"]) if metric is not None else 0.0
        inc = alpha * dt
        pos = max(inc, 0.0)
        neg = max(-inc, 0.0)
        direction = metric.get("direction") if metric is not None else None
        dir_inc = _direction_delta(data["last_dir"], direction)
        if direction is not None:
            data["last_dir"] = direction
        state = str(row.get("child_state", ""))
        if data["last_state"] and state and data["last_state"] != state:
            data["sign_flip_count"] += 1
        data["last_state"] = state
        data["packet_id"] = child_id
        data["K_values"].append(k)
        data["t_start"] = min(float(data["t_start"]), time)
        data["t_end"] = max(float(data["t_end"]), time)
        data["step_count"] += 1
        data["signed"] += inc
        data["positive"] += pos
        data["negative"] += neg
        data["mass_proxy"] += float(row["weighted_mass_float"])
        data["direct_amp"] += float(metric["stretch_amplitude"]) if metric is not None else 0.0
        data["dir_total"] += dir_inc
        data["conf"].append(float(row["parent_confidence_float"]))
        data["status"] = status_by_run.get(run, "source_truth_unavailable")
        data["available_steps"] += 1 if metric is not None else 0
        hysteresis_rows.append(
            {
                "run": run,
                "material_lineage_id": lineage,
                "time": _fmt(time),
                "dt": _fmt(dt),
                "K": str(k),
                "packet_id": child_id,
                "instant_state": state,
                "packet_local_alpha": _fmt(alpha) if metric is not None else "unavailable",
                "increment_signed_action": _fmt(inc) if metric is not None else "unavailable",
                "cumulative_signed_action": _fmt(float(data["signed"])),
                "cumulative_positive_action": _fmt(float(data["positive"])),
                "cumulative_negative_action": _fmt(float(data["negative"])),
                "direction_change_increment": _fmt(dir_inc) if metric is not None else "unavailable",
                "direction_change_integral": _fmt(float(data["dir_total"])),
                "instant_color_flip_count": str(int(data["sign_flip_count"])),
                "hysteresis_color_state": _label(float(data["signed"]), threshold),
                "packet_local_mask_available": _fmt(metric is not None),
                "packet_local_status": str(data["status"]),
            }
        )

    action_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    summary_by_k: dict[int, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "positive": 0.0, "weighted": 0.0, "direction": 0.0, "danger": 0.0, "small": 0.0}
    )
    by_k_weighted: dict[int, float] = defaultdict(float)
    available_lineages = 0
    dangerous = 0
    small = 0
    redirection_without_overwhelm = 0
    for data in lineage_data.values():
        final_k = int(list(data["K_values"])[-1]) if data["K_values"] else 0
        weighted_positive = (2.0 ** (0.5 * float(final_k))) * float(data["positive"])
        label = _label(float(data["signed"]), threshold)
        is_dangerous = float(data["positive"]) > threshold
        action_small = not is_dangerous
        available = int(data["available_steps"]) > 0
        redirection_small = bool(float(data["dir_total"]) >= redirection_threshold and action_small)
        available_lineages += 1 if available else 0
        dangerous += 1 if is_dangerous else 0
        small += 1 if action_small else 0
        redirection_without_overwhelm += 1 if redirection_small else 0
        conf = list(data["conf"])
        meta = truth_meta.get(str(data["run"]), {})
        action_row = {
            "run": str(data["run"]),
            "material_lineage_id": str(data["material_lineage_id"]),
            "packet_id": str(data["packet_id"]),
            "K": str(final_k),
            "t_start": _fmt(float(data["t_start"])),
            "t_end": _fmt(float(data["t_end"])),
            "step_count": str(int(data["step_count"])),
            "A_signed": _fmt(float(data["signed"])),
            "A_positive": _fmt(float(data["positive"])),
            "A_negative": _fmt(float(data["negative"])),
            "A_net": _fmt(float(data["signed"])),
            "direction_change_integral": _fmt(float(data["dir_total"])),
            "sign_flip_count": str(int(data["sign_flip_count"])),
            "weighted_A_positive": _fmt(weighted_positive),
            "mass_cycle_amplitude": _fmt(float(data["mass_proxy"])),
            "direct_stretch_cycle_amplitude": _fmt(float(data["direct_amp"])) if available else "unavailable",
            "packet_local_mask_available": _fmt(available),
            "lagrangian_trit_after_integration": label,
            "is_dangerous": _fmt(is_dangerous),
            "action_small": _fmt(action_small),
            "cadence": str(int(meta.get("save_every") or 0)),
            "N": str(int(meta.get("N") or sprint55._fallback_n_from_name(str(data["run"])))),
            "packet_confidence_min": _fmt(min(conf) if conf else 0.0),
            "packet_confidence_mean": _fmt(sum(conf) / len(conf) if conf else 0.0),
            "packet_local_status": str(data["status"]),
            "action_boundary": boundary,
        }
        action_rows.append(action_row)
        direction_rows.append(
            {
                "run": action_row["run"],
                "material_lineage_id": action_row["material_lineage_id"],
                "K": action_row["K"],
                "t_start": action_row["t_start"],
                "t_end": action_row["t_end"],
                "sign_flip_count": action_row["sign_flip_count"],
                "direction_change_integral": action_row["direction_change_integral"],
                "A_positive": action_row["A_positive"],
                "weighted_A_positive": action_row["weighted_A_positive"],
                "redirection_without_overwhelm": _fmt(redirection_small),
                "packet_local_mask_available": action_row["packet_local_mask_available"],
            }
        )
        stats = summary_by_k[final_k]
        stats["count"] += 1.0
        stats["positive"] += float(data["positive"])
        stats["weighted"] += weighted_positive
        stats["direction"] += float(data["dir_total"])
        stats["danger"] += 1.0 if is_dangerous else 0.0
        stats["small"] += 1.0 if action_small else 0.0
        by_k_weighted[final_k] += weighted_positive

    summary_rows: list[dict[str, Any]] = []
    for k, stats in sorted(summary_by_k.items()):
        summary_rows.append(
            {
                "K": str(k),
                "lineage_count": str(int(stats["count"])),
                "A_positive_total": _fmt(stats["positive"]),
                "weighted_A_positive_total": _fmt(stats["weighted"]),
                "direction_change_integral_total": _fmt(stats["direction"]),
                "dangerous_lineage_count": str(int(stats["danger"])),
                "action_small_fraction": _fmt(stats["small"] / max(stats["count"], 1.0)),
            }
        )

    sigma = sprint53._fit_sigma(by_k_weighted)
    action_small_fraction = small / max(len(action_rows), 1)
    packet_local_fraction = available_lineages / max(len(action_rows), 1)
    summability_gate = sigma > float(args.sigma_threshold)
    action_gate = bool(action_rows and action_small_fraction >= float(args.action_small_majority))
    if available_lineages == 0:
        route = "PACKET_LOCAL_MASK_JOIN_INSUFFICIENT"
    elif summability_gate:
        route = "PACKET_LOCAL_ACTION_SUMMABILITY_PROMISING_DIAGNOSTIC"
    elif action_gate:
        route = "PACKET_LOCAL_REDIRECTION_WITHOUT_OVERWHELM_DIAGNOSTIC"
    else:
        route = "PACKET_LOCAL_ACTION_SUMMABILITY_BLOCKED"
    summary = {
        "contract": "ns_sprint56_packet_local_action_artifact",
        "diagnostic_mode": "sprint56_packet_local_accumulated_stretch_action",
        "input_table_row_count": len(table_rows),
        "packet_local_action_row_count": len(action_rows),
        "packet_local_hysteresis_row_count": len(hysteresis_rows),
        "direction_change_row_count": len(direction_rows),
        "action_summary_row_count": len(summary_rows),
        "packet_local_available_lineage_count": available_lineages,
        "packet_local_available_fraction": packet_local_fraction,
        "action_threshold": threshold,
        "action_small_majority": float(args.action_small_majority),
        "redirection_threshold": redirection_threshold,
        "sigma_threshold": float(args.sigma_threshold),
        "A_positive_total": sum(float(row["A_positive"]) for row in action_rows),
        "weighted_A_positive_total": sum(float(row["weighted_A_positive"]) for row in action_rows),
        "direction_change_integral_total": sum(float(row["direction_change_integral"]) for row in action_rows),
        "redirection_without_overwhelm_count": redirection_without_overwhelm,
        "action_small_fraction": action_small_fraction,
        "dangerous_lineage_count": dangerous,
        "sigma_packet_local_action_fit": sigma,
        "does_packet_local_action_gate_close": action_gate,
        "does_packet_local_action_summability_gate_close": summability_gate,
        "packet_local_mask_reconstruction_available": bool(available_lineages),
        "packet_local_action_gate_proved": False,
        "weighted_packet_local_action_summability_proved": False,
        "direction_change_separation_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT56_PACKET_LOCAL_ACTION_DIAGNOSTIC",
        "route_decision": route,
        "packet_local_boundary": boundary,
        "packet_local_status": (
            "packet_local_mask_reconstructed_from_sprint49_packet_id_geometry"
            if available_lineages
            else "packet_local_mask_reconstruction_unavailable"
        ),
        "receipt_alignment": "DASHI.Physics.Closure.ClaySprintFiftySixPacketLocalStretchActionAuditReceipt",
    }
    return action_rows, hysteresis_rows, direction_rows, summary_rows, summary


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
    metrics, status_by_run = _load_packet_local_metrics(table_rows, truth_meta)
    action_rows, hysteresis_rows, direction_rows, summary_rows, summary = _build_outputs(
        table_rows, truth_meta, metrics, status_by_run, args
    )
    summary["inputs"] = [str(path) for path in args.inputs]
    summary["input_manifest_summaries"] = manifests
    summary["truth_metadata"] = list(truth_meta.values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    action_path = args.out_dir / "ns_packet_local_lagrangian_action.csv"
    hysteresis_path = args.out_dir / "ns_packet_local_action_hysteresis.csv"
    direction_path = args.out_dir / "ns_direction_change_separation.csv"
    by_shell_path = args.out_dir / "ns_packet_local_action_by_shell.csv"
    summary_path = args.out_dir / "ns_sprint56_packet_local_action_summary.json"
    summary["ns_packet_local_lagrangian_action_path"] = str(action_path)
    summary["ns_packet_local_action_hysteresis_path"] = str(hysteresis_path)
    summary["ns_direction_change_separation_path"] = str(direction_path)
    summary["ns_packet_local_action_by_shell_path"] = str(by_shell_path)
    _write_csv(action_path, PACKET_ACTION_FIELDS, action_rows)
    _write_csv(hysteresis_path, HYSTERESIS_FIELDS, hysteresis_rows)
    _write_csv(direction_path, DIRECTION_FIELDS, direction_rows)
    _write_csv(by_shell_path, SUMMARY_FIELDS, summary_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint56_packet_local_stretch_action_audit] wrote {action_path}")
    print(f"[ns_sprint56_packet_local_stretch_action_audit] wrote {hysteresis_path}")
    print(f"[ns_sprint56_packet_local_stretch_action_audit] wrote {direction_path}")
    print(f"[ns_sprint56_packet_local_stretch_action_audit] wrote {by_shell_path}")
    print(f"[ns_sprint56_packet_local_stretch_action_audit] wrote {summary_path}")
    print(
        "[ns_sprint56_packet_local_stretch_action_audit] "
        f"route={summary['route_decision']} "
        f"action_small_fraction={summary['action_small_fraction']} "
        f"sigma={summary['sigma_packet_local_action_fit']} "
        f"promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
