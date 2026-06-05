#!/usr/bin/env python3
"""Sprint 55 Lagrangian accumulated stretch-action audit.

This replaces instantaneous red/green/blue string danger with an accumulated
material-lineage action proxy:

    A = integral (omega dot S omega) / (|omega|^2 + eps) dt

The v1 implementation consumes Sprint 49 material-parent tables and truth
snapshots.  It follows packet IDs through the recorded parent -> child material
lineage and joins shell/time direct stretching rates from the truth artifact.
Packet-local support masks are not present in Sprint 49 artifacts, so the
reported action is a shell/time action along material packet lineages, not a
theorem-grade packet-local integral.
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


EPS = 1e-30

ACTION_FIELDS = [
    "run",
    "material_lineage_id",
    "packet_id",
    "K",
    "t_start",
    "t_end",
    "step_count",
    "sign_flip_count",
    "signed_stretch_integral",
    "positive_stretch_integral",
    "negative_stretch_integral",
    "normalized_stretch_action",
    "weighted_A_positive",
    "mass_amplitude_proxy",
    "direct_stretch_amplitude",
    "lagrangian_label",
    "is_dangerous",
    "action_small",
    "cadence",
    "N",
    "packet_confidence_min",
    "packet_confidence_mean",
    "direct_stretch_status",
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
    "instant_alpha",
    "increment_signed_action",
    "cumulative_signed_action",
    "cumulative_positive_action",
    "cumulative_negative_action",
    "instant_color_flip_count",
    "hysteresis_color_state",
    "direct_stretch_status",
]

SUMMARY_FIELDS = [
    "K",
    "lineage_count",
    "signed_action_total",
    "positive_action_total",
    "negative_action_total",
    "weighted_positive_action_total",
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
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _num(value: Any) -> float:
    return sprint53._num(value)


def _fallback_n_from_name(run: str) -> int:
    match = re.search(r"N(\d+)", run)
    return int(match.group(1)) if match else 0


def _truth_path(summary: dict[str, Any], truth_root: Path) -> Path | None:
    source_truth = str(summary.get("source_truth") or "")
    if not source_truth:
        return None
    path = Path(source_truth)
    if not path.is_absolute():
        path = truth_root / path
    return path


def _truth_meta_by_run(inputs: list[Path], truth_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for input_dir in inputs:
        summary = json.loads((input_dir / "ns_material_parent_summary.json").read_text(encoding="utf-8"))
        path = _truth_path(summary, truth_root)
        meta_json: dict[str, Any] = {}
        if path is not None and path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    if "meta_json" in data:
                        meta_json = json.loads(str(data["meta_json"]))
            except Exception as exc:  # pragma: no cover - defensive metadata boundary.
                meta_json = {"meta_load_error": str(exc)}
        out[input_dir.name] = {
            "run": input_dir.name,
            "input_dir": str(input_dir),
            "source_truth": str(summary.get("source_truth") or ""),
            "truth_path": str(path) if path is not None else "",
            "N": int(meta_json.get("N") or _fallback_n_from_name(input_dir.name)),
            "save_every": int(meta_json.get("save_every") or 0),
            "dt": float(meta_json.get("dt") or 0.0),
            "snapshots": int(meta_json.get("snapshots") or 0),
        }
    return out


def _load_shell_action_rates(
    truth_meta: dict[str, dict[str, Any]]
) -> tuple[dict[tuple[str, float, int], dict[str, float]], dict[str, str]]:
    rates: dict[tuple[str, float, int], dict[str, float]] = {}
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
        dt = float(meta_json.get("dt") or meta.get("dt") or 0.0)
        L = float(meta_json.get("domain_length") or (2.0 * math.pi))
        shell_map = sprint54._build_shell_map(int(omega.shape[1]), L)
        shell_ids = sorted(int(k) for k in np.unique(shell_map))
        for t_idx, frame in enumerate(omega):
            time = float(steps[t_idx] * dt) if dt > 0.0 and t_idx < len(steps) else float(t_idx)
            grad_u = sprint54._build_velocity_gradient(velocity[t_idx], L)
            stretch = np.einsum("...i,...ij,...j->...", frame, grad_u, frame)
            enstrophy = np.einsum("...i,...i->...", frame, frame)
            for k in shell_ids:
                mask = shell_map == k
                signed = float(np.sum(stretch[mask]))
                denom = float(np.sum(enstrophy[mask])) + EPS
                alpha = signed / denom
                rates[(run, time, k)] = {
                    "alpha": alpha,
                    "stretching_amplitude": abs(signed),
                    "shell_enstrophy": denom,
                }
        status_by_run[run] = "shell_time_normalized_stretch_available_packet_mask_join_unavailable"
    return rates, status_by_run


def _fit_sigma(by_k: dict[int, float]) -> float:
    return sprint53._fit_sigma(by_k)


def _label(net: float, threshold: float) -> str:
    if net > threshold:
        return "red"
    if net < -threshold:
        return "blue"
    return "green"


def _build_action_rows(
    table_rows: list[dict[str, Any]],
    truth_meta: dict[str, dict[str, Any]],
    rates: dict[tuple[str, float, int], dict[str, float]],
    status_by_run: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    threshold = float(args.action_threshold)
    boundary = (
        "Sprint 55 v1 accumulates shell/time normalized omega dot S omega along "
        "Sprint 49 material parent packet lineages. Ternary colors are derived "
        "after integration. Packet-local support masks are unavailable, so this "
        "is not theorem-grade packet-local stretch action."
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
                "sign_flip_count": 0,
                "last_state": "",
                "signed": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "mass_proxy": 0.0,
                "direct_amp": 0.0,
                "conf": [],
                "status": status_by_run.get(run, "source_truth_unavailable"),
            },
        )
        k = int(row["K_child_int"])
        time = float(row["time_float"])
        dt = float(row["dt_float"])
        direct = rates.get((run, time, k))
        alpha = float(direct["alpha"]) if direct is not None else 0.0
        inc = alpha * dt
        pos = max(inc, 0.0)
        neg = max(-inc, 0.0)
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
        data["direct_amp"] += float(direct["stretching_amplitude"]) if direct is not None else 0.0
        data["conf"].append(float(row["parent_confidence_float"]))
        data["status"] = status_by_run.get(run, "source_truth_unavailable")
        hysteresis_rows.append(
            {
                "run": run,
                "material_lineage_id": lineage,
                "time": _fmt(time),
                "dt": _fmt(dt),
                "K": str(k),
                "packet_id": child_id,
                "instant_state": state,
                "instant_alpha": _fmt(alpha) if direct is not None else "unavailable",
                "increment_signed_action": _fmt(inc) if direct is not None else "unavailable",
                "cumulative_signed_action": _fmt(float(data["signed"])),
                "cumulative_positive_action": _fmt(float(data["positive"])),
                "cumulative_negative_action": _fmt(float(data["negative"])),
                "instant_color_flip_count": str(int(data["sign_flip_count"])),
                "hysteresis_color_state": _label(float(data["signed"]), threshold),
                "direct_stretch_status": data["status"],
            }
        )

    action_rows: list[dict[str, Any]] = []
    summary_by_k: dict[int, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "signed": 0.0, "positive": 0.0, "negative": 0.0, "weighted": 0.0, "danger": 0.0, "small": 0.0}
    )
    by_k_weighted: dict[int, float] = defaultdict(float)
    dangerous = 0
    small = 0
    direct_available = 0
    for data in lineage_data.values():
        ks = list(data["K_values"])
        final_k = int(ks[-1]) if ks else 0
        weighted_positive = (2.0 ** (0.5 * float(final_k))) * float(data["positive"])
        label = _label(float(data["signed"]), threshold)
        is_dangerous = float(data["positive"]) > threshold
        action_small = not is_dangerous
        dangerous += 1 if is_dangerous else 0
        small += 1 if action_small else 0
        data_has_direct = str(data["status"]).startswith("shell_time_normalized_stretch_available")
        if data_has_direct:
            direct_available += 1
        conf = list(data["conf"])
        meta = truth_meta.get(str(data["run"]), {})
        action_rows.append(
            {
                "run": str(data["run"]),
                "material_lineage_id": str(data["material_lineage_id"]),
                "packet_id": str(data["packet_id"]),
                "K": str(final_k),
                "t_start": _fmt(float(data["t_start"])),
                "t_end": _fmt(float(data["t_end"])),
                "step_count": str(int(data["step_count"])),
                "sign_flip_count": str(int(data["sign_flip_count"])),
                "signed_stretch_integral": _fmt(float(data["signed"])),
                "positive_stretch_integral": _fmt(float(data["positive"])),
                "negative_stretch_integral": _fmt(float(data["negative"])),
                "normalized_stretch_action": _fmt(float(data["signed"])),
                "weighted_A_positive": _fmt(weighted_positive),
                "mass_amplitude_proxy": _fmt(float(data["mass_proxy"])),
                "direct_stretch_amplitude": _fmt(float(data["direct_amp"])) if data_has_direct else "unavailable",
                "lagrangian_label": label,
                "is_dangerous": _fmt(is_dangerous),
                "action_small": _fmt(action_small),
                "cadence": str(int(meta.get("save_every") or 0)),
                "N": str(int(meta.get("N") or _fallback_n_from_name(str(data["run"])))),
                "packet_confidence_min": _fmt(min(conf) if conf else 0.0),
                "packet_confidence_mean": _fmt(sum(conf) / len(conf) if conf else 0.0),
                "direct_stretch_status": str(data["status"]),
                "action_boundary": boundary,
            }
        )
        stats = summary_by_k[final_k]
        stats["count"] += 1.0
        stats["signed"] += float(data["signed"])
        stats["positive"] += float(data["positive"])
        stats["negative"] += float(data["negative"])
        stats["weighted"] += weighted_positive
        stats["danger"] += 1.0 if is_dangerous else 0.0
        stats["small"] += 1.0 if action_small else 0.0
        by_k_weighted[final_k] += weighted_positive

    summary_rows: list[dict[str, Any]] = []
    for k, stats in sorted(summary_by_k.items()):
        summary_rows.append(
            {
                "K": str(k),
                "lineage_count": str(int(stats["count"])),
                "signed_action_total": _fmt(stats["signed"]),
                "positive_action_total": _fmt(stats["positive"]),
                "negative_action_total": _fmt(stats["negative"]),
                "weighted_positive_action_total": _fmt(stats["weighted"]),
                "dangerous_lineage_count": str(int(stats["danger"])),
                "action_small_fraction": _fmt(stats["small"] / max(stats["count"], 1.0)),
            }
        )
    sigma = _fit_sigma(by_k_weighted)
    action_small_fraction = small / max(len(action_rows), 1)
    gate_closes = bool(action_rows and action_small_fraction >= float(args.action_small_majority))
    summability_gate = sigma > float(args.sigma_threshold)
    if not direct_available:
        route = "PACKET_MASK_JOIN_INSUFFICIENT"
    elif summability_gate:
        route = "LAGRANGIAN_STRETCH_ACTION_SUMMABILITY_PROMISING_DIAGNOSTIC"
    elif gate_closes:
        route = "LAGRANGIAN_STRETCH_ACTION_SMALL_DIAGNOSTIC"
    else:
        route = "LAGRANGIAN_ACTION_SUMMABILITY_BLOCKED"
    summary = {
        "contract": "ns_sprint55_lagrangian_action_artifact",
        "diagnostic_mode": "sprint55_lagrangian_accumulated_stretch_action",
        "input_table_row_count": len(table_rows),
        "lagrangian_action_row_count": len(action_rows),
        "hysteresis_row_count": len(hysteresis_rows),
        "action_summary_row_count": len(summary_rows),
        "material_lineage_count": len(action_rows),
        "direct_stretch_available_lineage_count": direct_available,
        "action_threshold": threshold,
        "action_small_majority": float(args.action_small_majority),
        "sigma_threshold": float(args.sigma_threshold),
        "positive_action_total": sum(float(row["positive_stretch_integral"]) for row in action_rows),
        "weighted_positive_action_total": sum(float(row["weighted_A_positive"]) for row in action_rows),
        "action_small_fraction": action_small_fraction,
        "dangerous_lineage_count": dangerous,
        "sigma_action_fit": sigma,
        "does_lagrangian_action_gate_close": gate_closes,
        "does_lagrangian_action_summability_gate_close": summability_gate,
        "lagrangian_action_gate_proved": False,
        "weighted_action_summability_proved": False,
        "packet_local_stretch_action_available": False,
        "packet_local_stretch_action_proved": False,
        "color_string_proxy_demoted": True,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT55_LAGRANGIAN_ACTION_DIAGNOSTIC",
        "route_decision": route,
        "action_boundary": boundary,
        "direct_stretch_status": (
            "shell_time_normalized_stretch_available_packet_mask_join_unavailable"
            if direct_available
            else "source_truth_or_direct_stretch_unavailable"
        ),
        "receipt_alignment": "DASHI.Physics.Closure.ClaySprintFiftyFiveLagrangianStretchActionAuditReceipt",
    }
    return action_rows, hysteresis_rows, summary_rows, summary


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    table_rows, _summary_rows, manifests = sprint53._read_inputs(args.inputs)
    truth_meta = _truth_meta_by_run(args.inputs, args.truth_root)
    rates, status_by_run = _load_shell_action_rates(truth_meta)
    action_rows, hysteresis_rows, summary_rows, summary = _build_action_rows(
        table_rows, truth_meta, rates, status_by_run, args
    )
    summary["inputs"] = [str(path) for path in args.inputs]
    summary["input_manifest_summaries"] = manifests
    summary["truth_metadata"] = list(truth_meta.values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    action_path = args.out_dir / "ns_lagrangian_stretch_action.csv"
    hysteresis_path = args.out_dir / "ns_packet_action_hysteresis.csv"
    summary_csv_path = args.out_dir / "ns_lagrangian_action_by_shell.csv"
    summary_path = args.out_dir / "ns_sprint55_lagrangian_action_summary.json"
    summary["ns_lagrangian_stretch_action_path"] = str(action_path)
    summary["ns_packet_action_hysteresis_path"] = str(hysteresis_path)
    summary["ns_lagrangian_action_by_shell_path"] = str(summary_csv_path)

    _write_csv(action_path, ACTION_FIELDS, action_rows)
    _write_csv(hysteresis_path, HYSTERESIS_FIELDS, hysteresis_rows)
    _write_csv(summary_csv_path, SUMMARY_FIELDS, summary_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint55_lagrangian_stretch_action_audit] wrote {action_path}")
    print(f"[ns_sprint55_lagrangian_stretch_action_audit] wrote {hysteresis_path}")
    print(f"[ns_sprint55_lagrangian_stretch_action_audit] wrote {summary_csv_path}")
    print(f"[ns_sprint55_lagrangian_stretch_action_audit] wrote {summary_path}")
    print(
        "[ns_sprint55_lagrangian_stretch_action_audit] "
        f"route={summary['route_decision']} "
        f"action_small_fraction={summary['action_small_fraction']} "
        f"sigma={summary['sigma_action_fit']} "
        f"promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
