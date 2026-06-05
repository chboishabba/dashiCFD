#!/usr/bin/env python3
"""Sprint 59 packet-local raw positive stretch-action audit.

Sprint 58 showed that packet-normalized stretch action is a non-additive
sum-of-ratios object.  Sprint 59 returns to the vessel-additive enstrophy
production integrand and computes packet-local

    A_raw_positive(P) = integral_P max(omega dot S omega, 0) dx dt.

The audit keeps the packet geometry used by Sprint 49/56 and records whether
the raw positive action has diagnostic shell decay.  It remains empirical: no
physical bridge, stretch absorption, or Clay/NS promotion is proved here.
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
    "A_raw_positive",
    "A_raw_negative",
    "A_raw_net",
    "A_raw_total",
    "packet_enstrophy",
    "packet_volume",
    "A_norm_enstrophy_weighted",
    "weighted_A_raw_positive",
    "lagrangian_trit_after_integration",
    "packet_local_mask_available",
    "raw_action_boundary",
]

BY_K_FIELDS = [
    "K",
    "packet_count",
    "A_raw_positive_total",
    "A_raw_positive_mean",
    "A_raw_negative_total",
    "A_raw_net_total",
    "packet_enstrophy_total",
    "packet_volume_total",
    "weighted_A_raw_positive_total",
    "sigma_raw_action_fit",
    "sigma_raw_action_mean_fit",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Sprint 49 material-parent output directories")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--truth-root", type=Path, default=Path("."), help="root used to resolve source_truth paths")
    p.add_argument("--raw-action-threshold", type=float, default=0.0)
    p.add_argument("--sigma-threshold", type=float, default=0.5)
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _label(net: float, threshold: float) -> str:
    if net > threshold:
        return "plus"
    if net < -threshold:
        return "minus"
    return "zero"


def _build_packet_rows(
    table_rows: list[dict[str, Any]],
    truth_meta: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    requested = sprint57._requested_packets_by_run_time(table_rows)
    rows: list[dict[str, Any]] = []
    status_by_run: dict[str, str] = {}
    last_time_by_run: dict[str, float] = {}
    boundary = (
        "Sprint 59 computes packet-local raw positive/negative omega dot S omega "
        "actions by voxelwise positive and negative parts. This is an additive "
        "diagnostic over the reconstructed Sprint 49 K_cell packet masks; "
        "summability, physical bridge, stretch absorption, and no-blowup remain "
        "unproved."
    )
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
            for k, packet_id in sorted(packet_keys):
                mask = sprint57._packet_mask(shell_map, cell_map, packet_id)
                if mask is None:
                    continue
                local_stretch = stretch[mask]
                raw_positive = float(np.sum(np.maximum(local_stretch, 0.0))) * dt
                raw_negative = float(np.sum(np.maximum(-local_stretch, 0.0))) * dt
                raw_net = float(np.sum(local_stretch)) * dt
                local_enstrophy = float(np.sum(enstrophy[mask]))
                volume = int(np.count_nonzero(mask))
                norm_weighted = raw_positive / (local_enstrophy + EPS)
                weighted = (2.0 ** (0.5 * float(k))) * raw_positive
                rows.append(
                    {
                        "run": run,
                        "time": _fmt(time),
                        "dt": _fmt(dt),
                        "K": str(int(k)),
                        "packet_id": packet_id,
                        "A_raw_positive": _fmt(raw_positive),
                        "A_raw_negative": _fmt(raw_negative),
                        "A_raw_net": _fmt(raw_net),
                        "A_raw_total": _fmt(raw_positive + raw_negative),
                        "packet_enstrophy": _fmt(local_enstrophy),
                        "packet_volume": str(volume),
                        "A_norm_enstrophy_weighted": _fmt(norm_weighted),
                        "weighted_A_raw_positive": _fmt(weighted),
                        "lagrangian_trit_after_integration": _label(raw_net, float(args.raw_action_threshold)),
                        "packet_local_mask_available": "true",
                        "raw_action_boundary": boundary,
                    }
                )
        status_by_run[run] = "raw_packet_stretch_action_available"
    return rows, status_by_run


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows)


def _route(summary: dict[str, Any]) -> str:
    if int(summary["packet_raw_action_row_count"]) == 0:
        return "RAW_ACTION_SOURCE_TRUTH_UNAVAILABLE"
    if bool(summary["does_raw_action_summability_gate_close"]):
        return "RAW_ACTION_SUMMABILITY_PROMISING_DIAGNOSTIC"
    return "RAW_ACTION_SUMMABILITY_BLOCKED"


def _build_by_k_rows(packet_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, float]:
    by_k: dict[int, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "positive": 0.0, "negative": 0.0, "net": 0.0, "enstrophy": 0.0, "volume": 0.0, "weighted": 0.0}
    )
    for row in packet_rows:
        k = int(row["K"])
        stats = by_k[k]
        stats["count"] += 1.0
        stats["positive"] += float(row["A_raw_positive"])
        stats["negative"] += float(row["A_raw_negative"])
        stats["net"] += float(row["A_raw_net"])
        stats["enstrophy"] += float(row["packet_enstrophy"])
        stats["volume"] += float(row["packet_volume"])
        stats["weighted"] += float(row["weighted_A_raw_positive"])
    sigma_total = sprint53._fit_sigma({k: v["positive"] for k, v in by_k.items()})
    sigma_mean = sprint53._fit_sigma({k: v["positive"] / max(v["count"], 1.0) for k, v in by_k.items()})
    rows: list[dict[str, Any]] = []
    for k, stats in sorted(by_k.items()):
        rows.append(
            {
                "K": str(k),
                "packet_count": str(int(stats["count"])),
                "A_raw_positive_total": _fmt(stats["positive"]),
                "A_raw_positive_mean": _fmt(stats["positive"] / max(stats["count"], 1.0)),
                "A_raw_negative_total": _fmt(stats["negative"]),
                "A_raw_net_total": _fmt(stats["net"]),
                "packet_enstrophy_total": _fmt(stats["enstrophy"]),
                "packet_volume_total": str(int(stats["volume"])),
                "weighted_A_raw_positive_total": _fmt(stats["weighted"]),
                "sigma_raw_action_fit": _fmt(sigma_total),
                "sigma_raw_action_mean_fit": _fmt(sigma_mean),
            }
        )
    return rows, sigma_total, sigma_mean


def _build_summary(
    packet_rows: list[dict[str, Any]],
    by_k_rows: list[dict[str, Any]],
    sigma_total: float,
    sigma_mean: float,
    status_by_run: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "contract": "ns_sprint59_raw_packet_stretch_action_artifact",
        "diagnostic_mode": "sprint59_raw_packet_positive_stretch_action",
        "packet_raw_action_row_count": len(packet_rows),
        "raw_action_by_k_row_count": len(by_k_rows),
        "A_raw_positive_total": _sum(packet_rows, "A_raw_positive"),
        "A_raw_negative_total": _sum(packet_rows, "A_raw_negative"),
        "A_raw_net_total": _sum(packet_rows, "A_raw_net"),
        "weighted_A_raw_positive_total": _sum(packet_rows, "weighted_A_raw_positive"),
        "packet_enstrophy_total": _sum(packet_rows, "packet_enstrophy"),
        "packet_volume_total": _sum(packet_rows, "packet_volume"),
        "sigma_raw_action_fit": sigma_total,
        "sigma_raw_action_mean_fit": sigma_mean,
        "sigma_threshold": float(args.sigma_threshold),
        "raw_action_threshold": float(args.raw_action_threshold),
        "does_raw_action_summability_gate_close": sigma_total > float(args.sigma_threshold),
        "raw_packet_action_additivity_proved": False,
        "weighted_raw_action_summability_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT59_RAW_PACKET_STRETCH_ACTION_DIAGNOSTIC",
        "status_by_run": status_by_run,
        "boundary": (
            "Sprint 59 measures the raw additive positive vortex-stretching "
            "source over reconstructed Sprint 49 packet masks. It does not "
            "prove continuum summability or any Clay/NS promotion."
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
    packet_rows, status_by_run = _build_packet_rows(table_rows, truth_meta, args)
    by_k_rows, sigma_total, sigma_mean = _build_by_k_rows(packet_rows)
    summary = _build_summary(packet_rows, by_k_rows, sigma_total, sigma_mean, status_by_run, args)
    summary["inputs"] = [str(path) for path in args.inputs]
    summary["input_manifest_summaries"] = manifests
    summary["truth_metadata"] = list(truth_meta.values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.out_dir / "ns_raw_packet_stretch_action.csv"
    by_k_path = args.out_dir / "ns_raw_packet_stretch_by_k.csv"
    summary_path = args.out_dir / "ns_sprint59_raw_packet_stretch_action_summary.json"
    summary["ns_raw_packet_stretch_action_path"] = str(packet_path)
    summary["ns_raw_packet_stretch_by_k_path"] = str(by_k_path)
    _write_csv(packet_path, PACKET_FIELDS, packet_rows)
    _write_csv(by_k_path, BY_K_FIELDS, by_k_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint59_raw_packet_stretch_action_audit] wrote {packet_path}")
    print(f"[ns_sprint59_raw_packet_stretch_action_audit] wrote {by_k_path}")
    print(f"[ns_sprint59_raw_packet_stretch_action_audit] wrote {summary_path}")
    print(
        "[ns_sprint59_raw_packet_stretch_action_audit] "
        f"route={summary['route_decision']} "
        f"sigma_raw={summary['sigma_raw_action_fit']} "
        f"promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
