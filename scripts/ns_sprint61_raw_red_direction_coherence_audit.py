#!/usr/bin/env python3
"""Sprint 61 raw-red direction-coherence anatomy audit.

Sprint 59/60 left the source-budget route blocked: raw positive
``omega dot S omega`` action is the right additive object, but its shell fit
remains nearly flat under Euclidean, smoothed, and provisional BT shell
assignments.  Sprint 61 pivots to the CFM-style question:

    are the high raw-red packets geometrically coherent enough that vortex
    stretching is depleted rather than concentrating?

This producer consumes the Sprint 59 raw-action packet CSV, rebuilds Sprint 49
packet masks from the source truth, and measures vorticity-direction coherence
plus simple concentration/Beltrami proxies on the high raw-red population.  It
is diagnostic only; no CFM theorem, physical bridge, stretch absorption,
no-blowup, or Clay/NS promotion is proved.
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
    "weighted_A_raw_positive",
    "packet_enstrophy",
    "packet_volume",
    "omega_magnitude_mean",
    "omega_magnitude_max",
    "direction_coherence_mean",
    "direction_coherence_min",
    "direction_lipschitz_proxy",
    "direction_change_increment",
    "direction_change_integral",
    "beltrami_defect_mean",
    "beltrami_defect_normalized_mean",
    "parent_confidence",
    "parent_relation",
    "parent_state",
    "child_state",
    "raw_action_trit",
    "top_raw_red_rank",
    "high_raw_red_selected",
    "coherence_route_label",
]

BY_K_FIELDS = [
    "K",
    "packet_count",
    "A_raw_positive_total",
    "weighted_A_raw_positive_total",
    "direction_coherence_mean",
    "direction_coherence_min",
    "direction_lipschitz_proxy_mean",
    "omega_magnitude_max",
    "beltrami_defect_normalized_mean",
    "incoherent_packet_fraction",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Sprint 49 material-parent output directories")
    p.add_argument("--raw-action-csv", type=Path, required=True, help="Sprint 59 ns_raw_packet_stretch_action.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--truth-root", type=Path, default=Path("."), help="root used to resolve source_truth paths")
    p.add_argument("--top-fraction", type=float, default=0.05, help="fraction of raw-red packets selected by weighted_A_raw_positive")
    p.add_argument("--min-selected", type=int, default=32)
    p.add_argument("--max-selected", type=int, default=5000)
    p.add_argument("--coherence-threshold", type=float, default=0.80)
    p.add_argument("--incoherent-fraction-threshold", type=float, default=0.20)
    p.add_argument("--lipschitz-threshold", type=float, default=2.0)
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _read_raw_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"run", "time", "dt", "K", "packet_id", "A_raw_positive", "A_raw_net", "weighted_A_raw_positive"}
    missing = sorted(required.difference(rows[0].keys() if rows else []))
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    return rows


def _read_parent_rows(inputs: list[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for input_dir in inputs:
        path = input_dir / "ns_material_parent_table.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                out[(input_dir.name, row.get("time", ""), row.get("child_packet_id", ""))] = row
    return out


def _select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    red_rows = [row for row in rows if float(row.get("A_raw_positive") or 0.0) > 0.0]
    red_rows.sort(key=lambda row: float(row.get("weighted_A_raw_positive") or 0.0), reverse=True)
    if not red_rows:
        return []
    count = max(int(args.min_selected), int(math.ceil(float(args.top_fraction) * len(red_rows))))
    count = min(max(1, count), int(args.max_selected), len(red_rows))
    return red_rows[:count]


def _unit_field(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.linalg.norm(frame, axis=-1)
    unit = np.zeros_like(frame, dtype=np.float64)
    valid = mag > EPS
    unit[valid] = frame[valid] / mag[valid, None]
    return unit, mag


def _frame_direction_cache(frame: np.ndarray, L: float) -> dict[str, np.ndarray]:
    unit, mag = _unit_field(frame)
    spacing = float(L) / float(frame.shape[0])
    grads = np.gradient(unit, spacing, axis=(0, 1, 2), edge_order=1)
    grad_mag = np.zeros(mag.shape, dtype=np.float64)
    for grad in grads:
        grad_mag += np.sum(grad * grad, axis=-1)
    return {"unit": unit, "mag": mag, "direction_grad_mag": np.sqrt(grad_mag)}


def _packet_direction_stats(
    frame: np.ndarray,
    velocity: np.ndarray,
    mask: np.ndarray,
    L: float,
    frame_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if frame_cache is None:
        frame_cache = _frame_direction_cache(frame, L)
    unit = frame_cache["unit"]
    mag = frame_cache["mag"]
    local_mag = mag[mask]
    local_unit = unit[mask]
    valid = local_mag > EPS
    if not bool(np.any(valid)):
        return {
            "direction": None,
            "omega_magnitude_mean": 0.0,
            "omega_magnitude_max": 0.0,
            "direction_coherence_mean": 0.0,
            "direction_coherence_min": 0.0,
            "direction_lipschitz_proxy": 0.0,
            "beltrami_defect_mean": 0.0,
            "beltrami_defect_normalized_mean": 0.0,
        }
    local_unit = local_unit[valid]
    local_mag = local_mag[valid]
    weights = local_mag * local_mag
    mean_vec = np.sum(local_unit * weights[:, None], axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    direction = mean_vec / mean_norm if mean_norm > EPS else None
    if direction is None:
        align = np.zeros_like(local_mag)
    else:
        align = np.abs(np.clip(np.einsum("ij,j->i", local_unit, direction), -1.0, 1.0))
    coherence_mean = float(np.average(align, weights=weights)) if float(np.sum(weights)) > EPS else 0.0
    coherence_min = float(np.min(align)) if align.size else 0.0
    direction_grad_mag = frame_cache["direction_grad_mag"]
    lipschitz = float(np.mean(direction_grad_mag[mask])) if bool(np.any(mask)) else 0.0
    cross = np.linalg.norm(np.cross(velocity[mask], frame[mask]), axis=1)
    speed = np.linalg.norm(velocity[mask], axis=1)
    omega_mag = np.linalg.norm(frame[mask], axis=1)
    defect_mean = float(np.mean(cross)) if cross.size else 0.0
    defect_norm = cross / (speed * omega_mag + EPS)
    return {
        "direction": direction,
        "omega_magnitude_mean": float(np.mean(local_mag)),
        "omega_magnitude_max": float(np.max(local_mag)),
        "direction_coherence_mean": coherence_mean,
        "direction_coherence_min": coherence_min,
        "direction_lipschitz_proxy": lipschitz,
        "beltrami_defect_mean": defect_mean,
        "beltrami_defect_normalized_mean": float(np.mean(defect_norm)) if defect_norm.size else 0.0,
    }


def _direction_delta(prev: np.ndarray | None, cur: np.ndarray | None) -> float:
    if prev is None or cur is None:
        return 0.0
    return float(math.acos(float(np.clip(np.dot(prev, cur), -1.0, 1.0))))


def _load_metrics(selected: list[dict[str, str]], truth_meta: dict[str, dict[str, Any]]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, str]]:
    needed: dict[str, dict[float, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in selected:
        needed[str(row["run"])][float(row["time"])].add(str(row["packet_id"]))
    metrics: dict[tuple[str, str, str], dict[str, Any]] = {}
    status_by_run: dict[str, str] = {}
    for meta in truth_meta.values():
        run = str(meta["run"])
        path = Path(str(meta.get("truth_path") or ""))
        if not path.exists():
            status_by_run[run] = "source_truth_unavailable"
            continue
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
            packet_ids = needed.get(run, {}).get(time, set())
            if not packet_ids:
                continue
            frame_cache = _frame_direction_cache(frame, L)
            for packet_id in packet_ids:
                mask = sprint57._packet_mask(shell_map, cell_map, packet_id)
                if mask is None or not bool(np.any(mask)):
                    continue
                metrics[(run, _fmt(time), packet_id)] = _packet_direction_stats(
                    frame,
                    velocity[t_idx],
                    mask,
                    L,
                    frame_cache,
                )
        status_by_run[run] = "raw_red_direction_metrics_available"
    return metrics, status_by_run


def _label(row: dict[str, str], metric: dict[str, Any] | None, args: argparse.Namespace) -> str:
    if metric is None:
        return "RAW_RED_LOW_CONFIDENCE_ARTIFACT"
    if float(metric["direction_coherence_mean"]) < float(args.coherence_threshold):
        return "direction_incoherent"
    if float(metric["direction_lipschitz_proxy"]) > float(args.lipschitz_threshold):
        return "direction_lipschitz_high"
    return "direction_coherent"


def _build_packet_rows(
    selected: list[dict[str, str]],
    metrics: dict[tuple[str, str, str], dict[str, Any]],
    parent_rows: dict[tuple[str, str, str], dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    prev_dir: dict[tuple[str, str], np.ndarray | None] = {}
    dir_total: dict[tuple[str, str], float] = defaultdict(float)
    for rank, row in enumerate(sorted(selected, key=lambda r: (str(r["run"]), float(r["time"]), -float(r["weighted_A_raw_positive"]))), start=1):
        key = (str(row["run"]), _fmt(float(row["time"])), str(row["packet_id"]))
        metric = metrics.get(key)
        lineage_key = (str(row["run"]), str(row["packet_id"]))
        cur_dir = metric.get("direction") if metric else None
        inc = _direction_delta(prev_dir.get(lineage_key), cur_dir)
        dir_total[lineage_key] += inc
        if cur_dir is not None:
            prev_dir[lineage_key] = cur_dir
        parent = parent_rows.get((str(row["run"]), str(row["time"]), str(row["packet_id"])), {})
        label = _label(row, metric, args)
        out.append(
            {
                "run": row["run"],
                "time": row["time"],
                "dt": row["dt"],
                "K": row["K"],
                "packet_id": row["packet_id"],
                "A_raw_positive": row["A_raw_positive"],
                "A_raw_negative": row.get("A_raw_negative", "0"),
                "A_raw_net": row["A_raw_net"],
                "weighted_A_raw_positive": row["weighted_A_raw_positive"],
                "packet_enstrophy": row.get("packet_enstrophy", "0"),
                "packet_volume": row.get("packet_volume", "0"),
                "omega_magnitude_mean": _fmt(0.0 if metric is None else metric["omega_magnitude_mean"]),
                "omega_magnitude_max": _fmt(0.0 if metric is None else metric["omega_magnitude_max"]),
                "direction_coherence_mean": _fmt(0.0 if metric is None else metric["direction_coherence_mean"]),
                "direction_coherence_min": _fmt(0.0 if metric is None else metric["direction_coherence_min"]),
                "direction_lipschitz_proxy": _fmt(0.0 if metric is None else metric["direction_lipschitz_proxy"]),
                "direction_change_increment": _fmt(inc),
                "direction_change_integral": _fmt(dir_total[lineage_key]),
                "beltrami_defect_mean": _fmt(0.0 if metric is None else metric["beltrami_defect_mean"]),
                "beltrami_defect_normalized_mean": _fmt(0.0 if metric is None else metric["beltrami_defect_normalized_mean"]),
                "parent_confidence": parent.get("parent_confidence", "unavailable"),
                "parent_relation": parent.get("parent_relation", "unavailable"),
                "parent_state": parent.get("parent_state", "unavailable"),
                "child_state": parent.get("child_state", "unavailable"),
                "raw_action_trit": row.get("lagrangian_trit_after_integration", "unavailable"),
                "top_raw_red_rank": str(rank),
                "high_raw_red_selected": "true",
                "coherence_route_label": label,
            }
        )
    return out


def _build_by_k(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    stats: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    mins: dict[int, float] = defaultdict(lambda: 1.0)
    max_omega: dict[int, float] = defaultdict(float)
    for row in rows:
        k = int(row["K"])
        stats[k]["count"] += 1.0
        stats[k]["pos"] += float(row["A_raw_positive"])
        stats[k]["weighted"] += float(row["weighted_A_raw_positive"])
        stats[k]["coh"] += float(row["direction_coherence_mean"])
        stats[k]["lip"] += float(row["direction_lipschitz_proxy"])
        stats[k]["bel"] += float(row["beltrami_defect_normalized_mean"])
        incoherent = float(row["direction_coherence_mean"]) < float(args.coherence_threshold)
        stats[k]["incoherent"] += 1.0 if incoherent else 0.0
        mins[k] = min(mins[k], float(row["direction_coherence_min"]))
        max_omega[k] = max(max_omega[k], float(row["omega_magnitude_max"]))
    out = []
    for k in sorted(stats):
        count = max(stats[k]["count"], 1.0)
        out.append(
            {
                "K": str(k),
                "packet_count": str(int(stats[k]["count"])),
                "A_raw_positive_total": _fmt(stats[k]["pos"]),
                "weighted_A_raw_positive_total": _fmt(stats[k]["weighted"]),
                "direction_coherence_mean": _fmt(stats[k]["coh"] / count),
                "direction_coherence_min": _fmt(mins[k]),
                "direction_lipschitz_proxy_mean": _fmt(stats[k]["lip"] / count),
                "omega_magnitude_max": _fmt(max_omega[k]),
                "beltrami_defect_normalized_mean": _fmt(stats[k]["bel"] / count),
                "incoherent_packet_fraction": _fmt(stats[k]["incoherent"] / count),
            }
        )
    return out


def _route(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    if not rows:
        return "RAW_RED_DIRECTION_ANATOMY_INCONCLUSIVE"
    low_conf = sum(1 for row in rows if row["coherence_route_label"] == "RAW_RED_LOW_CONFIDENCE_ARTIFACT")
    if low_conf / max(len(rows), 1) > 0.5:
        return "RAW_RED_LOW_CONFIDENCE_ARTIFACT"
    incoherent = sum(1 for row in rows if str(row["coherence_route_label"]).startswith("direction_incoherent"))
    frac = incoherent / max(len(rows), 1)
    if frac > float(args.incoherent_fraction_threshold):
        return "RAW_RED_DIRECTION_INCOHERENT_CONCENTRATION_BLOCKED"
    return "RAW_RED_DIRECTION_COHERENT_CFM_ROUTE_ALIVE"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = _parse_args()
    raw_rows = _read_raw_rows(args.raw_action_csv)
    selected = _select_rows(raw_rows, args)
    truth_meta = sprint55._truth_meta_by_run(args.inputs, args.truth_root)
    for input_dir in args.inputs:
        run = input_dir.name
        summary = json.loads((input_dir / "ns_material_parent_summary.json").read_text(encoding="utf-8"))
        truth_meta[run]["packet_grid"] = int(summary.get("packet_grid") or 8)
    metrics, status_by_run = _load_metrics(selected, truth_meta)
    parent_rows = _read_parent_rows(args.inputs)
    packet_rows = _build_packet_rows(selected, metrics, parent_rows, args)
    by_k_rows = _build_by_k(packet_rows, args)
    route = _route(packet_rows, args)
    incoherent_count = sum(1 for row in packet_rows if str(row["coherence_route_label"]).startswith("direction_incoherent"))
    summary: dict[str, Any] = {
        "contract": "ns_sprint61_raw_red_direction_coherence_artifact",
        "diagnostic_mode": "sprint61_raw_red_direction_coherence_anatomy",
        "inputs": [str(path) for path in args.inputs],
        "raw_action_csv": str(args.raw_action_csv),
        "raw_action_row_count": len(raw_rows),
        "selected_high_raw_red_count": len(packet_rows),
        "top_fraction": float(args.top_fraction),
        "min_selected": int(args.min_selected),
        "max_selected": int(args.max_selected),
        "coherence_threshold": float(args.coherence_threshold),
        "incoherent_fraction_threshold": float(args.incoherent_fraction_threshold),
        "incoherent_packet_count": incoherent_count,
        "incoherent_packet_fraction": incoherent_count / max(len(packet_rows), 1),
        "direction_coherence_mean_selected": (
            sum(float(row["direction_coherence_mean"]) for row in packet_rows) / max(len(packet_rows), 1)
        ),
        "direction_lipschitz_proxy_mean_selected": (
            sum(float(row["direction_lipschitz_proxy"]) for row in packet_rows) / max(len(packet_rows), 1)
        ),
        "status_by_run": status_by_run,
        "raw_red_direction_coherence_proved": False,
        "cfm_direction_regularity_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT61_RAW_RED_DIRECTION_COHERENCE_DIAGNOSTIC",
        "route_decision": route,
        "boundary": (
            "Sprint 61 measures direction-coherence anatomy of high raw-red "
            "packets. It is a diagnostic CFM-style evidence surface only and "
            "does not prove direction regularity, stretch absorption, or "
            "Navier-Stokes regularity."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.out_dir / "ns_raw_red_direction_coherence.csv"
    by_k_path = args.out_dir / "ns_raw_red_direction_coherence_by_k.csv"
    summary_path = args.out_dir / "ns_sprint61_direction_coherence_summary.json"
    summary["ns_raw_red_direction_coherence_path"] = str(packet_path)
    summary["ns_raw_red_direction_coherence_by_k_path"] = str(by_k_path)
    _write_csv(packet_path, PACKET_FIELDS, packet_rows)
    _write_csv(by_k_path, BY_K_FIELDS, by_k_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint61_raw_red_direction_coherence_audit] wrote {packet_path}")
    print(f"[ns_sprint61_raw_red_direction_coherence_audit] wrote {by_k_path}")
    print(f"[ns_sprint61_raw_red_direction_coherence_audit] wrote {summary_path}")
    print(
        "[ns_sprint61_raw_red_direction_coherence_audit] "
        f"route={route} selected={len(packet_rows)} promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
