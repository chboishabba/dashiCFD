#!/usr/bin/env python3
"""Sprint 64 CKN-style local critical concentration audit.

Sprint 59-63 exhausted the tested DASHI NS source-budget route: normalized
packet action was non-additive, raw action stayed flat under shell
reassignment, high raw-red packets were direction-incoherent, and simple
cross-shell parent-budget contractivity failed. Sprint 64 switches the
diagnostic surface to a scale-critical local concentration object inspired by
CKN/ESS regularity criteria.

This producer computes a velocity-only local L3 concentration over
non-overlapping spatial cubes and trailing parabolic time windows:

    E_u(Q_r) = r^-2 * integral_{Q_r} |u|^3 dx dt

If a truth artifact also contains pressure snapshots, it adds the compatible
pressure term:

    E_p(Q_r) = r^-2 * integral_{Q_r} |p|^(3/2) dx dt

Current truth artifacts generally contain velocity_snapshots but no pressure
field, so pressure reconstruction is reported as missing. This is a diagnostic
route-alignment artifact only; it is not a CKN epsilon-regularity certificate
and carries no Clay/NS promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-30

ROW_FIELDS = [
    "run",
    "input_path",
    "N",
    "scale_cells",
    "r_physical",
    "time_index",
    "t_start_index",
    "t_end",
    "block_i",
    "block_j",
    "block_k",
    "local_L3_velocity",
    "local_pressure_3over2",
    "local_critical_quantity",
    "local_concentration_ratio",
    "overflow_state",
    "criticality_route_label",
    "pressure_available",
    "pressure_reconstruction_missing",
]

BY_SCALE_FIELDS = [
    "run",
    "N",
    "scale_cells",
    "row_count",
    "grounded_count",
    "plateau_count",
    "ascended_count",
    "grounded_fraction",
    "plateau_fraction",
    "ascended_fraction",
    "max_local_critical_quantity",
    "mean_local_critical_quantity",
    "max_local_concentration_ratio",
    "pressure_available",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="truth3d NPZ artifacts")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scales", type=int, nargs="+", default=[8, 16], help="non-overlapping cube side lengths in grid cells")
    p.add_argument("--epsilon-critical", type=float, default=0.01, help="diagnostic critical threshold")
    p.add_argument("--plateau-fraction", type=float, default=0.5, help="plateau begins at this fraction of epsilon")
    p.add_argument("--ascended-fraction-threshold", type=float, default=0.0, help="ascended fraction tolerated for subcritical route")
    p.add_argument("--max-blocks-per-scale", type=int, default=0, help="optional deterministic cap for quick diagnostics/tests")
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def _meta(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "meta_json" not in npz.files:
        return {}
    try:
        return json.loads(str(npz["meta_json"]))
    except Exception:
        return {}


def _time_axis(npz: np.lib.npyio.NpzFile, meta: dict[str, Any], frame_count: int) -> np.ndarray:
    dt = float(meta.get("dt", 1.0))
    steps = np.asarray(npz["steps"], dtype=np.float64) if "steps" in npz.files else np.arange(frame_count, dtype=np.float64)
    if steps.shape[0] != frame_count:
        steps = np.arange(frame_count, dtype=np.float64)
    return steps * dt


def _block_sum(arr: np.ndarray, scale: int, max_blocks: int) -> list[tuple[int, int, int, float]]:
    n = arr.shape[0]
    rows: list[tuple[int, int, int, float]] = []
    for i in range(0, n - scale + 1, scale):
        for j in range(0, n - scale + 1, scale):
            for k in range(0, n - scale + 1, scale):
                rows.append((i, j, k, float(arr[i : i + scale, j : j + scale, k : k + scale].sum(dtype=np.float64))))
                if max_blocks > 0 and len(rows) >= max_blocks:
                    return rows
    return rows


def _state(ratio: float, plateau_fraction: float) -> tuple[str, str]:
    if ratio >= 1.0:
        return "ascended", "CRITICAL_ASCENDED"
    if ratio >= plateau_fraction:
        return "plateau", "NEAR_CRITICAL_PLATEAU"
    return "grounded", "SUBCRITICAL_GROUNDED"


def _rows_for_input(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    npz = np.load(path)
    if "velocity_snapshots" not in npz.files:
        return [], {
            "input_path": str(path),
            "source_available": False,
            "reason": "velocity_snapshots_missing",
        }

    velocity = np.asarray(npz["velocity_snapshots"])
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise SystemExit(f"{path} velocity_snapshots must have shape (T,N,N,N,3)")
    pressure = np.asarray(npz["pressure_snapshots"]) if "pressure_snapshots" in npz.files else None
    if pressure is not None and pressure.shape[:4] != velocity.shape[:4]:
        raise SystemExit(f"{path} pressure_snapshots must have shape (T,N,N,N)")

    meta = _meta(npz)
    times = _time_axis(npz, meta, velocity.shape[0])
    domain_length = float(meta.get("domain_length", 2.0 * np.pi))
    n = int(velocity.shape[1])
    dx = domain_length / float(n)
    run = path.stem
    rows: list[dict[str, Any]] = []

    speed3 = np.sum(np.asarray(velocity, dtype=np.float64) ** 2, axis=-1) ** 1.5
    pressure32 = np.abs(np.asarray(pressure, dtype=np.float64)) ** 1.5 if pressure is not None else None

    for scale in sorted(set(int(s) for s in args.scales if int(s) > 0)):
        if scale > n:
            continue
        r = scale * dx
        for t_end_idx in range(speed3.shape[0]):
            if t_end_idx == 0:
                frame_dt = float(times[1] - times[0]) if len(times) > 1 else float(meta.get("dt", 1.0))
            else:
                frame_dt = float(times[t_end_idx] - times[t_end_idx - 1])
            frame_dt = max(frame_dt, EPS)
            parabolic_width = max(1, int(round((r * r) / frame_dt)))
            t_start_idx = max(0, t_end_idx - parabolic_width + 1)
            window_times = times[t_start_idx : t_end_idx + 1]
            if len(window_times) <= 1:
                time_weight = frame_dt
            else:
                time_weight = float((window_times[-1] - window_times[0]) / max(len(window_times) - 1, 1))
            v_window = speed3[t_start_idx : t_end_idx + 1].sum(axis=0, dtype=np.float64) * (dx**3) * time_weight
            p_window = (
                pressure32[t_start_idx : t_end_idx + 1].sum(axis=0, dtype=np.float64) * (dx**3) * time_weight
                if pressure32 is not None
                else None
            )
            p_blocks: dict[tuple[int, int, int], float] = {}
            if p_window is not None:
                p_blocks = {(i, j, k): val for i, j, k, val in _block_sum(p_window, scale, int(args.max_blocks_per_scale))}
            for i, j, k, v_sum in _block_sum(v_window, scale, int(args.max_blocks_per_scale)):
                p_sum = p_blocks.get((i, j, k), 0.0)
                quantity = (v_sum + p_sum) / (r * r + EPS)
                ratio = quantity / (float(args.epsilon_critical) + EPS)
                overflow, label = _state(ratio, float(args.plateau_fraction))
                rows.append(
                    {
                        "run": run,
                        "input_path": str(path),
                        "N": n,
                        "scale_cells": scale,
                        "r_physical": r,
                        "time_index": t_end_idx,
                        "t_start_index": t_start_idx,
                        "t_end": float(times[t_end_idx]) if len(times) else float(t_end_idx),
                        "block_i": i,
                        "block_j": j,
                        "block_k": k,
                        "local_L3_velocity": v_sum / (r * r + EPS),
                        "local_pressure_3over2": p_sum / (r * r + EPS),
                        "local_critical_quantity": quantity,
                        "local_concentration_ratio": ratio,
                        "overflow_state": overflow,
                        "criticality_route_label": label,
                        "pressure_available": pressure is not None,
                        "pressure_reconstruction_missing": pressure is None,
                    }
                )

    return rows, {
        "input_path": str(path),
        "source_available": True,
        "N": n,
        "frame_count": int(velocity.shape[0]),
        "pressure_available": pressure is not None,
        "domain_length": domain_length,
        "dx": dx,
    }


def _build_by_scale(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["run"]), int(row["N"]), int(row["scale_cells"]))].append(row)
    out: list[dict[str, Any]] = []
    for (run, n, scale), group in sorted(groups.items()):
        counts = defaultdict(int)
        values = []
        ratios = []
        pressure_available = False
        for row in group:
            counts[str(row["overflow_state"])] += 1
            values.append(float(row["local_critical_quantity"]))
            ratios.append(float(row["local_concentration_ratio"]))
            pressure_available = pressure_available or bool(row["pressure_available"])
        total = max(len(group), 1)
        out.append(
            {
                "run": run,
                "N": n,
                "scale_cells": scale,
                "row_count": len(group),
                "grounded_count": counts["grounded"],
                "plateau_count": counts["plateau"],
                "ascended_count": counts["ascended"],
                "grounded_fraction": counts["grounded"] / total,
                "plateau_fraction": counts["plateau"] / total,
                "ascended_fraction": counts["ascended"] / total,
                "max_local_critical_quantity": max(values) if values else 0.0,
                "mean_local_critical_quantity": float(np.mean(values)) if values else 0.0,
                "max_local_concentration_ratio": max(ratios) if ratios else 0.0,
                "pressure_available": pressure_available,
            }
        )
    return out


def _route(summary: dict[str, Any], args: argparse.Namespace) -> str:
    if not summary["source_available"]:
        return "LOCAL_CRITICAL_CONCENTRATION_SOURCE_UNAVAILABLE"
    if int(summary["row_count"]) == 0:
        return "LOCAL_CRITICAL_CONCENTRATION_NO_ROWS"
    if not summary["pressure_available_all"]:
        return "LOCAL_CRITICAL_CONCENTRATION_PRESSURE_RECONSTRUCTION_MISSING"
    asc = float(summary["ascended_fraction"])
    plateau = float(summary["plateau_fraction"])
    if asc <= float(args.ascended_fraction_threshold) and plateau == 0.0:
        return "LOCAL_CRITICAL_CONCENTRATION_SUBCRITICAL_ON_AVAILABLE_DATA"
    if asc < 1.0:
        return "LOCAL_CRITICAL_CONCENTRATION_MIXED"
    return "LOCAL_CRITICAL_CONCENTRATION_CRITICAL_BLOCKED"


def _build_summary(rows: list[dict[str, Any]], by_scale: list[dict[str, Any]], manifests: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    counts = defaultdict(int)
    values = []
    ratios = []
    for row in rows:
        counts[str(row["overflow_state"])] += 1
        values.append(float(row["local_critical_quantity"]))
        ratios.append(float(row["local_concentration_ratio"]))
    total = max(len(rows), 1)
    pressure_available_any = any(bool(m.get("pressure_available", False)) for m in manifests)
    pressure_available_all = bool(manifests) and all(bool(m.get("pressure_available", False)) for m in manifests if m.get("source_available", False))
    summary: dict[str, Any] = {
        "contract": "ns_sprint64_local_critical_concentration_artifact",
        "diagnostic_mode": "sprint64_ckn_velocity_local_l3_concentration",
        "source_available": any(bool(m.get("source_available", False)) for m in manifests),
        "input_count": len(manifests),
        "row_count": len(rows),
        "by_scale_row_count": len(by_scale),
        "grounded_count": counts["grounded"],
        "plateau_count": counts["plateau"],
        "ascended_count": counts["ascended"],
        "grounded_fraction": counts["grounded"] / total,
        "plateau_fraction": counts["plateau"] / total,
        "ascended_fraction": counts["ascended"] / total,
        "max_local_critical_quantity": max(values) if values else 0.0,
        "mean_local_critical_quantity": float(np.mean(values)) if values else 0.0,
        "max_local_concentration_ratio": max(ratios) if ratios else 0.0,
        "epsilon_critical": float(args.epsilon_critical),
        "plateau_fraction_threshold": float(args.plateau_fraction),
        "pressure_available_any": pressure_available_any,
        "pressure_available_all": pressure_available_all,
        "pressure_reconstruction_missing": not pressure_available_all,
        "local_critical_concentration_proved": False,
        "ckn_epsilon_regularity_applied": False,
        "pressure_reconstruction_proved": False,
        "physical_bridge_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT64_CKN_DIAGNOSTIC",
        "formal_target": "NSCKNCriticalNormRoute",
        "boundary": (
            "Sprint 64 computes a velocity-only local scale-critical concentration "
            "surface from available truth artifacts. Without pressure reconstruction "
            "and a theorem-grade epsilon bridge, it is not a CKN certificate and "
            "does not promote Clay/NS."
        ),
        "input_manifest_summaries": manifests,
    }
    summary["route_decision"] = _route(summary, args)
    return summary


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fieldnames})


def main() -> None:
    args = _parse_args()
    all_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in args.inputs:
        rows, manifest = _rows_for_input(path, args)
        all_rows.extend(rows)
        manifests.append(manifest)
    by_scale = _build_by_scale(all_rows)
    summary = _build_summary(all_rows, by_scale, manifests, args)
    summary["inputs"] = [str(path) for path in args.inputs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    row_path = args.out_dir / "ns_local_critical_concentration.csv"
    by_scale_path = args.out_dir / "ns_local_critical_concentration_by_scale.csv"
    summary_path = args.out_dir / "ns_sprint64_local_critical_concentration_summary.json"
    summary["ns_local_critical_concentration_path"] = str(row_path)
    summary["ns_local_critical_concentration_by_scale_path"] = str(by_scale_path)

    _write_csv(row_path, ROW_FIELDS, all_rows)
    _write_csv(by_scale_path, BY_SCALE_FIELDS, by_scale)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint64_ckn_local_critical_concentration_audit] wrote {row_path}")
    print(f"[ns_sprint64_ckn_local_critical_concentration_audit] wrote {by_scale_path}")
    print(f"[ns_sprint64_ckn_local_critical_concentration_audit] wrote {summary_path}")
    print(
        "[ns_sprint64_ckn_local_critical_concentration_audit] "
        f"route={summary['route_decision']} "
        f"rows={summary['row_count']} "
        f"ascended_fraction={summary['ascended_fraction']} "
        f"pressure_missing={summary['pressure_reconstruction_missing']} "
        f"promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
