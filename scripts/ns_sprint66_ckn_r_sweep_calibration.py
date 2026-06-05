#!/usr/bin/env python3
"""Sprint 66 CKN r-sweep calibration audit.

Sprint 64/65 made the pressure-inclusive local critical concentration
diagnostic measurable, but a single fixed epsilon over broad blocks is a blunt
proxy.  Sprint 66 samples candidate hot spots and computes the scale-normalized
CKN-style quantity

    C(r) = r^-2 * integral_Q (|u|^3 + |p|^(3/2)) dx dt

over several radii.  The useful diagnostic is the r-sweep trend: decay under
zoom is bulk turbulence / CKN-safe evidence for the sampled centres, while
growth under zoom is a concentration candidate.

This remains a DNS/proxy calibration artifact only.  It does not apply a CKN
epsilon-regularity theorem, prove a suitable weak solution bridge, prove
continuum uniformity, or promote Clay/Navier-Stokes.
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


EPS = 1e-30

ROW_FIELDS = [
    "run",
    "input_path",
    "N",
    "hotspot_id",
    "hotspot_rank",
    "candidate_source",
    "time_index",
    "t_start_index",
    "t_end",
    "center_i",
    "center_j",
    "center_k",
    "seed_score",
    "epsilon_critical",
    "r_cells",
    "r_physical",
    "sample_cell_count",
    "local_L3_velocity",
    "local_pressure_3over2",
    "C_total",
    "C_epsilon_ratio",
    "overflow_state",
    "criticality_route_label",
    "pressure_available",
    "pressure_reconstruction_missing",
]

BY_R_FIELDS = [
    "run",
    "N",
    "epsilon_critical",
    "r_cells",
    "r_physical",
    "hotspot_count",
    "valid_count",
    "grounded_count",
    "plateau_count",
    "ascended_count",
    "grounded_fraction",
    "plateau_fraction",
    "ascended_fraction",
    "max_C_total",
    "mean_C_total",
    "median_C_total",
    "max_C_epsilon_ratio",
    "pressure_available",
]

HOTSPOT_FIELDS = [
    "run",
    "input_path",
    "N",
    "hotspot_id",
    "hotspot_rank",
    "candidate_source",
    "time_index",
    "center_i",
    "center_j",
    "center_k",
    "seed_score",
    "min_r_cells",
    "max_r_cells",
    "min_C_total",
    "max_C_total",
    "inner_C_total",
    "outer_C_total",
    "log_slope_dlogC_dlogr",
    "trend_label",
    "pressure_contribution_fraction_at_max",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="pressure-present truth3d NPZ artifacts")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--candidate-csv", type=Path, default=None, help="optional Sprint 59 ns_raw_packet_stretch_action.csv")
    p.add_argument("--top-hotspots", type=int, default=10)
    p.add_argument("--r-cells", type=int, nargs="+", default=[2, 4, 8, 16], help="cube side lengths in grid cells")
    p.add_argument("--epsilon-grid", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.5, 1.0])
    p.add_argument("--plateau-fraction", type=float, default=0.5)
    p.add_argument("--packet-grid", type=int, default=8, help="Sprint 49 packet grid used for candidate packet centres")
    p.add_argument("--trend-tolerance", type=float, default=0.15, help="absolute log-slope tolerance for flat trend labels")
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


def _state(ratio: float, plateau_fraction: float) -> tuple[str, str]:
    if ratio >= 1.0:
        return "ascended", "CRITICAL_ASCENDED"
    if ratio >= plateau_fraction:
        return "plateau", "NEAR_CRITICAL_PLATEAU"
    return "grounded", "SUBCRITICAL_GROUNDED"


def _run_key_from_stem(stem: str) -> str:
    clean = stem
    clean = re.sub(r"^sprint49_material_parent_", "", clean)
    clean = re.sub(r"^ns3d_", "", clean)
    clean = re.sub(r"_pressure$", "", clean)
    return clean


def _cell_center(cell_id: int, n: int, packet_grid: int) -> tuple[int, int, int]:
    pg = max(1, int(packet_grid))
    stride = max(1, int(n / pg))
    ci = int(cell_id) // (pg * pg)
    rem = int(cell_id) % (pg * pg)
    cj = rem // pg
    ck = rem % pg
    return (
        min(n - 1, ci * stride + stride // 2),
        min(n - 1, cj * stride + stride // 2),
        min(n - 1, ck * stride + stride // 2),
    )


def _packet_candidates(candidate_csv: Path | None, n_by_key: dict[str, int], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    if candidate_csv is None or not candidate_csv.exists():
        return {}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with candidate_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_key = _run_key_from_stem(str(row.get("run", "")))
            if run_key not in n_by_key:
                continue
            packet_id = str(row.get("packet_id", ""))
            match = re.fullmatch(r"K-?\d+_cell(\d+)", packet_id)
            if match is None:
                continue
            try:
                time = float(row.get("time", "0"))
                score = float(row.get("A_raw_positive", "0"))
                cell = int(match.group(1))
            except ValueError:
                continue
            ci, cj, ck = _cell_center(cell, n_by_key[run_key], int(args.packet_grid))
            out[run_key].append(
                {
                    "candidate_source": "sprint59_raw_packet",
                    "time_value": time,
                    "center_i": ci,
                    "center_j": cj,
                    "center_k": ck,
                    "seed_score": score,
                }
            )
    for key, rows in out.items():
        rows.sort(key=lambda r: float(r["seed_score"]), reverse=True)
        out[key] = rows[: max(0, int(args.top_hotspots))]
    return out


def _field_candidates(
    run_key: str,
    velocity: np.ndarray,
    pressure: np.ndarray,
    times: np.ndarray,
    top_hotspots: int,
) -> list[dict[str, Any]]:
    density = np.sum(np.asarray(velocity, dtype=np.float64) ** 2, axis=-1) ** 1.5
    density += np.abs(np.asarray(pressure, dtype=np.float64)) ** 1.5
    flat = density.reshape(-1)
    count = min(max(0, int(top_hotspots)), flat.size)
    if count == 0:
        return []
    # Stable top-k with deterministic final ordering.
    idx = np.argpartition(flat, -count)[-count:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    t_count, n, _, _ = density.shape
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for raw in idx:
        t, i, j, k = np.unravel_index(int(raw), (t_count, n, n, n))
        key = (int(t), int(i), int(j), int(k))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "candidate_source": "pointwise_ckn_density",
                "time_value": float(times[t]) if len(times) else float(t),
                "time_index": int(t),
                "center_i": int(i),
                "center_j": int(j),
                "center_k": int(k),
                "seed_score": float(flat[raw]),
            }
        )
    return out


def _nearest_time_index(times: np.ndarray, value: float) -> int:
    if len(times) == 0:
        return 0
    return int(np.argmin(np.abs(times - float(value))))


def _centered_cube_sum(arr: np.ndarray, center: tuple[int, int, int], side: int) -> tuple[float, int]:
    n = int(arr.shape[0])
    side = max(1, min(int(side), n))
    start = [int(c) - side // 2 for c in center]
    axes = [np.mod(np.arange(s, s + side), n) for s in start]
    sub = arr[np.ix_(axes[0], axes[1], axes[2])]
    return float(np.sum(sub, dtype=np.float64)), int(sub.size)


def _rows_for_input(path: Path, seeded: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], [], {"input_path": str(path), "source_available": False, "reason": "input_missing"}
    z = np.load(path)
    if "velocity_snapshots" not in z.files:
        return [], [], {"input_path": str(path), "source_available": False, "reason": "velocity_snapshots_missing"}
    velocity = np.asarray(z["velocity_snapshots"])
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise SystemExit(f"{path} velocity_snapshots must have shape (T,N,N,N,3)")
    pressure_available = "pressure_snapshots" in z.files
    if not pressure_available:
        return [], [], {
            "input_path": str(path),
            "source_available": True,
            "pressure_available": False,
            "N": int(velocity.shape[1]),
            "frame_count": int(velocity.shape[0]),
        }
    pressure = np.asarray(z["pressure_snapshots"])
    if pressure.shape[:4] != velocity.shape[:4]:
        raise SystemExit(f"{path} pressure_snapshots must have shape (T,N,N,N)")

    meta = _meta(z)
    times = _time_axis(z, meta, velocity.shape[0])
    n = int(velocity.shape[1])
    domain_length = float(meta.get("domain_length", 2.0 * np.pi))
    dx = domain_length / float(n)
    run = path.stem
    run_key = _run_key_from_stem(run)

    candidates = [dict(c) for c in seeded.get(run_key, [])]
    if not candidates:
        candidates = _field_candidates(run_key, velocity, pressure, times, int(args.top_hotspots))
    for c in candidates:
        c["time_index"] = int(c.get("time_index", _nearest_time_index(times, float(c.get("time_value", 0.0)))))

    speed3 = np.sum(np.asarray(velocity, dtype=np.float64) ** 2, axis=-1) ** 1.5
    pressure32 = np.abs(np.asarray(pressure, dtype=np.float64)) ** 1.5
    r_values = sorted(set(int(r) for r in args.r_cells if int(r) > 0 and int(r) <= n))
    eps_values = sorted(set(float(e) for e in args.epsilon_grid if float(e) > 0.0))

    rows: list[dict[str, Any]] = []
    hotspot_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[: max(0, int(args.top_hotspots))], start=1):
        t_end_idx = max(0, min(int(candidate["time_index"]), speed3.shape[0] - 1))
        center = (int(candidate["center_i"]) % n, int(candidate["center_j"]) % n, int(candidate["center_k"]) % n)
        values_by_r: dict[int, tuple[float, float, float]] = {}
        pressure_fraction_at_max = 0.0
        for r_cells in r_values:
            r_phys = float(r_cells) * dx
            if t_end_idx == 0:
                frame_dt = float(times[1] - times[0]) if len(times) > 1 else float(meta.get("dt", 1.0))
            else:
                frame_dt = float(times[t_end_idx] - times[t_end_idx - 1])
            frame_dt = max(frame_dt, EPS)
            parabolic_width = max(1, int(round((r_phys * r_phys) / frame_dt)))
            t_start_idx = max(0, t_end_idx - parabolic_width + 1)
            window_times = times[t_start_idx : t_end_idx + 1]
            if len(window_times) <= 1:
                time_weight = frame_dt
            else:
                time_weight = float((window_times[-1] - window_times[0]) / max(len(window_times) - 1, 1))
            v_window = speed3[t_start_idx : t_end_idx + 1].sum(axis=0, dtype=np.float64) * (dx**3) * time_weight
            p_window = pressure32[t_start_idx : t_end_idx + 1].sum(axis=0, dtype=np.float64) * (dx**3) * time_weight
            v_sum, sample_count = _centered_cube_sum(v_window, center, r_cells)
            p_sum, _ = _centered_cube_sum(p_window, center, r_cells)
            c_velocity = v_sum / (r_phys * r_phys + EPS)
            c_pressure = p_sum / (r_phys * r_phys + EPS)
            c_total = c_velocity + c_pressure
            values_by_r[int(r_cells)] = (float(r_phys), float(c_total), float(c_pressure / (c_total + EPS)))
            for epsilon in eps_values:
                ratio = c_total / (epsilon + EPS)
                overflow, label = _state(ratio, float(args.plateau_fraction))
                rows.append(
                    {
                        "run": run,
                        "input_path": str(path),
                        "N": n,
                        "hotspot_id": f"{run_key}_hotspot{rank}",
                        "hotspot_rank": rank,
                        "candidate_source": str(candidate["candidate_source"]),
                        "time_index": t_end_idx,
                        "t_start_index": t_start_idx,
                        "t_end": float(times[t_end_idx]) if len(times) else float(t_end_idx),
                        "center_i": center[0],
                        "center_j": center[1],
                        "center_k": center[2],
                        "seed_score": float(candidate["seed_score"]),
                        "epsilon_critical": epsilon,
                        "r_cells": int(r_cells),
                        "r_physical": r_phys,
                        "sample_cell_count": sample_count,
                        "local_L3_velocity": c_velocity,
                        "local_pressure_3over2": c_pressure,
                        "C_total": c_total,
                        "C_epsilon_ratio": ratio,
                        "overflow_state": overflow,
                        "criticality_route_label": label,
                        "pressure_available": True,
                        "pressure_reconstruction_missing": False,
                    }
                )
        if values_by_r:
            rs = np.asarray([values_by_r[k][0] for k in sorted(values_by_r)], dtype=np.float64)
            cs = np.asarray([values_by_r[k][1] for k in sorted(values_by_r)], dtype=np.float64)
            if len(rs) >= 2 and np.all(cs > 0.0):
                slope = float(np.polyfit(np.log(rs + EPS), np.log(cs + EPS), 1)[0])
            else:
                slope = 0.0
            if slope > float(args.trend_tolerance):
                trend = "CKN_DECAYS_UNDER_ZOOM"
            elif slope < -float(args.trend_tolerance):
                trend = "CKN_CONCENTRATES_UNDER_ZOOM"
            else:
                trend = "CKN_FLAT_MARGINAL"
            max_idx = int(np.argmax(cs))
            pressure_fraction_at_max = float([values_by_r[k][2] for k in sorted(values_by_r)][max_idx])
            hotspot_rows.append(
                {
                    "run": run,
                    "input_path": str(path),
                    "N": n,
                    "hotspot_id": f"{run_key}_hotspot{rank}",
                    "hotspot_rank": rank,
                    "candidate_source": str(candidate["candidate_source"]),
                    "time_index": t_end_idx,
                    "center_i": center[0],
                    "center_j": center[1],
                    "center_k": center[2],
                    "seed_score": float(candidate["seed_score"]),
                    "min_r_cells": min(values_by_r),
                    "max_r_cells": max(values_by_r),
                    "min_C_total": float(np.min(cs)),
                    "max_C_total": float(np.max(cs)),
                    "inner_C_total": float(cs[0]),
                    "outer_C_total": float(cs[-1]),
                    "log_slope_dlogC_dlogr": slope,
                    "trend_label": trend,
                    "pressure_contribution_fraction_at_max": pressure_fraction_at_max,
                }
            )

    manifest = {
        "input_path": str(path),
        "source_available": True,
        "pressure_available": True,
        "run": run,
        "run_key": run_key,
        "N": n,
        "frame_count": int(velocity.shape[0]),
        "hotspot_count": len(hotspot_rows),
        "row_count": len(rows),
        "domain_length": domain_length,
        "dx": dx,
    }
    return rows, hotspot_rows, manifest


def _build_by_r(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["run"]), int(row["N"]), float(row["epsilon_critical"]), int(row["r_cells"]))].append(row)
    out: list[dict[str, Any]] = []
    for (run, n, epsilon, r_cells), group in sorted(groups.items()):
        counts = defaultdict(int)
        values = []
        ratios = []
        pressure_available = False
        for row in group:
            counts[str(row["overflow_state"])] += 1
            values.append(float(row["C_total"]))
            ratios.append(float(row["C_epsilon_ratio"]))
            pressure_available = pressure_available or bool(row["pressure_available"])
        total = max(len(group), 1)
        out.append(
            {
                "run": run,
                "N": n,
                "epsilon_critical": epsilon,
                "r_cells": r_cells,
                "r_physical": float(group[0]["r_physical"]) if group else 0.0,
                "hotspot_count": len({str(row["hotspot_id"]) for row in group}),
                "valid_count": len(group),
                "grounded_count": counts["grounded"],
                "plateau_count": counts["plateau"],
                "ascended_count": counts["ascended"],
                "grounded_fraction": counts["grounded"] / total,
                "plateau_fraction": counts["plateau"] / total,
                "ascended_fraction": counts["ascended"] / total,
                "max_C_total": max(values) if values else 0.0,
                "mean_C_total": float(np.mean(values)) if values else 0.0,
                "median_C_total": float(np.median(values)) if values else 0.0,
                "max_C_epsilon_ratio": max(ratios) if ratios else 0.0,
                "pressure_available": pressure_available,
            }
        )
    return out


def _route(summary: dict[str, Any]) -> str:
    if not summary["source_available"]:
        return "CKN_R_SWEEP_SOURCE_UNAVAILABLE"
    if not summary["pressure_available_all"]:
        return "CKN_R_SWEEP_PRESSURE_RECONSTRUCTION_MISSING"
    if int(summary["row_count"]) == 0:
        return "CKN_R_SWEEP_NO_HOTSPOTS"
    if int(summary["ascended_count"]) == 0 and int(summary["plateau_count"]) == 0:
        return "CKN_R_SWEEP_SUBCRITICAL_ON_SAMPLED_HOTSPOTS"
    if int(summary["concentrating_hotspot_count"]) == 0 and int(summary["decaying_hotspot_count"]) > 0:
        return "CKN_R_SWEEP_DECAYS_UNDER_ZOOM"
    if int(summary["ascended_count"]) == int(summary["row_count"]) and int(summary["concentrating_hotspot_count"]) > 0:
        return "CKN_R_SWEEP_CRITICAL_BLOCKED"
    return "CKN_R_SWEEP_MIXED"


def _build_summary(
    rows: list[dict[str, Any]],
    by_r: list[dict[str, Any]],
    hotspots: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = defaultdict(int)
    for row in rows:
        counts[str(row["overflow_state"])] += 1
    trend_counts = defaultdict(int)
    for row in hotspots:
        trend_counts[str(row["trend_label"])] += 1
    total = max(len(rows), 1)
    pressure_available_all = bool(manifests) and all(bool(m.get("pressure_available", False)) for m in manifests if m.get("source_available", False))
    summary: dict[str, Any] = {
        "contract": "ns_sprint66_ckn_r_sweep_calibration_artifact",
        "diagnostic_mode": "candidate_centered_pressure_inclusive_ckn_r_sweep",
        "source_available": any(bool(m.get("source_available", False)) for m in manifests),
        "input_count": len(manifests),
        "row_count": len(rows),
        "by_r_row_count": len(by_r),
        "hotspot_count": len(hotspots),
        "grounded_count": counts["grounded"],
        "plateau_count": counts["plateau"],
        "ascended_count": counts["ascended"],
        "grounded_fraction": counts["grounded"] / total,
        "plateau_fraction": counts["plateau"] / total,
        "ascended_fraction": counts["ascended"] / total,
        "decaying_hotspot_count": trend_counts["CKN_DECAYS_UNDER_ZOOM"],
        "flat_hotspot_count": trend_counts["CKN_FLAT_MARGINAL"],
        "concentrating_hotspot_count": trend_counts["CKN_CONCENTRATES_UNDER_ZOOM"],
        "epsilon_grid": [float(e) for e in args.epsilon_grid],
        "r_cells": [int(r) for r in args.r_cells],
        "plateau_fraction_threshold": float(args.plateau_fraction),
        "trend_tolerance": float(args.trend_tolerance),
        "pressure_available_all": pressure_available_all,
        "pressure_reconstruction_missing": not pressure_available_all,
        "scale_normalization": "r^-2_parabolic_cylinder_cube_proxy",
        "candidate_policy": "Sprint59 packet candidates if provided, otherwise top pointwise |u|^3+|p|^(3/2) density",
        "local_critical_concentration_proved": False,
        "ckn_epsilon_regularity_applied": False,
        "pressure_reconstruction_proved": False,
        "physical_bridge_proved": False,
        "suitable_weak_solution_bridge_proved": False,
        "continuum_uniformity_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT66_CKN_R_SWEEP_CALIBRATION_DIAGNOSTIC",
        "input_manifest_summaries": manifests,
    }
    summary["route_decision"] = _route(summary)
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
    n_by_key: dict[str, int] = {}
    for path in args.inputs:
        if not path.exists():
            continue
        z = np.load(path)
        if "velocity_snapshots" in z.files:
            n_by_key[_run_key_from_stem(path.stem)] = int(np.asarray(z["velocity_snapshots"]).shape[1])
    seeded = _packet_candidates(args.candidate_csv, n_by_key, args)

    rows: list[dict[str, Any]] = []
    hotspot_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in args.inputs:
        r, h, m = _rows_for_input(path, seeded, args)
        rows.extend(r)
        hotspot_rows.extend(h)
        manifests.append(m)
    by_r = _build_by_r(rows)
    summary = _build_summary(rows, by_r, hotspot_rows, manifests, args)
    summary["inputs"] = [str(path) for path in args.inputs]
    summary["candidate_csv"] = str(args.candidate_csv) if args.candidate_csv is not None else ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    row_path = args.out_dir / "ns_ckn_r_sweep_calibration.csv"
    by_r_path = args.out_dir / "ns_ckn_r_sweep_by_radius.csv"
    hotspot_path = args.out_dir / "ns_ckn_r_sweep_hotspots.csv"
    summary_path = args.out_dir / "ns_sprint66_ckn_r_sweep_calibration_summary.json"
    summary["ns_ckn_r_sweep_calibration_path"] = str(row_path)
    summary["ns_ckn_r_sweep_by_radius_path"] = str(by_r_path)
    summary["ns_ckn_r_sweep_hotspots_path"] = str(hotspot_path)

    _write_csv(row_path, ROW_FIELDS, rows)
    _write_csv(by_r_path, BY_R_FIELDS, by_r)
    _write_csv(hotspot_path, HOTSPOT_FIELDS, hotspot_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint66_ckn_r_sweep_calibration] wrote {row_path}")
    print(f"[ns_sprint66_ckn_r_sweep_calibration] wrote {by_r_path}")
    print(f"[ns_sprint66_ckn_r_sweep_calibration] wrote {hotspot_path}")
    print(f"[ns_sprint66_ckn_r_sweep_calibration] wrote {summary_path}")
    print(
        "[ns_sprint66_ckn_r_sweep_calibration] "
        f"route={summary['route_decision']} rows={summary['row_count']} "
        f"hotspots={summary['hotspot_count']} ascended_fraction={summary['ascended_fraction']} "
        f"decaying={summary['decaying_hotspot_count']} concentrating={summary['concentrating_hotspot_count']} "
        f"promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
