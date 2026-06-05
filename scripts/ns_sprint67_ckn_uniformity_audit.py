#!/usr/bin/env python3
"""Sprint 67B CKN uniformity audit.

Sprint 66 sampled a small set of hot spots and found pressure-inclusive
CKN-style concentration decays under zoom.  Sprint 67B removes the strongest
selection bias by replaying fixed-block Sprint 64 ascended cylinders, sweeping
the scale-normalized quantity

    C(r) = r^-2 * integral_Q (|u|^3 + |p|^(3/2)) dx dt

around each selected cylinder centre, and clustering the fixed-block ascended
population.  This is still a DNS/proxy audit only.  It does not apply CKN
epsilon regularity, prove suitable weak-solution status, prove continuum
uniformity, or promote Clay/Navier-Stokes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-30

CYLINDER_FIELDS = [
    "run",
    "input_path",
    "N",
    "cylinder_id",
    "candidate_source",
    "source_scale_cells",
    "source_time_index",
    "source_block_i",
    "source_block_j",
    "source_block_k",
    "center_i",
    "center_j",
    "center_k",
    "seed_score",
    "r_cells_values",
    "min_C_total",
    "max_C_total",
    "inner_C_total",
    "outer_C_total",
    "log_slope_dlogC_dlogr",
    "trend_label",
    "pressure_fraction_at_max",
    "max_C_epsilon_ratio",
    "max_overflow_state",
    "route_label",
]

CLUSTER_FIELDS = [
    "run",
    "N",
    "cluster_id",
    "source_scale_cells",
    "source_time_index",
    "candidate_count",
    "decaying_count",
    "flat_count",
    "concentrating_count",
    "max_C_total",
    "median_C_total",
    "mean_slope",
    "pressure_fraction_max",
    "cluster_lifetime_frames",
    "route_label",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="pressure-present truth3d NPZ artifacts")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sprint64-csv", type=Path, default=None, help="optional ns_local_critical_concentration.csv")
    p.add_argument("--epsilon-critical", type=float, default=0.01)
    p.add_argument("--plateau-fraction", type=float, default=0.5)
    p.add_argument("--r-cells", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--trend-tolerance", type=float, default=0.15)
    p.add_argument("--max-candidates-per-run", type=int, default=5000)
    p.add_argument("--fallback-top-candidates", type=int, default=128)
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def _run_key_from_stem(stem: str) -> str:
    clean = stem
    clean = re.sub(r"^sprint49_material_parent_", "", clean)
    clean = re.sub(r"^ns3d_", "", clean)
    clean = re.sub(r"_pressure$", "", clean)
    return clean


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


def _state(ratio: float, plateau_fraction: float) -> str:
    if ratio >= 1.0:
        return "ascended"
    if ratio >= plateau_fraction:
        return "plateau"
    return "grounded"


def _trend(slope: float, tolerance: float) -> str:
    if slope > tolerance:
        return "CKN_DECAYS_UNDER_ZOOM"
    if slope < -tolerance:
        return "CKN_CONCENTRATION_CANDIDATE"
    return "CKN_PLATEAU"


def _route_from_trend(trend: str, pressure_fraction: float) -> str:
    if pressure_fraction > 0.9:
        return "CKN_PRESSURE_DOMINATED_ARTIFACT"
    if trend == "CKN_DECAYS_UNDER_ZOOM":
        return "CKN_UNIFORM_DECAY_SUPPORTED"
    if trend == "CKN_CONCENTRATION_CANDIDATE":
        return "CKN_CONCENTRATION_CANDIDATE_FOUND"
    return "CKN_LOCALIZED_PERSISTENT_PLATEAU"


def _centered_cube_sum(arr: np.ndarray, center: tuple[int, int, int], side: int) -> tuple[float, int]:
    n = int(arr.shape[0])
    side = max(1, min(int(side), n))
    start = [int(c) - side // 2 for c in center]
    axes = [np.mod(np.arange(s, s + side), n) for s in start]
    sub = arr[np.ix_(axes[0], axes[1], axes[2])]
    return float(np.sum(sub, dtype=np.float64)), int(sub.size)


def _load_sprint64_candidates(path: Path | None, run_keys: set[str], max_per_run: int) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("overflow_state", "")) != "ascended":
                continue
            run_key = _run_key_from_stem(str(row.get("run", "")))
            if run_key not in run_keys:
                continue
            try:
                scale = int(row["scale_cells"])
                time_idx = int(row["time_index"])
                block_i = int(row["block_i"])
                block_j = int(row["block_j"])
                block_k = int(row["block_k"])
                score = float(row.get("local_critical_quantity", row.get("local_concentration_ratio", "0")))
            except (KeyError, ValueError):
                continue
            groups[run_key].append(
                {
                    "candidate_source": "sprint64_fixed_block_ascended",
                    "source_scale_cells": scale,
                    "source_time_index": time_idx,
                    "source_block_i": block_i,
                    "source_block_j": block_j,
                    "source_block_k": block_k,
                    "center_i": block_i + scale // 2,
                    "center_j": block_j + scale // 2,
                    "center_k": block_k + scale // 2,
                    "seed_score": score,
                }
            )
    for key, rows in groups.items():
        rows.sort(key=lambda r: float(r["seed_score"]), reverse=True)
        groups[key] = rows[: max(0, int(max_per_run))]
    return groups


def _fallback_candidates(
    velocity: np.ndarray,
    pressure: np.ndarray,
    count: int,
) -> list[dict[str, Any]]:
    density = np.sum(np.asarray(velocity, dtype=np.float64) ** 2, axis=-1) ** 1.5
    density += np.abs(np.asarray(pressure, dtype=np.float64)) ** 1.5
    flat = density.reshape(-1)
    take = min(max(0, int(count)), flat.size)
    if take == 0:
        return []
    idx = np.argpartition(flat, -take)[-take:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    t_count, n, _, _ = density.shape
    out: list[dict[str, Any]] = []
    for rank, raw in enumerate(idx, start=1):
        t, i, j, k = np.unravel_index(int(raw), (t_count, n, n, n))
        out.append(
            {
                "candidate_source": "fallback_pointwise_ckn_density",
                "source_scale_cells": 1,
                "source_time_index": int(t),
                "source_block_i": int(i),
                "source_block_j": int(j),
                "source_block_k": int(k),
                "center_i": int(i),
                "center_j": int(j),
                "center_k": int(k),
                "seed_score": float(flat[raw]),
            }
        )
    return out


def _sweep_candidate(
    path: Path,
    run_key: str,
    n: int,
    times: np.ndarray,
    meta: dict[str, Any],
    dx: float,
    speed3: np.ndarray,
    pressure32: np.ndarray,
    candidate: dict[str, Any],
    args: argparse.Namespace,
    ordinal: int,
) -> dict[str, Any] | None:
    t_end_idx = max(0, min(int(candidate["source_time_index"]), speed3.shape[0] - 1))
    center = (
        int(candidate["center_i"]) % n,
        int(candidate["center_j"]) % n,
        int(candidate["center_k"]) % n,
    )
    values: list[tuple[int, float, float, float, float]] = []
    for r_cells in sorted(set(int(r) for r in args.r_cells if 0 < int(r) <= n)):
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
        v_sum, _ = _centered_cube_sum(v_window, center, r_cells)
        p_sum, _ = _centered_cube_sum(p_window, center, r_cells)
        c_vel = v_sum / (r_phys * r_phys + EPS)
        c_prs = p_sum / (r_phys * r_phys + EPS)
        c_total = c_vel + c_prs
        values.append((int(r_cells), float(r_phys), float(c_total), float(c_prs), float(c_prs / (c_total + EPS))))
    if not values:
        return None
    rs = np.asarray([v[1] for v in values], dtype=np.float64)
    cs = np.asarray([v[2] for v in values], dtype=np.float64)
    if len(values) >= 2 and np.all(cs > 0.0):
        slope = float(np.polyfit(np.log(rs + EPS), np.log(cs + EPS), 1)[0])
    else:
        slope = 0.0
    trend = _trend(slope, float(args.trend_tolerance))
    max_idx = int(np.argmax(cs))
    pressure_fraction = float(values[max_idx][4])
    max_c = float(np.max(cs))
    max_ratio = max_c / (float(args.epsilon_critical) + EPS)
    overflow = _state(max_ratio, float(args.plateau_fraction))
    return {
        "run": path.stem,
        "input_path": str(path),
        "N": n,
        "cylinder_id": f"{run_key}_cyl{ordinal}",
        "candidate_source": str(candidate["candidate_source"]),
        "source_scale_cells": int(candidate["source_scale_cells"]),
        "source_time_index": t_end_idx,
        "source_block_i": int(candidate["source_block_i"]),
        "source_block_j": int(candidate["source_block_j"]),
        "source_block_k": int(candidate["source_block_k"]),
        "center_i": center[0],
        "center_j": center[1],
        "center_k": center[2],
        "seed_score": float(candidate["seed_score"]),
        "r_cells_values": ";".join(str(v[0]) for v in values),
        "min_C_total": float(np.min(cs)),
        "max_C_total": max_c,
        "inner_C_total": float(cs[0]),
        "outer_C_total": float(cs[-1]),
        "log_slope_dlogC_dlogr": slope,
        "trend_label": trend,
        "pressure_fraction_at_max": pressure_fraction,
        "max_C_epsilon_ratio": max_ratio,
        "max_overflow_state": overflow,
        "route_label": _route_from_trend(trend, pressure_fraction),
    }


def _cluster_candidates(candidates: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_key[(int(c["source_scale_cells"]), int(c["source_time_index"]))].append(c)
    clusters: list[list[dict[str, Any]]] = []
    for (scale, _time), rows in by_key.items():
        index = {
            (
                int(r["source_block_i"]) // max(1, scale),
                int(r["source_block_j"]) // max(1, scale),
                int(r["source_block_k"]) // max(1, scale),
            ): r
            for r in rows
        }
        seen: set[tuple[int, int, int]] = set()
        for key in sorted(index):
            if key in seen:
                continue
            q: deque[tuple[int, int, int]] = deque([key])
            seen.add(key)
            group: list[dict[str, Any]] = []
            while q:
                cur = q.popleft()
                row = index[cur]
                group.append(row)
                for axis in range(3):
                    for delta in (-1, 1):
                        nxt = list(cur)
                        nxt[axis] += delta
                        nt = tuple(nxt)
                        if nt in index and nt not in seen:
                            seen.add(nt)
                            q.append(nt)
            clusters.append(group)
    return clusters


def _cluster_rows(run: str, n: int, candidates: list[dict[str, Any]], cylinder_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source = {
        (
            int(row["source_scale_cells"]),
            int(row["source_time_index"]),
            int(row["source_block_i"]),
            int(row["source_block_j"]),
            int(row["source_block_k"]),
        ): row
        for row in cylinder_rows
    }
    out: list[dict[str, Any]] = []
    for idx, cluster in enumerate(_cluster_candidates(candidates, n), start=1):
        rows = []
        for c in cluster:
            key = (
                int(c["source_scale_cells"]),
                int(c["source_time_index"]),
                int(c["source_block_i"]),
                int(c["source_block_j"]),
                int(c["source_block_k"]),
            )
            if key in by_source:
                rows.append(by_source[key])
        if not rows:
            continue
        trend_counts = defaultdict(int)
        for row in rows:
            trend_counts[str(row["trend_label"])] += 1
        values = [float(row["max_C_total"]) for row in rows]
        slopes = [float(row["log_slope_dlogC_dlogr"]) for row in rows]
        pressure = [float(row["pressure_fraction_at_max"]) for row in rows]
        route = "CKN_UNIFORM_DECAY_SUPPORTED"
        if trend_counts["CKN_CONCENTRATION_CANDIDATE"] > 0:
            route = "CKN_CONCENTRATION_CANDIDATE_FOUND"
        elif trend_counts["CKN_PLATEAU"] > 0:
            route = "CKN_LOCALIZED_PERSISTENT_PLATEAU"
        if pressure and max(pressure) > 0.9:
            route = "CKN_PRESSURE_DOMINATED_ARTIFACT"
        out.append(
            {
                "run": run,
                "N": n,
                "cluster_id": f"{run}_cluster{idx}",
                "source_scale_cells": int(cluster[0]["source_scale_cells"]),
                "source_time_index": int(cluster[0]["source_time_index"]),
                "candidate_count": len(rows),
                "decaying_count": trend_counts["CKN_DECAYS_UNDER_ZOOM"],
                "flat_count": trend_counts["CKN_PLATEAU"],
                "concentrating_count": trend_counts["CKN_CONCENTRATION_CANDIDATE"],
                "max_C_total": max(values) if values else 0.0,
                "median_C_total": float(np.median(values)) if values else 0.0,
                "mean_slope": float(np.mean(slopes)) if slopes else 0.0,
                "pressure_fraction_max": max(pressure) if pressure else 0.0,
                "cluster_lifetime_frames": 1,
                "route_label": route,
            }
        )
    return out


def _rows_for_input(path: Path, seeded: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], [], {"input_path": str(path), "source_available": False, "reason": "input_missing"}
    z = np.load(path)
    if "velocity_snapshots" not in z.files:
        return [], [], {"input_path": str(path), "source_available": False, "reason": "velocity_missing"}
    velocity = np.asarray(z["velocity_snapshots"])
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise SystemExit(f"{path} velocity_snapshots must have shape (T,N,N,N,3)")
    if "pressure_snapshots" not in z.files:
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
    run_key = _run_key_from_stem(path.stem)
    candidates = [dict(c) for c in seeded.get(run_key, [])]
    if not candidates:
        candidates = _fallback_candidates(velocity, pressure, int(args.fallback_top_candidates))
    candidates = candidates[: max(0, int(args.max_candidates_per_run))]
    speed3 = np.sum(np.asarray(velocity, dtype=np.float64) ** 2, axis=-1) ** 1.5
    pressure32 = np.abs(np.asarray(pressure, dtype=np.float64)) ** 1.5
    cylinder_rows: list[dict[str, Any]] = []
    for ordinal, c in enumerate(candidates, start=1):
        row = _sweep_candidate(path, run_key, n, times, meta, dx, speed3, pressure32, c, args, ordinal)
        if row is not None:
            cylinder_rows.append(row)
    cluster_rows = _cluster_rows(path.stem, n, candidates, cylinder_rows)
    return cylinder_rows, cluster_rows, {
        "input_path": str(path),
        "source_available": True,
        "pressure_available": True,
        "run": path.stem,
        "run_key": run_key,
        "N": n,
        "frame_count": int(velocity.shape[0]),
        "candidate_count": len(candidates),
        "cylinder_count": len(cylinder_rows),
        "cluster_count": len(cluster_rows),
        "domain_length": domain_length,
        "dx": dx,
    }


def _route(summary: dict[str, Any]) -> str:
    if not summary["source_available"]:
        return "CKN_UNIFORMITY_SOURCE_UNAVAILABLE"
    if summary["pressure_reconstruction_missing"]:
        return "CKN_PRESSURE_DOMINATED_ARTIFACT"
    if int(summary["cylinder_count"]) == 0:
        return "CKN_INCONCLUSIVE_NEEDS_HIGHER_N"
    if int(summary["concentrating_count"]) > 0:
        return "CKN_CONCENTRATION_CANDIDATE_FOUND"
    if int(summary["flat_count"]) > 0 or int(summary["persistent_cluster_count"]) > 0:
        return "CKN_LOCALIZED_PERSISTENT_PLATEAU"
    if float(summary["pressure_fraction_max"]) > 0.9:
        return "CKN_PRESSURE_DOMINATED_ARTIFACT"
    return "CKN_UNIFORM_DECAY_SUPPORTED"


def _build_summary(cylinder_rows: list[dict[str, Any]], cluster_rows: list[dict[str, Any]], manifests: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    trend_counts = defaultdict(int)
    route_counts = defaultdict(int)
    for row in cylinder_rows:
        trend_counts[str(row["trend_label"])] += 1
        route_counts[str(row["route_label"])] += 1
    persistent_clusters = [
        row
        for row in cluster_rows
        if int(row["flat_count"]) > 0 or int(row["concentrating_count"]) > 0
    ]
    pressure_fraction_max = max([float(row["pressure_fraction_at_max"]) for row in cylinder_rows], default=0.0)
    max_c_by_n: dict[int, float] = {}
    for row in cylinder_rows:
        n = int(row["N"])
        max_c_by_n[n] = max(max_c_by_n.get(n, 0.0), float(row["max_C_total"]))
    ns = sorted(max_c_by_n)
    max_ckn_grows_with_n = False
    if len(ns) >= 2:
        max_ckn_grows_with_n = any(max_c_by_n[ns[i]] > max_c_by_n[ns[i - 1]] * 1.25 for i in range(1, len(ns)))
    summary: dict[str, Any] = {
        "contract": "ns_sprint67_ckn_uniformity_audit_artifact",
        "diagnostic_mode": "fixed_block_ascended_candidate_ckn_r_sweep_uniformity",
        "source_available": any(bool(m.get("source_available", False)) for m in manifests),
        "input_count": len(manifests),
        "cylinder_count": len(cylinder_rows),
        "cluster_count": len(cluster_rows),
        "decaying_count": trend_counts["CKN_DECAYS_UNDER_ZOOM"],
        "flat_count": trend_counts["CKN_PLATEAU"],
        "concentrating_count": trend_counts["CKN_CONCENTRATION_CANDIDATE"],
        "pressure_fraction_max": pressure_fraction_max,
        "max_C_total": max([float(row["max_C_total"]) for row in cylinder_rows], default=0.0),
        "median_C_total": float(np.median([float(row["max_C_total"]) for row in cylinder_rows])) if cylinder_rows else 0.0,
        "persistent_cluster_count": len(persistent_clusters),
        "max_ckn_by_N": {str(k): v for k, v in sorted(max_c_by_n.items())},
        "max_ckn_grows_with_N": max_ckn_grows_with_n,
        "pressure_available_all": bool(manifests) and all(bool(m.get("pressure_available", False)) for m in manifests if m.get("source_available", False)),
        "pressure_reconstruction_missing": not (bool(manifests) and all(bool(m.get("pressure_available", False)) for m in manifests if m.get("source_available", False))),
        "epsilon_critical": float(args.epsilon_critical),
        "r_cells": [int(r) for r in args.r_cells],
        "trend_tolerance": float(args.trend_tolerance),
        "candidate_policy": "Sprint64 fixed-block ascended cylinders if provided, otherwise top pointwise |u|^3+|p|^(3/2) density",
        "all_or_nearly_all_decay_under_zoom": bool(cylinder_rows) and trend_counts["CKN_CONCENTRATION_CANDIDATE"] == 0 and trend_counts["CKN_PLATEAU"] == 0,
        "ascended_clusters_short_lived": len(persistent_clusters) == 0,
        "local_critical_concentration_proved": False,
        "ckn_epsilon_regularity_applied": False,
        "suitable_weak_solution_bridge_proved": False,
        "continuum_uniformity_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT67_CKN_UNIFORMITY_DIAGNOSTIC",
        "route_label_counts": dict(route_counts),
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
    run_keys: set[str] = set()
    for path in args.inputs:
        run_keys.add(_run_key_from_stem(path.stem))
    seeded = _load_sprint64_candidates(args.sprint64_csv, run_keys, int(args.max_candidates_per_run))
    cylinder_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in args.inputs:
        c, cl, m = _rows_for_input(path, seeded, args)
        cylinder_rows.extend(c)
        cluster_rows.extend(cl)
        manifests.append(m)
    summary = _build_summary(cylinder_rows, cluster_rows, manifests, args)
    summary["inputs"] = [str(path) for path in args.inputs]
    summary["sprint64_csv"] = str(args.sprint64_csv) if args.sprint64_csv is not None else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cyl_path = args.out_dir / "ns_sprint67_ckn_uniformity_by_cylinder.csv"
    cl_path = args.out_dir / "ns_sprint67_ckn_uniformity_by_cluster.csv"
    summary_path = args.out_dir / "ns_sprint67_ckn_uniformity_summary.json"
    summary["by_cylinder_path"] = str(cyl_path)
    summary["by_cluster_path"] = str(cl_path)
    _write_csv(cyl_path, CYLINDER_FIELDS, cylinder_rows)
    _write_csv(cl_path, CLUSTER_FIELDS, cluster_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint67_ckn_uniformity_audit] wrote {cyl_path}")
    print(f"[ns_sprint67_ckn_uniformity_audit] wrote {cl_path}")
    print(f"[ns_sprint67_ckn_uniformity_audit] wrote {summary_path}")
    print(
        "[ns_sprint67_ckn_uniformity_audit] "
        f"route={summary['route_decision']} cylinders={summary['cylinder_count']} "
        f"clusters={summary['cluster_count']} decaying={summary['decaying_count']} "
        f"flat={summary['flat_count']} concentrating={summary['concentrating_count']} "
        f"promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
