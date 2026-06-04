#!/usr/bin/env python3
"""Sprint 54 no-2-cycle resolution/cadence audit.

This is an additive calibration layer over Sprint 53.  It consumes Sprint 49
material-parent directories, recomputes the Sprint 53 material net-residue
cycle rows, then groups them by truth snapshot cadence and resolution.  The
direct vortex-stretching packet amplitude is reported as unavailable in v1
unless a later packet-mask join is supplied; the mass-based no-2-cycle proxy is
kept separate from that unavailable theorem-grade stretch quantity.
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


EPS = 1e-30

STRETCH_FIELDS = [
    "run",
    "cadence",
    "resolution_N",
    "K",
    "t",
    "cycle_id",
    "packet_mass_amplitude",
    "stretching_amplitude",
    "weighted_stretching_amplitude",
    "signed_stretching_imbalance",
    "physical_small_by_stretch",
    "physical_small_by_mass",
    "sigma_stretching_amplitude_fit",
    "sigma_mass_amplitude_fit",
    "direct_stretch_status",
    "direct_stretch_boundary",
]

CADENCE_FIELDS = [
    "run",
    "N",
    "cadence",
    "save_every",
    "dt",
    "shell_convention",
    "large_cycle_count",
    "physical_small_cycle_fraction",
    "sigma_packet_cycle",
    "sigma_direct_stretch_cycle",
    "weighted_packet_cycle_amplitude",
    "weighted_stretch_cycle_amplitude",
    "shell_boundary_sensitivity",
    "direct_stretch_status",
    "route_decision",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Sprint 49 material-parent output directories",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cycle-damping-threshold", type=float, default=1.0 / math.sqrt(2.0))
    p.add_argument("--physical-amplitude-small-fraction", type=float, default=0.05)
    p.add_argument("--physical-amplitude-small-majority", type=float, default=0.90)
    p.add_argument("--sigma-threshold", type=float, default=0.5)
    p.add_argument(
        "--truth-root",
        type=Path,
        default=Path("."),
        help="root used to resolve source_truth paths in Sprint 49 JSON summaries",
    )
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _num(value: Any) -> float:
    return sprint53._num(value)


def _fit_sigma_by_k(rows: list[dict[str, Any]], value_key: str = "weighted_abs_delta_N") -> float:
    by_k: dict[int, float] = defaultdict(float)
    for row in rows:
        if row.get("proxy_failed") != "true":
            continue
        by_k[int(float(row["K"]))] += _num(row.get(value_key))
    return sprint53._fit_sigma(by_k)


def _fallback_n_from_name(run: str) -> int:
    match = re.search(r"N(\d+)", run)
    return int(match.group(1)) if match else 0


def _truth_meta(input_dir: Path, summary: dict[str, Any], truth_root: Path) -> dict[str, Any]:
    source_truth = str(summary.get("source_truth") or "")
    truth_path = Path(source_truth)
    if source_truth and not truth_path.is_absolute():
        truth_path = truth_root / truth_path
    meta: dict[str, Any] = {}
    if source_truth and truth_path.exists():
        try:
            with np.load(truth_path, allow_pickle=False) as data:
                if "meta_json" in data:
                    meta = json.loads(str(data["meta_json"]))
        except Exception as exc:  # pragma: no cover - defensive metadata boundary.
            meta = {"meta_load_error": str(exc)}
    return {
        "input_dir": str(input_dir),
        "run": input_dir.name,
        "source_truth": source_truth,
        "truth_path": str(truth_path) if source_truth else "",
        "N": int(meta.get("N") or _fallback_n_from_name(input_dir.name)),
        "save_every": int(meta.get("save_every") or 0),
        "dt": float(meta.get("dt") or 0.0),
        "snapshots": int(meta.get("snapshots") or 0),
        "direct_stretch_status": (
            "truth_fields_available_packet_mask_join_unavailable"
            if source_truth
            else "source_truth_unavailable"
        ),
    }


def _read_truth_metadata(inputs: list[Path], truth_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for input_dir in inputs:
        summary_path = input_dir / "ns_material_parent_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        out[str(input_dir)] = _truth_meta(input_dir, summary, truth_root)
    return out


def _build_shell_map(n: int, L: float) -> np.ndarray:
    dk = np.fft.fftfreq(n, d=L / float(n)) * 2.0 * math.pi
    kz, ky, kx = np.meshgrid(dk, dk, dk, indexing="ij")
    radius = np.sqrt(kx * kx + ky * ky + kz * kz)
    shells = np.zeros_like(radius, dtype=np.int64)
    mask = radius >= 1.0
    shells[mask] = np.floor(radius[mask]).astype(np.int64)
    return shells


def _build_velocity_gradient(u: np.ndarray, L: float) -> np.ndarray:
    h = L / float(u.shape[0])
    inv_two_h = 0.5 / h
    grad = np.empty(u.shape + (3,), dtype=np.float64)
    for comp in range(3):
        grad_comp = grad[..., comp, :]
        for axis in range(3):
            grad_comp[..., axis] = (
                np.roll(u[..., comp], -1, axis=axis)
                - np.roll(u[..., comp], 1, axis=axis)
            ) * inv_two_h
        grad[..., comp, :] = grad_comp
    return grad


def _load_direct_shell_stretch(
    truth_meta: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, float, int], dict[str, float]], dict[str, str]]:
    stretch_by_key: dict[tuple[str, float, int], dict[str, float]] = {}
    status_by_run: dict[str, str] = {}
    for meta in truth_meta.values():
        run = str(meta["run"])
        truth_path = Path(str(meta.get("truth_path") or ""))
        if not truth_path.exists():
            status_by_run[run] = "source_truth_unavailable"
            continue
        try:
            with np.load(truth_path, allow_pickle=False) as data:
                if "omega_snapshots" not in data.files or "velocity_snapshots" not in data.files:
                    status_by_run[run] = "truth_velocity_or_omega_unavailable"
                    continue
                omega = np.asarray(data["omega_snapshots"], dtype=np.float64)
                velocity = np.asarray(data["velocity_snapshots"], dtype=np.float64)
                steps = np.asarray(data["steps"], dtype=np.float64) if "steps" in data.files else np.arange(len(omega))
                meta_json = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
        except Exception as exc:  # pragma: no cover - defensive metadata boundary.
            status_by_run[run] = f"truth_load_failed:{exc}"
            continue
        if omega.shape != velocity.shape or omega.ndim != 5 or omega.shape[-1] != 3:
            status_by_run[run] = "truth_shape_incompatible"
            continue
        dt = float(meta_json.get("dt") or meta.get("dt") or 0.0)
        L = float(meta_json.get("domain_length") or (2.0 * math.pi))
        shell_map = _build_shell_map(int(omega.shape[1]), L)
        shell_ids = sorted(int(k) for k in np.unique(shell_map))
        for t_idx, frame in enumerate(omega):
            time = float(steps[t_idx] * dt) if dt > 0.0 and t_idx < len(steps) else float(t_idx)
            grad_u = _build_velocity_gradient(velocity[t_idx], L)
            stretch = np.einsum("...i,...ij,...j->...", frame, grad_u, frame)
            for k in shell_ids:
                mask = shell_map == k
                signed = float(np.sum(stretch[mask]))
                amplitude = abs(signed)
                stretch_by_key[(run, time, k)] = {
                    "signed_stretching_imbalance": signed,
                    "stretching_amplitude": amplitude,
                    "weighted_stretching_amplitude": (2.0 ** (0.5 * float(k))) * amplitude,
                }
        status_by_run[run] = "shell_time_direct_stretch_available_packet_mask_join_unavailable"
    return stretch_by_key, status_by_run


def _route(summary: dict[str, Any]) -> str:
    if summary["direct_stretch_status"] == "stretch_packet_mask_join_unavailable":
        if summary["cadence_sensitivity"] == "single_cadence_unresolved":
            return "NO2CYCLE_UNRESOLVED_NEEDS_HIGHER_N"
        if summary["physical_small_by_mass_fraction"] < summary["physical_amplitude_small_majority"]:
            return "NO2CYCLE_PHYSICAL_AMPLITUDE_BLOCKED"
    if bool(summary.get("does_direct_stretch_gate_close")):
        return "NO2CYCLE_PROXY_OVERCONSERVATIVE_STRETCH_SMALL"
    if summary["cadence_sensitivity"] == "large_cycles_shrink_with_denser_cadence":
        return "NO2CYCLE_TEMPORAL_ALIASING"
    return "NO2CYCLE_UNRESOLVED_NEEDS_HIGHER_N"


def _cadence_sensitivity(cadence_rows: list[dict[str, Any]]) -> str:
    cadences = sorted({int(row["save_every"]) for row in cadence_rows if int(row["save_every"]) > 0})
    if len(cadences) < 2:
        return "single_cadence_unresolved"
    by_cadence: dict[int, list[float]] = defaultdict(list)
    for row in cadence_rows:
        by_cadence[int(row["save_every"])].append(float(row["physical_small_cycle_fraction"]))
    dense = min(cadences)
    coarse = max(cadences)
    dense_fraction = sum(by_cadence[dense]) / len(by_cadence[dense])
    coarse_fraction = sum(by_cadence[coarse]) / len(by_cadence[coarse])
    if dense_fraction > coarse_fraction + 0.10:
        return "large_cycles_shrink_with_denser_cadence"
    if dense_fraction + 0.10 < coarse_fraction:
        return "large_cycles_persist_or_worsen_with_denser_cadence"
    return "cadence_difference_inconclusive"


def _resolution_sensitivity(cadence_rows: list[dict[str, Any]]) -> str:
    ns = sorted({int(row["N"]) for row in cadence_rows if int(row["N"]) > 0})
    if len(ns) < 2:
        return "single_resolution_unresolved"
    by_n: dict[int, list[float]] = defaultdict(list)
    for row in cadence_rows:
        by_n[int(row["N"])].append(float(row["physical_small_cycle_fraction"]))
    low = min(ns)
    high = max(ns)
    low_fraction = sum(by_n[low]) / len(by_n[low])
    high_fraction = sum(by_n[high]) / len(by_n[high])
    if high_fraction > low_fraction + 0.10:
        return "large_cycles_shrink_with_higher_resolution"
    if high_fraction + 0.10 < low_fraction:
        return "large_cycles_persist_or_worsen_with_higher_resolution"
    return "resolution_difference_inconclusive"


def _build_outputs(
    amp_rows: list[dict[str, Any]],
    amp_summary: dict[str, Any],
    truth_meta: dict[str, dict[str, Any]],
    direct_stretch: dict[tuple[str, float, int], dict[str, float]],
    direct_status_by_run: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sigma_mass = _fit_sigma_by_k(amp_rows)
    boundary = (
        "Sprint 54 v1 records packet-mass sign-cycle amplitude and cadence/resolution "
        "metadata. When truth fields are present it also records shell/time omega dot "
        "S omega amplitudes; packet-local stretch attribution still requires packet "
        "support masks and is unavailable in this artifact."
    )
    stretch_rows: list[dict[str, Any]] = []
    by_k_stretch: dict[int, float] = defaultdict(float)
    stretch_small_count = 0
    stretch_available_count = 0
    for row in amp_rows:
        meta = truth_meta.get(str(Path(row["run"])), {})
        if not meta:
            meta = next((m for m in truth_meta.values() if m["run"] == row["run"]), {})
        save_every = int(meta.get("save_every") or row.get("save_every") or 0)
        run = row["run"]
        k = int(float(row["K"]))
        time = float(row["time"])
        direct = direct_stretch.get((run, time, k))
        direct_status = direct_status_by_run.get(run, "source_truth_unavailable")
        weighted_stretch = "unavailable"
        stretch_amp = "unavailable"
        signed_stretch = "unavailable"
        stretch_small = "unavailable"
        if row.get("proxy_failed") == "true" and direct is not None:
            stretch_available_count += 1
            stretch_amp_float = float(direct["stretching_amplitude"])
            weighted_stretch_float = float(direct["weighted_stretching_amplitude"])
            by_k_stretch[k] += weighted_stretch_float
            stretch_denominator = max(_num(row["Rplus_t"]), _num(row["Rplus_t_next"]), EPS)
            is_small = stretch_amp_float / stretch_denominator <= float(args.physical_amplitude_small_fraction)
            stretch_small_count += 1 if is_small else 0
            weighted_stretch = _fmt(weighted_stretch_float)
            stretch_amp = _fmt(stretch_amp_float)
            signed_stretch = _fmt(float(direct["signed_stretching_imbalance"]))
            stretch_small = _fmt(is_small)
        stretch_rows.append(
            {
                "run": run,
                "cadence": str(save_every),
                "resolution_N": str(int(meta.get("N") or _fallback_n_from_name(row["run"]))),
                "K": row["K"],
                "t": row["time"],
                "cycle_id": row["cycle_id"],
                "packet_mass_amplitude": row["abs_delta_N"],
                "stretching_amplitude": stretch_amp,
                "weighted_stretching_amplitude": weighted_stretch,
                "signed_stretching_imbalance": signed_stretch,
                "physical_small_by_stretch": stretch_small,
                "physical_small_by_mass": row["physical_amplitude_small"],
                "sigma_stretching_amplitude_fit": "unavailable",
                "sigma_mass_amplitude_fit": _fmt(sigma_mass),
                "direct_stretch_status": direct_status,
                "direct_stretch_boundary": boundary,
            }
        )

    run_rows: list[dict[str, Any]] = []
    fail_rows = [row for row in amp_rows if row["proxy_failed"] == "true"]
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fail_rows:
        by_run[row["run"]].append(row)
    sigma_stretch = sprint53._fit_sigma(by_k_stretch)
    for row in stretch_rows:
        row["sigma_stretching_amplitude_fit"] = _fmt(sigma_stretch) if stretch_available_count else "unavailable"
    for run, rows in sorted(by_run.items()):
        meta = next((m for m in truth_meta.values() if m["run"] == run), {})
        failures = len(rows)
        small = sum(1 for row in rows if row["physical_amplitude_small"] == "true")
        large = failures - small
        weighted = sum(_num(row["weighted_abs_delta_N"]) for row in rows)
        stretch_weighted = sum(
            _num(row["weighted_stretching_amplitude"])
            for row in stretch_rows
            if row["run"] == run and row["weighted_stretching_amplitude"] != "unavailable"
        )
        run_direct_status = direct_status_by_run.get(run, "source_truth_unavailable")
        save_every = int(meta.get("save_every") or rows[0].get("save_every") or 0)
        fraction = small / max(failures, 1)
        run_rows.append(
            {
                "run": run,
                "N": str(int(meta.get("N") or _fallback_n_from_name(run))),
                "cadence": str(save_every),
                "save_every": str(save_every),
                "dt": _fmt(float(meta.get("dt") or rows[0].get("dt") or 0.0)),
                "shell_convention": "integer_fourier_radius_shell_index_from_sprint49_material_parent_K",
                "large_cycle_count": str(large),
                "physical_small_cycle_fraction": _fmt(fraction),
                "sigma_packet_cycle": _fmt(sigma_mass),
                "sigma_direct_stretch_cycle": _fmt(sigma_stretch) if stretch_available_count else "unavailable",
                "weighted_packet_cycle_amplitude": _fmt(weighted),
                "weighted_stretch_cycle_amplitude": _fmt(stretch_weighted) if stretch_weighted > 0.0 else "unavailable",
                "shell_boundary_sensitivity": "not_tested_v1",
                "direct_stretch_status": run_direct_status,
                "route_decision": "pending",
            }
        )

    cadence_status = _cadence_sensitivity(run_rows)
    resolution_status = _resolution_sensitivity(run_rows)
    physical_small_by_mass = sum(1 for row in fail_rows if row["physical_amplitude_small"] == "true")
    physical_mass_fraction = physical_small_by_mass / max(len(fail_rows), 1)
    stretch_fraction = stretch_small_count / max(stretch_available_count, 1)
    direct_status = (
        "shell_time_direct_stretch_available_packet_mask_join_unavailable"
        if stretch_available_count
        else "stretch_packet_mask_join_unavailable"
    )
    summary: dict[str, Any] = {
        "contract": "ns_sprint54_cycle_amplitude_artifact",
        "diagnostic_mode": "sprint54_no2cycle_resolution_cadence_from_material_residue",
        "input_table_row_count": int(amp_summary.get("input_table_row_count", 0)),
        "cycle_amplitude_row_count": len(stretch_rows),
        "cadence_comparison_row_count": len(run_rows),
        "no2cycle_candidate_count": int(amp_summary["no2cycle_candidate_count"]),
        "no2cycle_proxy_failure_count": int(amp_summary["no2cycle_proxy_failure_count"]),
        "physical_large_cycle_count": int(amp_summary["physical_large_cycle_count"]),
        "physical_small_by_mass_count": physical_small_by_mass,
        "physical_small_by_mass_fraction": physical_mass_fraction,
        "physical_amplitude_small_majority": float(args.physical_amplitude_small_majority),
        "direct_stretch_available_cycle_count": stretch_available_count,
        "physical_small_by_stretch_count": stretch_small_count,
        "small_fraction_by_stretch": stretch_fraction if stretch_available_count else 0.0,
        "small_fraction_by_mass": physical_mass_fraction,
        "sigma_stretching_amplitude": sigma_stretch if stretch_available_count else "unavailable",
        "sigma_mass_amplitude": sigma_mass,
        "cadence_sensitivity": cadence_status,
        "resolution_sensitivity": resolution_status,
        "direct_stretch_status": direct_status,
        "direct_stretch_boundary": boundary,
        "shell_boundary_sensitivity": "not_tested_v1",
        "does_no2cycle_cadence_gate_close": False,
        "does_no2cycle_resolution_gate_close": False,
        "does_direct_stretch_gate_close": bool(
            stretch_available_count and stretch_fraction >= float(args.physical_amplitude_small_majority)
        ),
        "does_mass_proxy_gate_close": physical_mass_fraction >= float(args.physical_amplitude_small_majority),
        "no2cycle_aliasing_proved": False,
        "no2cycle_shell_boundary_artifact_proved": False,
        "no2cycle_proxy_overconservative_proved": False,
        "no2cycle_physical_obstruction_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT54_NO2CYCLE_RESOLUTION_CADENCE_DIAGNOSTIC",
        "receipt_alignment": "DASHI.Physics.Closure.ClaySprintFiftyFourNo2CycleResolutionCadenceAuditReceipt",
    }
    summary["route_decision"] = _route(summary)
    for row in run_rows:
        row["route_decision"] = summary["route_decision"]
    return stretch_rows, run_rows, summary


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    table_rows, summary_rows, _manifests = sprint53._read_inputs(args.inputs)
    _material_summary = sprint53._material_source_gate(summary_rows, float(args.sigma_threshold))
    amp_rows, _lyap_rows, amp_summary = sprint53._cycle_rows(
        table_rows,
        damping_threshold=float(args.cycle_damping_threshold),
        small_fraction=float(args.physical_amplitude_small_fraction),
    )
    amp_summary["input_table_row_count"] = len(table_rows)
    truth_meta = _read_truth_metadata(args.inputs, args.truth_root)
    direct_stretch, direct_status_by_run = _load_direct_shell_stretch(truth_meta)
    stretch_rows, cadence_rows, summary = _build_outputs(
        amp_rows,
        amp_summary,
        truth_meta,
        direct_stretch,
        direct_status_by_run,
        args,
    )
    summary["inputs"] = [str(path) for path in args.inputs]
    summary["truth_metadata"] = list(truth_meta.values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stretch_path = args.out_dir / "ns_stretch_cycle_physical_amplitude.csv"
    cadence_path = args.out_dir / "ns_no2cycle_cadence_comparison.csv"
    summary_path = args.out_dir / "ns_sprint54_cycle_amplitude_summary.json"
    summary["ns_stretch_cycle_physical_amplitude_path"] = str(stretch_path)
    summary["ns_no2cycle_cadence_comparison_path"] = str(cadence_path)
    _write_csv(stretch_path, STRETCH_FIELDS, stretch_rows)
    _write_csv(cadence_path, CADENCE_FIELDS, cadence_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint54_no2cycle_resolution_cadence_audit] wrote {stretch_path}")
    print(f"[ns_sprint54_no2cycle_resolution_cadence_audit] wrote {cadence_path}")
    print(f"[ns_sprint54_no2cycle_resolution_cadence_audit] wrote {summary_path}")
    print(
        "[ns_sprint54_no2cycle_resolution_cadence_audit] "
        f"route={summary['route_decision']} "
        f"mass_fraction={summary['physical_small_by_mass_fraction']} "
        f"cadence={summary['cadence_sensitivity']} "
        f"resolution={summary['resolution_sensitivity']}"
    )


if __name__ == "__main__":
    main()
