#!/usr/bin/env python3
"""Sprint 53 physical no-2-cycle amplitude audit.

This consumes Sprint 49 material-parent artifacts and recalibrates Sprint 52
no-2-cycle proxy failures against shell/time net-residue motion:

    N_K(t) = Rplus_K(t) - Rminus_K(t)

The v1 "physical" amplitude is the observed material shell residue delta
``abs(N_K(t_next) - N_K(t))`` at the second leg of a +/- packet cycle.  This is
still a diagnostic proxy, not theorem-grade continuum stretch amplitude.  The
script does not rerun GPU truth, FFTs, or material-parent matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


EPS = 1e-30

PHYSICAL_AMPLITUDE_FIELDS = [
    "run",
    "time",
    "K",
    "cycle_id",
    "Rplus_t",
    "Rminus_t",
    "N_t",
    "Rplus_t_next",
    "Rminus_t_next",
    "N_t_next",
    "delta_N",
    "abs_delta_N",
    "weighted_abs_delta_N",
    "proxy_failed",
    "amplitude_small",
    "physical_amplitude_small",
    "packet_confidence",
    "parent_relation",
    "dt",
    "save_every",
]

CADENCE_FIELDS = [
    "run",
    "dt",
    "save_every",
    "no2cycle_proxy_failure_count",
    "physical_large_cycle_count",
    "physical_small_cycle_fraction",
    "weighted_physical_cycle_amplitude_total",
    "sigma_physical_cycle_fit",
    "does_physical_cycle_gate_close",
]

LYAPUNOV_FIELDS = [
    "run",
    "K",
    "time",
    "N_t",
    "N_t_next",
    "delta_N",
    "abs_delta_N",
    "weighted_abs_delta_N",
    "q_net_residue_proxy",
    "q_sqrt2",
    "does_net_residue_contract",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Sprint 49 output directories containing material-parent table and summary artifacts",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cycle-damping-threshold", type=float, default=1.0 / math.sqrt(2.0))
    p.add_argument(
        "--physical-amplitude-small-fraction",
        type=float,
        default=0.05,
        help="physical amplitude is small if <= this fraction of shell plus weighted mass",
    )
    p.add_argument(
        "--physical-amplitude-small-majority",
        type=float,
        default=0.90,
        help="fraction of proxy-failing cycles that must be physically small to clear the gate",
    )
    p.add_argument("--sigma-threshold", type=float, default=0.5)
    return p.parse_args()


def _num(value: str | float | int | None) -> float:
    try:
        out = float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _int(value: str | int | None) -> int:
    try:
        return int(float(value)) if value not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


def _fmt(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def _require_input_dir(path: Path) -> tuple[Path, Path, dict[str, Any]]:
    table = path / "ns_material_parent_table.csv"
    summary_csv = path / "ns_material_parent_summary.csv"
    summary_json = path / "ns_material_parent_summary.json"
    if not table.exists():
        raise SystemExit(f"{path} lacks ns_material_parent_table.csv")
    if not summary_csv.exists():
        raise SystemExit(f"{path} lacks ns_material_parent_summary.csv")
    if not summary_json.exists():
        raise SystemExit(f"{path} lacks ns_material_parent_summary.json")
    try:
        meta = json.loads(summary_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{summary_json} is not valid JSON: {exc}") from exc
    return table, summary_csv, meta


def _read_inputs(inputs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    table_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    table_required = {
        "time",
        "dt",
        "K_parent",
        "K_child",
        "child_packet_id",
        "parent_packet_id",
        "child_state",
        "parent_state",
        "child_mass",
        "credited_mass",
        "parent_confidence",
        "parent_relation",
    }
    summary_required = {"time", "K_child", "weighted_true_new", "sigma_true_new_fit"}
    for input_dir in inputs:
        table, summary_csv, meta = _require_input_dir(input_dir)
        manifests.append(
            {
                "input_dir": str(input_dir),
                "summary_contract": meta.get("contract"),
                "summary_row_count": meta.get("summary_row_count"),
                "table_row_count": meta.get("table_row_count"),
                "material_parent_route_status": meta.get("material_parent_route_status"),
                "source_truth": meta.get("source_truth"),
            }
        )
        with table.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(table_required.difference(reader.fieldnames or []))
            if missing:
                raise SystemExit(f"{table} is missing columns: {', '.join(missing)}")
            for raw in reader:
                k_child = _int(raw.get("K_child"))
                weighted = (2.0 ** (0.5 * float(k_child))) * _num(raw.get("credited_mass"))
                table_rows.append(
                    {
                        **raw,
                        "input_dir": str(input_dir),
                        "run": input_dir.name,
                        "time_float": _num(raw.get("time")),
                        "dt_float": _num(raw.get("dt")),
                        "K_parent_int": _int(raw.get("K_parent")),
                        "K_child_int": k_child,
                        "credited_mass_float": _num(raw.get("credited_mass")),
                        "child_mass_float": _num(raw.get("child_mass")),
                        "weighted_mass_float": weighted,
                        "parent_confidence_float": _num(raw.get("parent_confidence")),
                    }
                )
        with summary_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(summary_required.difference(reader.fieldnames or []))
            if missing:
                raise SystemExit(f"{summary_csv} is missing columns: {', '.join(missing)}")
            for raw in reader:
                summary_rows.append({**raw, "input_dir": str(input_dir), "run": input_dir.name})
    return table_rows, summary_rows, manifests


def _transition(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("parent_state", "")), str(row.get("child_state", ""))


def _save_every(rows: list[dict[str, Any]]) -> int:
    dts = sorted({float(row["dt_float"]) for row in rows if float(row["dt_float"]) > 0.0})
    if not dts:
        return 1
    min_dt = min(dts)
    return max(1, int(round(dts[0] / max(min_dt, EPS))))


def _material_source_gate(summary_rows: list[dict[str, Any]], sigma_threshold: float) -> dict[str, Any]:
    weighted_true = sum(_num(row.get("weighted_true_new")) for row in summary_rows)
    sigma = max((_num(row.get("sigma_true_new_fit")) for row in summary_rows), default=0.0)
    absent = weighted_true <= EPS
    beats = sigma > sigma_threshold
    return {
        "weighted_true_new_material_total": weighted_true,
        "sigma_true_new_material": sigma,
        "material_true_new_source_absent": absent,
        "material_source_beats_half_derivative": beats,
        "does_material_source_gate_close": absent or beats,
    }


def _shell_residue(table_rows: list[dict[str, Any]]) -> dict[tuple[str, float, int], dict[str, float]]:
    residue: dict[tuple[str, float, int], dict[str, float]] = defaultdict(lambda: {"Rplus": 0.0, "Rminus": 0.0})
    for row in table_rows:
        key = (str(row["input_dir"]), float(row["time_float"]), int(row["K_child_int"]))
        weighted_child = float(row["weighted_mass_float"])
        if row.get("child_state") == "plus":
            residue[key]["Rplus"] += weighted_child
        elif row.get("child_state") == "minus":
            residue[key]["Rminus"] += weighted_child
    return residue


def _next_key(times_by_run_k: dict[tuple[str, int], list[float]], run: str, time: float, k: int) -> tuple[str, float, int] | None:
    times = times_by_run_k.get((run, k), [])
    for candidate in times:
        if candidate > time + EPS:
            return (run, candidate, k)
    return None


def _cycle_rows(
    table_rows: list[dict[str, Any]],
    damping_threshold: float,
    small_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    residue = _shell_residue(table_rows)
    times_by_run_k: dict[tuple[str, int], list[float]] = defaultdict(list)
    for run, time, k in residue:
        times_by_run_k[(run, k)].append(time)
    for key in list(times_by_run_k):
        times_by_run_k[key] = sorted(set(times_by_run_k[key]))

    flips = [
        row
        for row in table_rows
        if _transition(row) in {("minus", "plus"), ("plus", "minus")} and float(row["credited_mass_float"]) > 0.0
    ]
    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        rows_by_run[str(row["input_dir"])].append(row)
    save_every_by_run = {run: _save_every(rows) for run, rows in rows_by_run.items()}
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in flips:
        by_parent[(str(row["input_dir"]), str(row.get("parent_packet_id", "")))].append(row)

    rows: list[dict[str, Any]] = []
    lyap_rows: list[dict[str, Any]] = []
    by_k_weighted: dict[int, float] = defaultdict(float)
    run_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"failures": 0.0, "small": 0.0, "large": 0.0, "weighted": 0.0, "dt": 0.0, "save_every": 1.0}
    )
    cycle_idx = 0
    for first in flips:
        first_child_key = (str(first["input_dir"]), str(first.get("child_packet_id", "")))
        parent, child = _transition(first)
        opposite = (child, parent)
        candidates = [
            second
            for second in by_parent.get(first_child_key, [])
            if _transition(second) == opposite and float(second["time_float"]) > float(first["time_float"])
        ]
        if not candidates:
            continue
        second = min(candidates, key=lambda item: float(item["time_float"]))
        first_weight = float(first["weighted_mass_float"])
        second_weight = float(second["weighted_mass_float"])
        ratio = second_weight / max(first_weight, EPS)
        proxy_failed = ratio > damping_threshold
        run = str(second["input_dir"])
        k = int(second["K_child_int"])
        time = float(second["time_float"])
        key_t = (run, time, k)
        key_next = _next_key(times_by_run_k, run, time, k)
        if key_next is None:
            continue
        r_t = residue[key_t]
        r_next = residue[key_next]
        n_t = r_t["Rplus"] - r_t["Rminus"]
        n_next = r_next["Rplus"] - r_next["Rminus"]
        delta_n = n_next - n_t
        abs_delta = abs(delta_n)
        weighted_abs = (2.0 ** (0.5 * float(k))) * abs_delta
        shell_denominator = max(r_t["Rplus"], r_next["Rplus"], EPS)
        amplitude_fraction = abs_delta / shell_denominator
        physical_small = amplitude_fraction <= small_fraction
        cycle_idx += 1
        if proxy_failed:
            run_stats[run]["failures"] += 1.0
            run_stats[run]["small"] += 1.0 if physical_small else 0.0
            run_stats[run]["large"] += 0.0 if physical_small else 1.0
            run_stats[run]["weighted"] += weighted_abs
            by_k_weighted[k] += weighted_abs
        run_stats[run]["dt"] = float(second["dt_float"])
        run_stats[run]["save_every"] = float(save_every_by_run.get(run, 1))
        rows.append(
            {
                "run": str(second["run"]),
                "time": _fmt(time),
                "K": str(k),
                "cycle_id": f"cycle_{cycle_idx}",
                "Rplus_t": _fmt(r_t["Rplus"]),
                "Rminus_t": _fmt(r_t["Rminus"]),
                "N_t": _fmt(n_t),
                "Rplus_t_next": _fmt(r_next["Rplus"]),
                "Rminus_t_next": _fmt(r_next["Rminus"]),
                "N_t_next": _fmt(n_next),
                "delta_N": _fmt(delta_n),
                "abs_delta_N": _fmt(abs_delta),
                "weighted_abs_delta_N": _fmt(weighted_abs),
                "proxy_failed": _fmt(proxy_failed),
                "amplitude_small": _fmt(physical_small),
                "physical_amplitude_small": _fmt(physical_small),
                "packet_confidence": _fmt(float(second["parent_confidence_float"])),
                "parent_relation": str(second.get("parent_relation", "")),
                "dt": _fmt(float(second["dt_float"])),
                "save_every": str(int(run_stats[run]["save_every"])),
            }
        )
        q = abs(n_next) / max(abs(n_t), EPS)
        lyap_rows.append(
            {
                "run": str(second["run"]),
                "K": str(k),
                "time": _fmt(time),
                "N_t": _fmt(n_t),
                "N_t_next": _fmt(n_next),
                "delta_N": _fmt(delta_n),
                "abs_delta_N": _fmt(abs_delta),
                "weighted_abs_delta_N": _fmt(weighted_abs),
                "q_net_residue_proxy": _fmt(q),
                "q_sqrt2": _fmt(q * math.sqrt(2.0)),
                "does_net_residue_contract": _fmt(q * math.sqrt(2.0) < 1.0),
            }
        )

    fail_rows = [row for row in rows if row["proxy_failed"] == "true"]
    small_failures = [row for row in fail_rows if row["physical_amplitude_small"] == "true"]
    large_failures = [row for row in fail_rows if row["physical_amplitude_small"] == "false"]
    weighted_total = sum(_num(row["weighted_abs_delta_N"]) for row in fail_rows)
    sigma = _fit_sigma(by_k_weighted)
    cadence_rows = []
    for run, stats in sorted(run_stats.items()):
        failures = int(stats["failures"])
        small_fraction = stats["small"] / max(stats["failures"], 1.0)
        cadence_rows.append(
            {
                "run": Path(run).name,
                "dt": _fmt(stats["dt"]),
                "save_every": str(int(stats["save_every"])),
                "no2cycle_proxy_failure_count": str(failures),
                "physical_large_cycle_count": str(int(stats["large"])),
                "physical_small_cycle_fraction": _fmt(small_fraction),
                "weighted_physical_cycle_amplitude_total": _fmt(stats["weighted"]),
                "sigma_physical_cycle_fit": _fmt(sigma),
                "does_physical_cycle_gate_close": _fmt(small_fraction >= 0.90),
            }
        )
    return rows, lyap_rows, {
        "no2cycle_candidate_count": len(rows),
        "no2cycle_proxy_failure_count": len(fail_rows),
        "physical_large_cycle_count": len(large_failures),
        "physical_amplitude_small_failure_count": len(small_failures),
        "physical_small_cycle_fraction": len(small_failures) / max(len(fail_rows), 1),
        "weighted_physical_cycle_amplitude_total": weighted_total,
        "sigma_physical_cycle_fit": sigma,
        "max_physical_amplitude_fraction_of_shell": max(
            (_num(row["abs_delta_N"]) / max(_num(row["Rplus_t"]), _num(row["Rplus_t_next"]), EPS) for row in rows),
            default=0.0,
        ),
        "cadence_rows": cadence_rows,
    }


def _fit_sigma(by_k_weighted: dict[int, float]) -> float:
    points = sorted((k, v) for k, v in by_k_weighted.items() if v > EPS)
    if len(points) < 2:
        return 0.0
    n = float(len(points))
    sx = sum(float(k) for k, _ in points)
    sy = sum(math.log(v, 2.0) for _, v in points)
    sxx = sum(float(k * k) for k, _ in points)
    sxy = sum(float(k) * math.log(v, 2.0) for k, v in points)
    denom = n * sxx - sx * sx
    if abs(denom) <= EPS:
        return 0.0
    slope = (n * sxy - sx * sy) / denom
    return -slope


def _route(summary: dict[str, Any]) -> str:
    if int(summary["no2cycle_candidate_count"]) == 0:
        return "PHYSICAL_TRUTH_JOIN_INSUFFICIENT"
    material = bool(summary["does_material_source_gate_close"])
    physical = bool(summary["does_physical_cycle_gate_close"])
    summability = bool(summary["does_physical_cycle_summability_gate_close"])
    if material and (physical or summability):
        return "NS_SOURCE_BUDGET_PHYSICAL_NO2CYCLE_ROUTE_ALIVE_DIAGNOSTIC"
    if material and not physical:
        return "MATERIAL_SOURCE_GATE_CLOSED_PHYSICAL_NO2CYCLE_AMPLITUDE_BLOCKED"
    return "MATERIAL_SOURCE_GATE_BLOCKED_PHYSICAL_NO2CYCLE_UNRESOLVED"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    table_rows, summary_rows, manifests = _read_inputs(args.inputs)
    material_summary = _material_source_gate(summary_rows, float(args.sigma_threshold))
    amp_rows, lyap_rows, amp_summary = _cycle_rows(
        table_rows,
        damping_threshold=float(args.cycle_damping_threshold),
        small_fraction=float(args.physical_amplitude_small_fraction),
    )
    physical_fraction = float(amp_summary["physical_small_cycle_fraction"])
    physical_gate = physical_fraction >= float(args.physical_amplitude_small_majority)
    sigma = float(amp_summary["sigma_physical_cycle_fit"])
    summability_gate = sigma > float(args.sigma_threshold)
    summary: dict[str, Any] = {
        "contract": "ns_sprint53_no2cycle_physical_artifact",
        "diagnostic_mode": "sprint53_no2cycle_physical_amplitude_from_material_residue",
        "input_table_row_count": len(table_rows),
        "physical_no2cycle_row_count": len(amp_rows),
        "physical_amplitude_row_count": len(amp_rows),
        "net_residue_lyapunov_row_count": len(lyap_rows),
        "cadence_comparison_row_count": len(amp_summary["cadence_rows"]),
        "physical_amplitude_boundary": (
            "v1 uses material shell/time net-residue delta abs(N_K(t_next)-N_K(t)); "
            "it does not recompute continuum omega dot S omega from truth snapshots"
        ),
        "cadence_boundary": "compares save cadence metadata among supplied Sprint 49 directories only",
        "physical_amplitude_small_fraction": float(args.physical_amplitude_small_fraction),
        "physical_amplitude_small_majority": float(args.physical_amplitude_small_majority),
        "sigma_threshold": float(args.sigma_threshold),
        **material_summary,
        "physical_no2cycle_failure_count": amp_summary["no2cycle_proxy_failure_count"],
        "physical_amplitude_small_failure_fraction": physical_fraction,
        "does_physical_no2cycle_amplitude_gate_close": physical_gate,
        "does_physical_cycle_gate_close": physical_gate,
        "does_physical_cycle_summability_gate_close": summability_gate,
        **{k: v for k, v in amp_summary.items() if k != "cadence_rows"},
        "physical_no2cycle_amplitude_proved": False,
        "weighted_physical_no2cycle_amplitude_summability_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT53_NO2CYCLE_PHYSICAL_DIAGNOSTIC",
        "inputs": [str(path) for path in args.inputs],
        "input_manifest_summaries": manifests,
        "receipt_alignment": "DASHI.Physics.Closure.ClaySprintFiftyThreeNo2CyclePhysicalAmplitudeReceipt",
    }
    summary["route_decision"] = _route(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    amp_path = args.out_dir / "ns_no2cycle_physical_amplitude.csv"
    cadence_path = args.out_dir / "ns_no2cycle_cadence_comparison.csv"
    lyap_path = args.out_dir / "ns_net_residue_physical_lyapunov.csv"
    summary_path = args.out_dir / "ns_sprint53_no2cycle_summary.json"
    summary["ns_no2cycle_physical_amplitude_path"] = str(amp_path)
    summary["ns_no2cycle_cadence_comparison_path"] = str(cadence_path)
    summary["ns_net_residue_physical_lyapunov_path"] = str(lyap_path)
    _write_csv(amp_path, PHYSICAL_AMPLITUDE_FIELDS, amp_rows)
    _write_csv(cadence_path, CADENCE_FIELDS, amp_summary["cadence_rows"])
    _write_csv(lyap_path, LYAPUNOV_FIELDS, lyap_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint53_no2cycle_physical_amplitude_audit] wrote {amp_path}")
    print(f"[ns_sprint53_no2cycle_physical_amplitude_audit] wrote {cadence_path}")
    print(f"[ns_sprint53_no2cycle_physical_amplitude_audit] wrote {lyap_path}")
    print(f"[ns_sprint53_no2cycle_physical_amplitude_audit] wrote {summary_path}")
    print(
        "[ns_sprint53_no2cycle_physical_amplitude_audit] "
        f"route={summary['route_decision']} "
        f"physical_gate={summary['does_physical_cycle_gate_close']} "
        f"sigma={summary['sigma_physical_cycle_fit']}"
    )


if __name__ == "__main__":
    main()
