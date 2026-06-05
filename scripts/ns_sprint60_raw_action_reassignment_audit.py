#!/usr/bin/env python3
"""Sprint 60 raw-action shell reassignment audit.

Sprint 59 measured the vessel-additive packet source

    A_raw_plus(P) = integral_P max(omega dot S omega, 0) dx dt

on the reconstructed Sprint 49 Euclidean ``K_cell`` packet geometry.  Sprint 60
does not recompute the physical action.  It treats the Sprint 59 packet rows as
the canonical action ledger and redistributes that same action through explicit
shell assignment schemes:

* ``euclidean``: original Sprint 59 ``K``.
* ``smoothed``: a partition-of-unity split between ``K`` and ``K+1``.
* ``bt_p``: a provisional p-adic/BT-style cell valuation address.

Any improved shell decay is interpretable only if the reassignment conserves
raw positive/negative/net action per run/time and globally.  This remains a
diagnostic artifact; no physical bridge, stretch absorption, no-blowup, or
Clay/NS promotion is proved.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import ns_sprint53_no2cycle_physical_amplitude_audit as sprint53
import ns_sprint56_packet_local_stretch_action_audit as sprint56


EPS = 1e-30

LEDGER_FIELDS = [
    "assignment_scheme",
    "run",
    "time",
    "dt",
    "source_packet_id",
    "source_K",
    "source_cell",
    "assigned_K",
    "assigned_weight",
    "A_raw_positive_source",
    "A_raw_positive_assigned",
    "A_raw_negative_assigned",
    "A_raw_net_assigned",
    "packet_enstrophy_assigned",
    "packet_volume_assigned",
    "conservation_residual_positive_source",
    "assignment_status",
]

BY_K_FIELDS = [
    "assignment_scheme",
    "K",
    "packet_or_weight_count",
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

SCHEME_FIELDS = [
    "assignment_scheme",
    "source_row_count",
    "ledger_row_count",
    "A_raw_positive_source_total",
    "A_raw_positive_assigned_total",
    "A_raw_negative_source_total",
    "A_raw_negative_assigned_total",
    "A_raw_net_source_total",
    "A_raw_net_assigned_total",
    "max_run_time_conservation_error_positive",
    "max_run_time_raw_positive_conservation_error",
    "max_run_time_conservation_error_negative",
    "max_run_time_conservation_error_net",
    "global_conservation_error_positive",
    "global_raw_positive_conservation_error",
    "global_conservation_error_negative",
    "global_conservation_error_net",
    "unassigned_action_fraction",
    "overassigned_action_fraction",
    "partition_of_unity_failure_count",
    "sigma_raw_action_fit",
    "sigma_raw_action_mean_fit",
    "does_conservation_gate_pass",
    "does_sigma_gate_pass",
    "scheme_route_decision",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="*", default=[], help="optional Sprint 49 dirs recorded for manifest alignment")
    p.add_argument("--raw-action-csv", type=Path, required=True, help="Sprint 59 ns_raw_packet_stretch_action.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--schemes", "--assignment-schemes", nargs="+", default=["euclidean_K_cell", "smoothed_shell", "bt_ultrametric"])
    p.add_argument("--bt-prime", type=int, default=3)
    p.add_argument("--smoothed-forward-weight", type=float, default=0.5)
    p.add_argument("--conservation-tolerance", type=float, default=1e-12)
    p.add_argument("--sigma-threshold", type=float, default=0.5)
    return p.parse_args()


def _fmt(value: float | int | str | bool) -> str:
    return sprint53._fmt(value)


def _read_raw_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"run", "time", "dt", "K", "packet_id", "A_raw_positive", "A_raw_negative", "A_raw_net"}
    missing = sorted(required.difference(rows[0].keys() if rows else []))
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    return rows


def _valuation(n: int, p: int) -> int:
    if p <= 1:
        raise SystemExit("--bt-prime must be > 1")
    if n == 0:
        return 0
    out = 0
    value = abs(int(n))
    while value % p == 0:
        out += 1
        value //= p
    return out


def _source_k_cell(row: dict[str, str]) -> tuple[int, int | None]:
    k = int(row["K"])
    parsed = sprint56._parse_packet_id(str(row["packet_id"]))
    if parsed is None:
        return k, None
    return int(parsed[0]), int(parsed[1])


def _assignments(row: dict[str, str], scheme: str, args: argparse.Namespace) -> list[tuple[int, float, str]]:
    k, cell = _source_k_cell(row)
    scheme_key = {
        "euclidean": "euclidean_K_cell",
        "euclidean_k_cell": "euclidean_K_cell",
        "smoothed": "smoothed_shell",
        "bt_p": "bt_ultrametric",
        "bt": "bt_ultrametric",
    }.get(scheme, scheme)
    if scheme_key == "euclidean_K_cell":
        return [(k, 1.0, "assigned")]
    if scheme_key == "smoothed_shell":
        fwd = float(args.smoothed_forward_weight)
        back = 1.0 - fwd
        if fwd < 0.0 or fwd > 1.0:
            return [(k, 0.0, "partition_of_unity_failed")]
        if fwd == 0.0:
            return [(k, 1.0, "assigned")]
        if back == 0.0:
            return [(k + 1, 1.0, "assigned")]
        return [(k, back, "assigned"), (k + 1, fwd, "assigned")]
    if scheme_key == "bt_ultrametric":
        if cell is None:
            return [(k, 0.0, "unassigned_bad_packet_id")]
        # Provisional diagnostic address: cell+1 avoids every cell 0 being
        # infinitely divisible, while keeping a deterministic p-adic valuation.
        return [(k + _valuation(cell + 1, int(args.bt_prime)), 1.0, "assigned")]
    raise SystemExit(f"unknown assignment scheme: {scheme}")


def _float(row: dict[str, str], key: str) -> float:
    return float(row.get(key) or 0.0)


def _build_ledger(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for scheme in args.schemes:
        for row in rows:
            source_k, source_cell = _source_k_cell(row)
            pos = _float(row, "A_raw_positive")
            neg = _float(row, "A_raw_negative")
            net = _float(row, "A_raw_net")
            enst = _float(row, "packet_enstrophy")
            vol = _float(row, "packet_volume")
            assignment = _assignments(row, str(scheme), args)
            weight_sum = sum(weight for _k, weight, _status in assignment)
            residual = weight_sum - 1.0
            for assigned_k, weight, status in assignment:
                if abs(residual) > 1e-12 and status == "assigned":
                    status = "partition_of_unity_failed"
                ledger.append(
                    {
                        "assignment_scheme": str(scheme),
                        "run": row["run"],
                        "time": row["time"],
                        "dt": row["dt"],
                        "source_packet_id": row["packet_id"],
                        "source_K": str(source_k),
                        "source_cell": "" if source_cell is None else str(source_cell),
                        "assigned_K": str(int(assigned_k)),
                        "assigned_weight": _fmt(weight),
                        "A_raw_positive_source": _fmt(pos),
                        "A_raw_positive_assigned": _fmt(pos * weight),
                        "A_raw_negative_assigned": _fmt(neg * weight),
                        "A_raw_net_assigned": _fmt(net * weight),
                        "packet_enstrophy_assigned": _fmt(enst * weight),
                        "packet_volume_assigned": _fmt(vol * weight),
                        "conservation_residual_positive_source": _fmt(pos * residual),
                        "assignment_status": status,
                    }
                )
    return ledger


def _fit_by_k(ledger: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]]]:
    by_scheme_k: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "pos": 0.0, "neg": 0.0, "net": 0.0, "enst": 0.0, "vol": 0.0}
    )
    for row in ledger:
        key = (str(row["assignment_scheme"]), int(row["assigned_K"]))
        stats = by_scheme_k[key]
        stats["count"] += float(row["assigned_weight"])
        stats["pos"] += float(row["A_raw_positive_assigned"])
        stats["neg"] += float(row["A_raw_negative_assigned"])
        stats["net"] += float(row["A_raw_net_assigned"])
        stats["enst"] += float(row["packet_enstrophy_assigned"])
        stats["vol"] += float(row["packet_volume_assigned"])
    scheme_sigmas: dict[str, tuple[float, float]] = {}
    by_k_rows: list[dict[str, Any]] = []
    schemes = sorted({scheme for scheme, _k in by_scheme_k})
    for scheme in schemes:
        totals = {k: v["pos"] for (s, k), v in by_scheme_k.items() if s == scheme}
        means = {k: v["pos"] / max(v["count"], EPS) for (s, k), v in by_scheme_k.items() if s == scheme}
        sigma_total = sprint53._fit_sigma(totals)
        sigma_mean = sprint53._fit_sigma(means)
        scheme_sigmas[scheme] = (sigma_total, sigma_mean)
        for (s, k), stats in sorted(by_scheme_k.items()):
            if s != scheme:
                continue
            weighted = (2.0 ** (0.5 * float(k))) * stats["pos"]
            by_k_rows.append(
                {
                    "assignment_scheme": scheme,
                    "K": str(k),
                    "packet_or_weight_count": _fmt(stats["count"]),
                    "A_raw_positive_total": _fmt(stats["pos"]),
                    "A_raw_positive_mean": _fmt(stats["pos"] / max(stats["count"], EPS)),
                    "A_raw_negative_total": _fmt(stats["neg"]),
                    "A_raw_net_total": _fmt(stats["net"]),
                    "packet_enstrophy_total": _fmt(stats["enst"]),
                    "packet_volume_total": _fmt(stats["vol"]),
                    "weighted_A_raw_positive_total": _fmt(weighted),
                    "sigma_raw_action_fit": _fmt(sigma_total),
                    "sigma_raw_action_mean_fit": _fmt(sigma_mean),
                }
            )
    return by_k_rows, scheme_sigmas


def _source_totals(rows: list[dict[str, str]], key: str) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        totals[(str(row["run"]), str(row["time"]))] += _float(row, key)
    return totals


def _assigned_totals(ledger: list[dict[str, Any]], scheme: str, key: str) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in ledger:
        if str(row["assignment_scheme"]) == scheme:
            totals[(str(row["run"]), str(row["time"]))] += float(row[key])
    return totals


def _max_error(reference: dict[tuple[str, str], float], assigned: dict[tuple[str, str], float]) -> float:
    keys = set(reference) | set(assigned)
    if not keys:
        return 0.0
    return max(abs(assigned.get(key, 0.0) - reference.get(key, 0.0)) / (abs(reference.get(key, 0.0)) + EPS) for key in keys)


def _global_error(source: float, assigned: float) -> float:
    return abs(assigned - source) / (abs(source) + EPS)


def _scheme_route(row: dict[str, Any], sigma_threshold: float) -> str:
    if not bool(row["does_conservation_gate_pass"]):
        if int(row["partition_of_unity_failure_count"]) > 0:
            return "RAW_ACTION_REASSIGNMENT_NON_PARTITION_WINDOW_FAILED"
        return "RAW_ACTION_REASSIGNMENT_CONSERVATION_FAILED"
    if bool(row["does_sigma_gate_pass"]):
        return "RAW_ACTION_REASSIGNMENT_SUMMABILITY_PROMISING_DIAGNOSTIC"
    return "RAW_ACTION_REASSIGNMENT_FLAT_BLOCKED"


def _build_scheme_rows(
    source_rows: list[dict[str, str]],
    ledger: list[dict[str, Any]],
    scheme_sigmas: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    src_pos_rt = _source_totals(source_rows, "A_raw_positive")
    src_neg_rt = _source_totals(source_rows, "A_raw_negative")
    src_net_rt = _source_totals(source_rows, "A_raw_net")
    src_pos = sum(src_pos_rt.values())
    src_neg = sum(src_neg_rt.values())
    src_net = sum(src_net_rt.values())
    for scheme in args.schemes:
        scheme = str(scheme)
        rows = [row for row in ledger if str(row["assignment_scheme"]) == scheme]
        assigned_pos_rt = _assigned_totals(ledger, scheme, "A_raw_positive_assigned")
        assigned_neg_rt = _assigned_totals(ledger, scheme, "A_raw_negative_assigned")
        assigned_net_rt = _assigned_totals(ledger, scheme, "A_raw_net_assigned")
        assigned_pos = sum(assigned_pos_rt.values())
        assigned_neg = sum(assigned_neg_rt.values())
        assigned_net = sum(assigned_net_rt.values())
        partition_failures = sum(1 for row in rows if str(row["assignment_status"]) == "partition_of_unity_failed")
        unassigned = sum(float(row["A_raw_positive_source"]) for row in rows if str(row["assignment_status"]).startswith("unassigned"))
        overassigned = sum(max(float(row["conservation_residual_positive_source"]), 0.0) for row in rows)
        sigma_total, sigma_mean = scheme_sigmas.get(scheme, (0.0, 0.0))
        max_pos = _max_error(src_pos_rt, assigned_pos_rt)
        max_neg = _max_error(src_neg_rt, assigned_neg_rt)
        max_net = _max_error(src_net_rt, assigned_net_rt)
        glob_pos = _global_error(src_pos, assigned_pos)
        glob_neg = _global_error(src_neg, assigned_neg)
        glob_net = _global_error(src_net, assigned_net)
        conservation = (
            max(max_pos, max_neg, max_net, glob_pos, glob_neg, glob_net) <= float(args.conservation_tolerance)
            and partition_failures == 0
        )
        item: dict[str, Any] = {
            "assignment_scheme": scheme,
            "source_row_count": len(source_rows),
            "ledger_row_count": len(rows),
            "A_raw_positive_source_total": src_pos,
            "A_raw_positive_assigned_total": assigned_pos,
            "A_raw_negative_source_total": src_neg,
            "A_raw_negative_assigned_total": assigned_neg,
            "A_raw_net_source_total": src_net,
            "A_raw_net_assigned_total": assigned_net,
            "max_run_time_conservation_error_positive": max_pos,
            "max_run_time_raw_positive_conservation_error": max_pos,
            "max_run_time_conservation_error_negative": max_neg,
            "max_run_time_conservation_error_net": max_net,
            "global_conservation_error_positive": glob_pos,
            "global_raw_positive_conservation_error": glob_pos,
            "global_conservation_error_negative": glob_neg,
            "global_conservation_error_net": glob_net,
            "unassigned_action_fraction": unassigned / (abs(src_pos) + EPS),
            "overassigned_action_fraction": overassigned / (abs(src_pos) + EPS),
            "partition_of_unity_failure_count": partition_failures,
            "sigma_raw_action_fit": sigma_total,
            "sigma_raw_action_mean_fit": sigma_mean,
            "does_conservation_gate_pass": conservation,
            "does_sigma_gate_pass": sigma_total > float(args.sigma_threshold),
        }
        item["scheme_route_decision"] = _scheme_route(item, float(args.sigma_threshold))
        out.append(item)
    return out


def _summary_route(scheme_rows: list[dict[str, Any]]) -> str:
    if not scheme_rows:
        return "RAW_ACTION_REASSIGNMENT_SOURCE_UNAVAILABLE"
    if any(str(row["scheme_route_decision"]) == "RAW_ACTION_REASSIGNMENT_SUMMABILITY_PROMISING_DIAGNOSTIC" for row in scheme_rows):
        return "RAW_ACTION_REASSIGNMENT_SUMMABILITY_PROMISING_DIAGNOSTIC"
    if any(not bool(row["does_conservation_gate_pass"]) for row in scheme_rows):
        if any(int(row["partition_of_unity_failure_count"]) > 0 for row in scheme_rows):
            return "RAW_ACTION_REASSIGNMENT_NON_PARTITION_WINDOW_FAILED"
        return "RAW_ACTION_REASSIGNMENT_CONSERVATION_FAILED"
    return "RAW_ACTION_REASSIGNMENT_FLAT_BLOCKED"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key, "")) for key in fieldnames})


def main() -> None:
    args = _parse_args()
    source_rows = _read_raw_rows(args.raw_action_csv)
    ledger = _build_ledger(source_rows, args)
    by_k_rows, scheme_sigmas = _fit_by_k(ledger)
    scheme_rows = _build_scheme_rows(source_rows, ledger, scheme_sigmas, args)
    summary: dict[str, Any] = {
        "contract": "ns_sprint60_raw_action_reassignment_artifact",
        "diagnostic_mode": "sprint60_raw_action_shell_reassignment",
        "raw_action_csv": str(args.raw_action_csv),
        "inputs": [str(path) for path in args.inputs],
        "assignment_schemes": [str(s) for s in args.schemes],
        "bt_prime": int(args.bt_prime),
        "smoothed_forward_weight": float(args.smoothed_forward_weight),
        "conservation_tolerance": float(args.conservation_tolerance),
        "sigma_threshold": float(args.sigma_threshold),
        "source_row_count": len(source_rows),
        "ledger_row_count": len(ledger),
        "scheme_count": len(scheme_rows),
        "scheme_summaries": scheme_rows,
        "raw_action_reassignment_conservation_proved": False,
        "weighted_raw_action_summability_proved": False,
        "physical_bridge_proved": False,
        "stretch_absorption_proved": False,
        "no_finite_time_blowup_proved": False,
        "clay_promotion": False,
        "navier_stokes_promotion": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT60_RAW_ACTION_REASSIGNMENT_DIAGNOSTIC",
        "boundary": (
            "Sprint 60 redistributes the Sprint 59 raw additive packet action "
            "through candidate shell assignments. Improved sigma is diagnostic "
            "only and is rejected unless raw positive/negative/net action is "
            "conserved per run/time and globally."
        ),
    }
    summary["route_decision"] = _summary_route(scheme_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out_dir / "ns_raw_action_reassignment_ledger.csv"
    by_k_path = args.out_dir / "ns_raw_action_reassignment_by_k.csv"
    scheme_path = args.out_dir / "ns_raw_action_reassignment_scheme_summary.csv"
    scheme_alias_path = args.out_dir / "ns_raw_action_reassignment_by_scheme.csv"
    summary_path = args.out_dir / "ns_sprint60_raw_action_reassignment_summary.json"
    summary["ns_raw_action_reassignment_ledger_path"] = str(ledger_path)
    summary["ns_raw_action_reassignment_by_k_path"] = str(by_k_path)
    summary["ns_raw_action_reassignment_scheme_summary_path"] = str(scheme_path)
    summary["ns_raw_action_reassignment_by_scheme_path"] = str(scheme_alias_path)
    _write_csv(ledger_path, LEDGER_FIELDS, ledger)
    _write_csv(by_k_path, BY_K_FIELDS, by_k_rows)
    _write_csv(scheme_path, SCHEME_FIELDS, scheme_rows)
    _write_csv(scheme_alias_path, SCHEME_FIELDS, scheme_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint60_raw_action_reassignment_audit] wrote {ledger_path}")
    print(f"[ns_sprint60_raw_action_reassignment_audit] wrote {by_k_path}")
    print(f"[ns_sprint60_raw_action_reassignment_audit] wrote {scheme_path}")
    print(f"[ns_sprint60_raw_action_reassignment_audit] wrote {summary_path}")
    print(
        "[ns_sprint60_raw_action_reassignment_audit] "
        f"route={summary['route_decision']} promotion={summary['promotion_status']}"
    )


if __name__ == "__main__":
    main()
