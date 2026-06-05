from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from test_sprint57_vessel_action_reconciliation_audit import _write_input, _write_truth


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_ROUTE_DECISIONS = {
    "RAW_ACTION_REASSIGNMENT_SOURCE_UNAVAILABLE",
    "RAW_ACTION_REASSIGNMENT_CONSERVATION_FAILED",
    "RAW_ACTION_REASSIGNMENT_NON_PARTITION_WINDOW_FAILED",
    "RAW_ACTION_REASSIGNMENT_FLAT_BLOCKED",
    "RAW_ACTION_REASSIGNMENT_SUMMABILITY_PROMISING_DIAGNOSTIC",
}


def _run_sprint59_raw_action_fixture(source: Path, out: Path) -> Path:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_sprint59_raw_packet_stretch_action_audit.py"),
            "--inputs",
            str(source),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )
    raw_csv = out / "ns_raw_packet_stretch_action.csv"
    assert raw_csv.exists()
    return raw_csv


def _find_euclidean_scheme(summary: dict[str, Any], scheme_rows: list[dict[str, str]]) -> dict[str, Any]:
    for row in scheme_rows:
        if row.get("assignment_scheme") == "euclidean_K_cell":
            return row
    for key in ("assignment_schemes", "schemes", "scheme_summaries"):
        for row in summary.get(key, []) or []:
            if row.get("assignment_scheme") == "euclidean_K_cell":
                return row
    raise AssertionError("missing euclidean_K_cell scheme row")


def _raw_positive_conservation_error(row: dict[str, Any]) -> float:
    for key, value in row.items():
        lowered = key.lower()
        if "conservation_error" in lowered and "raw" in lowered and ("positive" in lowered or "plus" in lowered):
            return abs(float(value))
    raise AssertionError(f"missing raw positive conservation error field in {sorted(row)}")


def test_sprint60_raw_action_reassignment_contract_fail_closed(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    _write_truth(truth)
    source = tmp_path / "sprint49_N4_seed0"
    _write_input(source, truth)
    raw_csv = _run_sprint59_raw_action_fixture(source, tmp_path / "sprint59")
    out = tmp_path / "sprint60"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_sprint60_raw_action_reassignment_audit.py"),
            "--inputs",
            str(source),
            "--raw-action-csv",
            str(raw_csv),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint60_raw_action_reassignment_summary.json").read_text(encoding="utf-8"))
    scheme_rows = list(csv.DictReader((out / "ns_raw_action_reassignment_by_scheme.csv").open(newline="", encoding="utf-8")))
    euclidean = _find_euclidean_scheme(summary, scheme_rows)

    assert summary["contract"] == "ns_sprint60_raw_action_reassignment_artifact"
    assert scheme_rows

    promotion_keys = [
        key
        for key, value in summary.items()
        if ("promotion" in key.lower() or "promoted" in key.lower()) and isinstance(value, bool)
    ]
    assert promotion_keys
    for key in promotion_keys:
        assert summary[key] is False
    assert summary["clay_promotion"] is False
    assert summary["navier_stokes_promotion"] is False

    tolerance = float(summary.get("conservation_tolerance", 1e-12))
    assert _raw_positive_conservation_error(euclidean) <= tolerance

    if "route_decision" in summary:
        assert summary["route_decision"] in ALLOWED_ROUTE_DECISIONS
    for route_key in ("route_decision", "route_status", "scheme_route_decision"):
        if route_key in euclidean:
            assert euclidean[route_key] in ALLOWED_ROUTE_DECISIONS
            assert euclidean[route_key] != "RAW_ACTION_REASSIGNMENT_CONSERVATION_FAILED"
