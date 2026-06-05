from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_ROUTE_DECISIONS = {
    "LOCAL_CRITICAL_CONCENTRATION_SOURCE_UNAVAILABLE",
    "LOCAL_CRITICAL_CONCENTRATION_NO_ROWS",
    "LOCAL_CRITICAL_CONCENTRATION_PRESSURE_RECONSTRUCTION_MISSING",
    "LOCAL_CRITICAL_CONCENTRATION_SUBCRITICAL_ON_AVAILABLE_DATA",
    "LOCAL_CRITICAL_CONCENTRATION_MIXED",
    "LOCAL_CRITICAL_CONCENTRATION_CRITICAL_BLOCKED",
}


def _write_truth(path: Path, *, with_pressure: bool) -> None:
    velocity = np.zeros((2, 4, 4, 4, 3), dtype=np.float32)
    velocity[:, :2, :2, :2, 0] = 0.05
    velocity[:, 2:, 2:, 2:, 0] = 2.0
    kwargs = {
        "velocity_snapshots": velocity,
        "steps": np.asarray([0, 1], dtype=np.int64),
        "meta_json": json.dumps(
            {
                "N": 4,
                "dt": 0.1,
                "domain_length": 1.0,
                "has_velocity_snapshots": True,
                "periodic": True,
            }
        ),
    }
    if with_pressure:
        pressure = np.zeros((2, 4, 4, 4), dtype=np.float32)
        pressure[:, 2:, 2:, 2:] = 0.5
        kwargs["pressure_snapshots"] = pressure
    np.savez(path, **kwargs)


def test_sprint64_local_critical_concentration_contract(tmp_path: Path) -> None:
    truth = tmp_path / "truth_with_pressure.npz"
    out = tmp_path / "out"
    _write_truth(truth, with_pressure=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint64_ckn_local_critical_concentration_audit.py",
            "--inputs",
            str(truth),
            "--out-dir",
            str(out),
            "--scales",
            "2",
            "--epsilon-critical",
            "0.01",
            "--plateau-fraction",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
    )

    summary_path = out / "ns_sprint64_local_critical_concentration_summary.json"
    rows_path = out / "ns_local_critical_concentration.csv"
    by_scale_path = out / "ns_local_critical_concentration_by_scale.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(rows_path.open(newline="", encoding="utf-8")))
    by_scale = list(csv.DictReader(by_scale_path.open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_sprint64_local_critical_concentration_artifact"
    assert summary["route_decision"] in ALLOWED_ROUTE_DECISIONS
    assert summary["route_decision"] == "LOCAL_CRITICAL_CONCENTRATION_MIXED"
    assert summary["pressure_reconstruction_missing"] is False
    assert summary["local_critical_concentration_proved"] is False
    assert summary["ckn_epsilon_regularity_applied"] is False
    assert summary["physical_bridge_proved"] is False
    assert summary["clay_navier_stokes_promoted"] is False
    assert rows
    assert by_scale
    assert "local_concentration_ratio" in rows[0]
    assert "criticality_route_label" in rows[0]
    assert "scale_cells" in rows[0]
    assert all(float(row["local_concentration_ratio"]) >= 0.0 for row in rows)
    assert any(row["criticality_route_label"] == "SUBCRITICAL_GROUNDED" for row in rows)
    assert any(row["criticality_route_label"] == "CRITICAL_ASCENDED" for row in rows)


def test_sprint64_routes_pressure_missing_for_current_truth_shape(tmp_path: Path) -> None:
    truth = tmp_path / "truth_no_pressure.npz"
    out = tmp_path / "out_missing"
    _write_truth(truth, with_pressure=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint64_ckn_local_critical_concentration_audit.py",
            "--inputs",
            str(truth),
            "--out-dir",
            str(out),
            "--scales",
            "2",
            "--epsilon-critical",
            "0.01",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint64_local_critical_concentration_summary.json").read_text(encoding="utf-8"))
    assert summary["route_decision"] == "LOCAL_CRITICAL_CONCENTRATION_PRESSURE_RECONSTRUCTION_MISSING"
    assert summary["pressure_reconstruction_missing"] is True
    assert summary["clay_navier_stokes_promoted"] is False
