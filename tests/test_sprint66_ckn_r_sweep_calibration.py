from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_ROUTE_DECISIONS = {
    "CKN_R_SWEEP_SOURCE_UNAVAILABLE",
    "CKN_R_SWEEP_NO_HOTSPOTS",
    "CKN_R_SWEEP_PRESSURE_RECONSTRUCTION_MISSING",
    "CKN_R_SWEEP_SUBCRITICAL_ON_SAMPLED_HOTSPOTS",
    "CKN_R_SWEEP_DECAYS_UNDER_ZOOM",
    "CKN_R_SWEEP_MIXED",
    "CKN_R_SWEEP_CRITICAL_BLOCKED",
}


def _write_truth(path: Path, *, with_pressure: bool = True) -> None:
    velocity = np.zeros((2, 4, 4, 4, 3), dtype=np.float32)
    velocity[..., 0] = 0.05
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
                "has_pressure_snapshots": with_pressure,
                "periodic": True,
            }
        ),
    }
    if with_pressure:
        pressure = np.zeros((2, 4, 4, 4), dtype=np.float32)
        pressure[:, 2:, 2:, 2:] = 0.5
        kwargs["pressure_snapshots"] = pressure
    np.savez(path, **kwargs)


def test_sprint66_ckn_r_sweep_pressure_present_contract(tmp_path: Path) -> None:
    truth = tmp_path / "ns3d_N4_seed0_gpu_pressure.npz"
    out = tmp_path / "out"
    _write_truth(truth, with_pressure=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint66_ckn_r_sweep_calibration.py",
            "--inputs",
            str(truth),
            "--out-dir",
            str(out),
            "--r-cells",
            "1",
            "2",
            "4",
            "--epsilon-grid",
            "0.01",
            "1.0",
            "--top-hotspots",
            "3",
            "--plateau-fraction",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint66_ckn_r_sweep_calibration_summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((out / "ns_ckn_r_sweep_calibration.csv").open(newline="", encoding="utf-8")))
    by_r = list(csv.DictReader((out / "ns_ckn_r_sweep_by_radius.csv").open(newline="", encoding="utf-8")))
    hotspots = list(csv.DictReader((out / "ns_ckn_r_sweep_hotspots.csv").open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_sprint66_ckn_r_sweep_calibration_artifact"
    assert summary["route_decision"] in ALLOWED_ROUTE_DECISIONS
    assert summary["pressure_reconstruction_missing"] is False
    assert summary["pressure_available_all"] is True
    assert summary["ckn_epsilon_regularity_applied"] is False
    assert summary["local_critical_concentration_proved"] is False
    assert summary["clay_navier_stokes_promoted"] is False
    assert rows
    assert by_r
    assert hotspots
    required = {
        "r_cells",
        "r_physical",
        "epsilon_critical",
        "C_total",
        "C_epsilon_ratio",
        "overflow_state",
        "criticality_route_label",
    }
    assert required.issubset(rows[0].keys())
    assert any(row["overflow_state"] == "ascended" and row["epsilon_critical"] == "0.01" for row in rows)
    assert any(row["trend_label"] in {"CKN_DECAYS_UNDER_ZOOM", "CKN_FLAT_MARGINAL", "CKN_CONCENTRATES_UNDER_ZOOM"} for row in hotspots)


def test_sprint66_routes_pressure_missing_fail_closed(tmp_path: Path) -> None:
    truth = tmp_path / "ns3d_N4_seed0_gpu.npz"
    out = tmp_path / "out_missing"
    _write_truth(truth, with_pressure=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint66_ckn_r_sweep_calibration.py",
            "--inputs",
            str(truth),
            "--out-dir",
            str(out),
            "--r-cells",
            "1",
            "2",
            "--epsilon-grid",
            "0.01",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint66_ckn_r_sweep_calibration_summary.json").read_text(encoding="utf-8"))
    assert summary["route_decision"] == "CKN_R_SWEEP_PRESSURE_RECONSTRUCTION_MISSING"
    assert summary["pressure_reconstruction_missing"] is True
    assert summary["row_count"] == 0
    assert summary["ckn_epsilon_regularity_applied"] is False
    assert summary["clay_navier_stokes_promoted"] is False


def test_sprint66_routes_no_hotspots_for_invalid_r(tmp_path: Path) -> None:
    truth = tmp_path / "ns3d_N4_seed0_gpu_pressure.npz"
    out = tmp_path / "out_no_rows"
    _write_truth(truth, with_pressure=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint66_ckn_r_sweep_calibration.py",
            "--inputs",
            str(truth),
            "--out-dir",
            str(out),
            "--r-cells",
            "8",
            "16",
            "--epsilon-grid",
            "0.01",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint66_ckn_r_sweep_calibration_summary.json").read_text(encoding="utf-8"))
    assert summary["route_decision"] == "CKN_R_SWEEP_NO_HOTSPOTS"
    assert summary["row_count"] == 0
    assert summary["clay_navier_stokes_promoted"] is False


def test_sprint66_epsilon_sweep_is_monotone(tmp_path: Path) -> None:
    truth = tmp_path / "ns3d_N4_seed0_gpu_pressure.npz"
    out = tmp_path / "out_eps"
    _write_truth(truth, with_pressure=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint66_ckn_r_sweep_calibration.py",
            "--inputs",
            str(truth),
            "--out-dir",
            str(out),
            "--r-cells",
            "2",
            "--epsilon-grid",
            "0.01",
            "1.0",
            "--top-hotspots",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )

    rows = list(csv.DictReader((out / "ns_ckn_r_sweep_by_radius.csv").open(newline="", encoding="utf-8")))
    by_eps = {float(row["epsilon_critical"]): row for row in rows}
    assert 0.01 in by_eps
    assert 1.0 in by_eps
    assert float(by_eps[0.01]["max_C_total"]) == float(by_eps[1.0]["max_C_total"])
    assert float(by_eps[0.01]["max_C_epsilon_ratio"]) > float(by_eps[1.0]["max_C_epsilon_ratio"])
    assert float(by_eps[0.01]["ascended_fraction"]) >= float(by_eps[1.0]["ascended_fraction"])
