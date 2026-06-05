from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_ROUTE_DECISIONS = {
    "CKN_UNIFORM_DECAY_SUPPORTED",
    "CKN_LOCALIZED_PERSISTENT_PLATEAU",
    "CKN_CONCENTRATION_CANDIDATE_FOUND",
    "CKN_PRESSURE_DOMINATED_ARTIFACT",
    "CKN_INCONCLUSIVE_NEEDS_HIGHER_N",
    "CKN_UNIFORMITY_SOURCE_UNAVAILABLE",
}


def _write_truth(path: Path, *, with_pressure: bool = True) -> None:
    velocity = np.zeros((3, 4, 4, 4, 3), dtype=np.float32)
    velocity[..., 0] = 0.05
    velocity[:, 2:, 2:, 2:, 0] = 2.0
    kwargs = {
        "velocity_snapshots": velocity,
        "steps": np.asarray([0, 1, 2], dtype=np.int64),
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
        pressure = np.zeros((3, 4, 4, 4), dtype=np.float32)
        pressure[:, 2:, 2:, 2:] = 0.5
        kwargs["pressure_snapshots"] = pressure
    np.savez(path, **kwargs)


def _write_sprint64_csv(path: Path) -> None:
    fieldnames = [
        "run",
        "N",
        "scale_cells",
        "time_index",
        "block_i",
        "block_j",
        "block_k",
        "local_critical_quantity",
        "overflow_state",
    ]
    rows = [
        {
            "run": "ns3d_N4_seed0_gpu_pressure",
            "N": "4",
            "scale_cells": "2",
            "time_index": "1",
            "block_i": "2",
            "block_j": "2",
            "block_k": "2",
            "local_critical_quantity": "10.0",
            "overflow_state": "ascended",
        },
        {
            "run": "ns3d_N4_seed0_gpu_pressure",
            "N": "4",
            "scale_cells": "2",
            "time_index": "1",
            "block_i": "0",
            "block_j": "0",
            "block_k": "0",
            "local_critical_quantity": "0.001",
            "overflow_state": "grounded",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_sprint67_ckn_uniformity_contract_with_sprint64_candidates(tmp_path: Path) -> None:
    truth = tmp_path / "ns3d_N4_seed0_gpu_pressure.npz"
    sprint64 = tmp_path / "ns_local_critical_concentration.csv"
    out = tmp_path / "out"
    _write_truth(truth, with_pressure=True)
    _write_sprint64_csv(sprint64)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint67_ckn_uniformity_audit.py",
            "--inputs",
            str(truth),
            "--sprint64-csv",
            str(sprint64),
            "--out-dir",
            str(out),
            "--r-cells",
            "1",
            "2",
            "4",
            "--epsilon-critical",
            "0.01",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint67_ckn_uniformity_summary.json").read_text(encoding="utf-8"))
    cylinders = list(csv.DictReader((out / "ns_sprint67_ckn_uniformity_by_cylinder.csv").open(newline="", encoding="utf-8")))
    clusters = list(csv.DictReader((out / "ns_sprint67_ckn_uniformity_by_cluster.csv").open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_sprint67_ckn_uniformity_audit_artifact"
    assert summary["route_decision"] in ALLOWED_ROUTE_DECISIONS
    assert summary["pressure_reconstruction_missing"] is False
    assert summary["cylinder_count"] >= 1
    assert summary["cluster_count"] >= 1
    assert summary["ckn_epsilon_regularity_applied"] is False
    assert summary["clay_navier_stokes_promoted"] is False
    assert cylinders
    assert clusters
    assert cylinders[0]["candidate_source"] == "sprint64_fixed_block_ascended"
    assert "log_slope_dlogC_dlogr" in cylinders[0]
    assert "route_label" in clusters[0]


def test_sprint67_ckn_uniformity_pressure_missing_fail_closed(tmp_path: Path) -> None:
    truth = tmp_path / "ns3d_N4_seed0_gpu.npz"
    out = tmp_path / "out_missing"
    _write_truth(truth, with_pressure=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint67_ckn_uniformity_audit.py",
            "--inputs",
            str(truth),
            "--out-dir",
            str(out),
            "--r-cells",
            "1",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint67_ckn_uniformity_summary.json").read_text(encoding="utf-8"))
    assert summary["route_decision"] == "CKN_PRESSURE_DOMINATED_ARTIFACT"
    assert summary["pressure_reconstruction_missing"] is True
    assert summary["cylinder_count"] == 0
    assert summary["clay_navier_stokes_promoted"] is False
