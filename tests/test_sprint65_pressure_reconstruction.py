from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _write_truth(path: Path) -> None:
    n = 8
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    z, y, xg = np.meshgrid(x, x, x, indexing="ij")
    velocity = np.zeros((1, n, n, n, 3), dtype=np.float32)
    velocity[0, ..., 0] = np.sin(y)
    velocity[0, ..., 1] = np.sin(xg)
    np.savez(
        path,
        velocity_snapshots=velocity,
        omega_snapshots=np.zeros_like(velocity),
        steps=np.asarray([0], dtype=np.int64),
        meta_json=json.dumps({"N": n, "dt": 0.1, "domain_length": 2.0 * np.pi, "periodic": True}),
    )


def test_sprint65_pressure_reconstruction_adds_pressure_and_manifest(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    out = tmp_path / "pressure"
    _write_truth(truth)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint65_pressure_reconstruction.py",
            "--inputs",
            str(truth),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    augmented = out / "truth_pressure.npz"
    manifest = json.loads((out / "ns_sprint65_pressure_reconstruction_manifest.json").read_text(encoding="utf-8"))
    z = np.load(augmented)

    assert manifest["contract"] == "ns_sprint65_pressure_reconstruction_manifest"
    assert manifest["artifact_count"] == 1
    assert manifest["pressure_reconstruction_proved"] is False
    assert manifest["ckn_epsilon_regularity_applied"] is False
    assert manifest["clay_navier_stokes_promoted"] is False
    assert "pressure_snapshots" in z.files
    assert z["pressure_snapshots"].shape == (1, 8, 8, 8)
    assert abs(float(z["pressure_snapshots"][0].mean())) < 1e-6
    assert manifest["max_poisson_relative_residual_rms"] < 1e-10


def test_sprint64_accepts_sprint65_pressure_artifact(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    pressure_out = tmp_path / "pressure"
    audit_out = tmp_path / "audit"
    _write_truth(truth)

    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint65_pressure_reconstruction.py",
            "--inputs",
            str(truth),
            "--out-dir",
            str(pressure_out),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/ns_sprint64_ckn_local_critical_concentration_audit.py",
            "--inputs",
            str(pressure_out / "truth_pressure.npz"),
            "--out-dir",
            str(audit_out),
            "--scales",
            "4",
            "--epsilon-critical",
            "0.01",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((audit_out / "ns_sprint64_local_critical_concentration_summary.json").read_text(encoding="utf-8"))
    assert summary["pressure_reconstruction_missing"] is False
    assert summary["route_decision"] != "LOCAL_CRITICAL_CONCENTRATION_PRESSURE_RECONSTRUCTION_MISSING"
    assert summary["clay_navier_stokes_promoted"] is False
