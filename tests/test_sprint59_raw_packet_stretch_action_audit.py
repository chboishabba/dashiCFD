from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ns_sprint56_packet_local_stretch_action_audit as sprint56
import ns_sprint57_vessel_action_reconciliation_audit as sprint57

from test_sprint57_vessel_action_reconciliation_audit import _write_input, _write_truth


def test_sprint59_raw_packet_stretch_action_uses_voxelwise_positive_part(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    _write_truth(truth)
    source = tmp_path / "sprint49_N4_seed0"
    _write_input(source, truth)
    out = tmp_path / "out"

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

    packet_rows = list(csv.DictReader((out / "ns_raw_packet_stretch_action.csv").open(newline="", encoding="utf-8")))
    by_k_rows = list(csv.DictReader((out / "ns_raw_packet_stretch_by_k.csv").open(newline="", encoding="utf-8")))
    summary = json.loads((out / "ns_sprint59_raw_packet_stretch_action_summary.json").read_text(encoding="utf-8"))
    assert summary["contract"] == "ns_sprint59_raw_packet_stretch_action_artifact"
    assert summary["clay_navier_stokes_promoted"] is False
    assert packet_rows
    assert by_k_rows

    loaded = sprint57._load_truth_run({"truth_path": str(truth)})
    assert loaded is not None
    omega, velocity, steps, truth_json = loaded
    n = int(omega.shape[1])
    L = float(truth_json["domain_length"])
    shell_map = sprint57.sprint54._build_shell_map(n, L)
    source_summary = json.loads((source / "ns_material_parent_summary.json").read_text(encoding="utf-8"))
    cell_map = sprint56._cell_map(n, int(source_summary["packet_grid"]))

    expected_total = 0.0
    for row in packet_rows:
        time = float(row["time"])
        idx = list(steps * float(truth_json["dt"])).index(time)
        dt = float(row["dt"])
        grad_u = sprint57.sprint54._build_velocity_gradient(velocity[idx], L)
        stretch = np.einsum("...i,...ij,...j->...", omega[idx], grad_u, omega[idx])
        mask = sprint57._packet_mask(shell_map, cell_map, row["packet_id"])
        assert mask is not None
        expected = float(np.sum(np.maximum(stretch[mask], 0.0))) * dt
        assert float(row["A_raw_positive"]) == expected
        assert float(row["weighted_A_raw_positive"]) == (2.0 ** (0.5 * float(row["K"]))) * expected
        expected_total += expected

    assert float(summary["A_raw_positive_total"]) == expected_total
