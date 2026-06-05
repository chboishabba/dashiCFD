from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

TABLE_FIELDS = [
    "time",
    "dt",
    "K_parent",
    "K_child",
    "child_packet_id",
    "parent_packet_id",
    "child_state",
    "parent_state",
    "child_mass",
    "parent_mass",
    "credited_mass",
    "source_true_new",
    "source_tracking_uncertain",
    "source_cross_shell",
    "source_low_shell_injection",
    "advected_overlap",
    "centroid_distance",
    "direction_cosine",
    "shell_delta",
    "parent_confidence",
    "parent_relation",
    "classification",
]

SUMMARY_FIELDS = [
    "time",
    "K_child",
    "M_plus_plus_material",
    "source_true_new",
    "source_tracking_uncertain",
    "source_cross_shell",
    "source_low_shell_injection",
    "source_total_material",
    "weighted_true_new",
    "weighted_tracking_uncertain",
    "weighted_cross_shell",
    "weighted_low_shell_injection",
    "weighted_total_material",
    "sigma_true_new_fit",
    "sigma_tracking_uncertain_fit",
    "sigma_cross_shell_fit",
    "sigma_low_shell_fit",
    "sigma_total_material_fit",
    "route_status",
]


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "time": "0.1",
        "dt": "0.1",
        "K_parent": "1",
        "K_child": "1",
        "child_packet_id": "K1_cell0",
        "parent_packet_id": "K1_cell0",
        "child_state": "plus",
        "parent_state": "minus",
        "child_mass": "100",
        "parent_mass": "100",
        "credited_mass": "1",
        "source_true_new": "0",
        "source_tracking_uncertain": "0",
        "source_cross_shell": "1",
        "source_low_shell_injection": "0",
        "advected_overlap": "0.5",
        "centroid_distance": "1",
        "direction_cosine": "0",
        "shell_delta": "0",
        "parent_confidence": "0.5",
        "parent_relation": "cross_shell_parent",
        "classification": "minus_to_plus",
    }
    row.update(overrides)
    return row


def _write_truth(path: Path) -> None:
    n = 4
    L = 2.0 * np.pi
    x = np.arange(n, dtype=np.float64) * (L / n)
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    velocity = np.zeros((2, n, n, n, 3), dtype=np.float64)
    omega = np.zeros_like(velocity)
    velocity[:, ..., 0] = np.sin(xx)
    omega[0, ..., 0] = 1.0
    omega[1, ..., 0] = 1.0
    omega[1, ..., 1] = 0.25
    np.savez(
        path,
        omega_snapshots=omega,
        velocity_snapshots=velocity,
        steps=np.asarray([1, 2], dtype=np.int64),
        meta_json=json.dumps({"dt": 0.1, "save_every": 1, "N": n, "domain_length": L}),
    )


def _write_input(path: Path, truth: Path) -> None:
    path.mkdir()
    rows = [
        _row(time="0.1", child_packet_id="K1_cell0", parent_packet_id="K1_cell0"),
        _row(time="0.2", child_packet_id="K1_cell1", parent_packet_id="K1_cell0"),
    ]
    with (path / "ns_material_parent_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (path / "ns_material_parent_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "time": "0.1",
                "K_child": "1",
                "M_plus_plus_material": "0.4",
                "source_true_new": "0",
                "source_tracking_uncertain": "0",
                "source_cross_shell": "1",
                "source_low_shell_injection": "0",
                "source_total_material": "1",
                "weighted_true_new": "0",
                "weighted_tracking_uncertain": "0",
                "weighted_cross_shell": "1.414",
                "weighted_low_shell_injection": "0",
                "weighted_total_material": "1.414",
                "sigma_true_new_fit": "0",
                "sigma_tracking_uncertain_fit": "0.1",
                "sigma_cross_shell_fit": "0.1",
                "sigma_low_shell_fit": "0",
                "sigma_total_material_fit": "0.1",
                "route_status": "ADJACENT_PACKET_THEOREM_INSUFFICIENT",
            }
        )
    (path / "ns_material_parent_summary.json").write_text(
        json.dumps(
            {
                "contract": "ns_material_parent_artifact",
                "table_row_count": len(rows),
                "summary_row_count": 1,
                "packet_grid": 2,
                "source_truth": str(truth),
            }
        ),
        encoding="utf-8",
    )


def test_sprint57_reconciles_packet_and_global_action(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    _write_truth(truth)
    source = tmp_path / "sprint49_N4_seed0"
    _write_input(source, truth)
    out = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_sprint57_vessel_action_reconciliation_audit.py"),
            "--inputs",
            str(source),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint57_vessel_action_reconciliation_summary.json").read_text(encoding="utf-8"))
    time_rows = list(csv.DictReader((out / "ns_vessel_action_reconciliation_by_time.csv").open(newline="", encoding="utf-8")))
    summary_rows = list(csv.DictReader((out / "ns_vessel_action_reconciliation_summary.csv").open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_sprint57_vessel_action_reconciliation_artifact"
    assert summary["diagnostic_mode"] == "sprint57_global_vessel_action_reconciliation"
    assert summary["time_window_count"] == 2
    assert summary["packet_local_action_row_count"] >= 1
    assert summary["packet_action_reconstructs_global_stretch_proved"] is False
    assert summary["weighted_packet_local_action_summability_proved"] is False
    assert summary["clay_promotion"] is False
    assert summary["navier_stokes_promotion"] is False
    assert time_rows
    assert summary_rows
    assert float(summary["global_raw_positive_stretch_action_total"]) >= 0.0
    assert float(summary["packet_raw_positive_stretch_action_total"]) >= 0.0
    assert "epsilon_raw_positive_vs_covered" in summary
