from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from test_sprint57_vessel_action_reconciliation_audit import TABLE_FIELDS, SUMMARY_FIELDS, _row


ROOT = Path(__file__).resolve().parents[1]


def _write_truth(path: Path) -> None:
    n = 4
    L = 2.0 * np.pi
    x = np.arange(n, dtype=np.float64) * (L / n)
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    velocity = np.zeros((2, n, n, n, 3), dtype=np.float64)
    omega = np.zeros_like(velocity)
    velocity[:, ..., 0] = np.sin(xx)
    omega[:, ..., 0] = 1.0
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
        _row(time="0.1", dt="0.1", child_packet_id="K1_cell0", parent_packet_id="K1_cell0"),
        _row(time="0.2", dt="0.1", child_packet_id="K1_cell1", parent_packet_id="K1_cell0"),
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


def test_sprint58_reports_normalized_action_inflation_contract(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    _write_truth(truth)
    source = tmp_path / "sprint49_N4_seed0"
    _write_input(source, truth)
    out = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_sprint58_normalized_action_inflation_audit.py"),
            "--inputs",
            str(source),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint58_normalized_action_inflation_summary.json").read_text(encoding="utf-8"))
    packet_rows = list(csv.DictReader((out / "ns_normalized_action_inflation_packets.csv").open(newline="", encoding="utf-8")))
    time_rows = list(csv.DictReader((out / "ns_normalized_action_inflation_by_time.csv").open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_sprint58_normalized_action_inflation_artifact"
    assert summary["diagnostic_mode"] == "sprint58_normalized_packet_action_inflation"
    assert summary["time_window_count"] == 2
    assert summary["packet_inflation_row_count"] >= 1
    assert "sum_ratios_over_ratio_of_sums_covered" in summary
    assert summary["normalized_action_additivity_proved"] is False
    assert summary["denominator_inflation_theorem_proved"] is False
    assert summary["clay_promotion"] is False
    assert summary["navier_stokes_promotion"] is False
    assert packet_rows
    assert time_rows
