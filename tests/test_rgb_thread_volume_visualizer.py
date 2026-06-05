from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
import shutil

import numpy as np

from test_sprint57_vessel_action_reconciliation_audit import TABLE_FIELDS, SUMMARY_FIELDS, _row


ROOT = Path(__file__).resolve().parents[1]


def _write_truth(path: Path) -> None:
    n = 4
    omega = np.zeros((1, n, n, n, 3), dtype=np.float64)
    omega[..., 0] = 1.0
    np.savez(
        path,
        omega_snapshots=omega,
        steps=np.asarray([1], dtype=np.int64),
        meta_json=json.dumps({"dt": 0.1, "save_every": 1, "N": n, "domain_length": 2.0 * np.pi}),
    )


def _write_input(path: Path, truth: Path) -> None:
    path.mkdir()
    rows = [
        _row(time="0.1", K_child="1", child_packet_id="K1_cell0", child_state="plus"),
        _row(time="0.1", K_child="1", child_packet_id="K1_cell1", child_state="minus"),
        _row(time="0.1", K_child="1", child_packet_id="K1_cell2", child_state="zero"),
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


def _write_multitime_input(path: Path, truth: Path) -> None:
    path.mkdir()
    rows = [
        _row(time="0.1", K_child="1", child_packet_id="K1_cell0", child_state="plus"),
        _row(time="0.1", K_child="1", child_packet_id="K1_cell1", child_state="minus"),
        _row(time="0.2", K_child="1", child_packet_id="K1_cell0", child_state="minus"),
        _row(time="0.2", K_child="1", child_packet_id="K1_cell2", child_state="zero"),
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


def test_rgb_thread_volume_visualizer_outputs_png_and_nan_volume(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    _write_truth(truth)
    source = tmp_path / "sprint49_N4_seed0"
    _write_input(source, truth)
    out = tmp_path / "viz"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_rgb_thread_volume_visualizer.py"),
            "--input",
            str(source),
            "--out-dir",
            str(out),
            "--time",
            "0.1",
            "--alpha",
            "0.3",
            "--max-points",
            "1000",
        ],
        cwd=ROOT,
        check=True,
    )

    summary_path = next(out.glob("*_summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["contract"] == "ns_rgb_thread_volume_visualization"
    assert summary["alpha"] == 0.3
    assert summary["nil_nan_transparent"] is True
    assert Path(summary["scatter_png"]).exists()
    assert Path(summary["projections_png"]).exists()
    with np.load(summary["volume_npz"], allow_pickle=False) as data:
        labels = data["labels"]
    assert labels.shape == (4, 4, 4)
    assert np.count_nonzero(~np.isfinite(labels)) > 0


def test_rgb_thread_volume_visualizer_animates_all_times(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    _write_truth(truth)
    source = tmp_path / "sprint49_N4_seed0"
    _write_multitime_input(source, truth)
    out = tmp_path / "viz"
    fmt = "both" if shutil.which("ffmpeg") else "gif"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_rgb_thread_volume_visualizer.py"),
            "--input",
            str(source),
            "--out-dir",
            str(out),
            "--all-times",
            "--animation-format",
            fmt,
            "--fps",
            "2",
            "--max-points",
            "1000",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads(next(out.glob("*_animation_summary.json")).read_text(encoding="utf-8"))
    assert summary["frame_count"] == 2
    assert Path(summary["gif"]).exists()
    if fmt == "both":
        assert Path(summary["webm"]).exists()
    assert len(list((out / f"{source.name}_child_state_frames").glob("frame_*.png"))) == 2


def test_rgb_thread_volume_visualizer_raw_action_trits(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    _write_truth(truth)
    source = tmp_path / "sprint49_N4_seed0"
    _write_input(source, truth)
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["packet_id", "t_end", "A_raw_net"])
        writer.writeheader()
        writer.writerow({"packet_id": "K1_cell0", "t_end": "0.1", "A_raw_net": "1"})
        writer.writerow({"packet_id": "K1_cell1", "t_end": "0.1", "A_raw_net": "-1"})
        writer.writerow({"packet_id": "K1_cell2", "t_end": "0.1", "A_raw_net": "0"})
    out = tmp_path / "viz"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_rgb_thread_volume_visualizer.py"),
            "--input",
            str(source),
            "--out-dir",
            str(out),
            "--time",
            "0.1",
            "--trit-source",
            "raw_action",
            "--raw-action-csv",
            str(raw),
            "--max-points",
            "1000",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads(next(out.glob("*raw_action*_summary.json")).read_text(encoding="utf-8"))
    assert summary["trit_source"] == "raw_action"
    assert summary["counts"]["plus"] > 0
    assert summary["counts"]["minus"] > 0
    assert summary["counts"]["zero"] > 0
