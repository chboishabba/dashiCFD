from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


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
        "dt": "0.01",
        "K_parent": "1",
        "K_child": "4",
        "child_packet_id": "child",
        "parent_packet_id": "parent",
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
        "shell_delta": "3",
        "parent_confidence": "0.5",
        "parent_relation": "cross_shell_parent",
        "classification": "minus_to_plus",
    }
    row.update(overrides)
    return row


def _write_input(path: Path, table_rows: list[dict[str, str]], weighted_true: str = "0") -> None:
    path.mkdir()
    with (path / "ns_material_parent_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(table_rows)
    with (path / "ns_material_parent_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "time": "0.1",
                "K_child": "4",
                "M_plus_plus_material": "0.4",
                "source_true_new": weighted_true,
                "source_tracking_uncertain": "0",
                "source_cross_shell": "1",
                "source_low_shell_injection": "0",
                "source_total_material": "1",
                "weighted_true_new": weighted_true,
                "weighted_tracking_uncertain": "0",
                "weighted_cross_shell": "4",
                "weighted_low_shell_injection": "0",
                "weighted_total_material": "4",
                "sigma_true_new_fit": "0",
                "sigma_tracking_uncertain_fit": "0",
                "sigma_cross_shell_fit": "0.1",
                "sigma_low_shell_fit": "0",
                "sigma_total_material_fit": "0.1",
                "route_status": "ADJACENT_PACKET_THEOREM_INSUFFICIENT",
            }
        )
    (path / "ns_material_parent_summary.json").write_text(
        json.dumps({"contract": "ns_material_parent_artifact", "table_row_count": len(table_rows), "summary_row_count": 1}),
        encoding="utf-8",
    )


def _run(source: Path, out: Path, *extra: str) -> dict[str, object]:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_sprint53_no2cycle_physical_amplitude_audit.py"),
            "--inputs",
            str(source),
            "--out-dir",
            str(out),
            *extra,
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads((out / "ns_sprint53_no2cycle_summary.json").read_text(encoding="utf-8"))


def test_sprint53_small_physical_cycle_route_alive(tmp_path: Path) -> None:
    source = tmp_path / "sprint49"
    rows = [
        _row(time="0.1", K_child="4", child_packet_id="a", parent_packet_id="p", parent_state="minus", child_state="plus", credited_mass="10"),
        _row(time="0.2", K_parent="4", K_child="4", child_packet_id="b", parent_packet_id="a", parent_state="plus", child_state="minus", credited_mass="8"),
        _row(time="0.3", K_child="4", child_packet_id="c", parent_packet_id="b", parent_state="plus", child_state="plus", credited_mass="1000"),
        _row(time="0.3", K_child="4", child_packet_id="d", parent_packet_id="q", parent_state="minus", child_state="minus", credited_mass="1007.5"),
    ]
    _write_input(source, rows)

    summary = _run(source, tmp_path / "out", "--physical-amplitude-small-majority", "0.5")
    amp_rows = list(csv.DictReader((tmp_path / "out" / "ns_no2cycle_physical_amplitude.csv").open(newline="", encoding="utf-8")))

    assert summary["does_material_source_gate_close"] is True
    assert summary["does_physical_cycle_gate_close"] is True
    assert summary["route_decision"] == "NS_SOURCE_BUDGET_PHYSICAL_NO2CYCLE_ROUTE_ALIVE_DIAGNOSTIC"
    assert amp_rows[0]["physical_amplitude_small"] == "true"
    assert summary["clay_promotion"] is False


def test_sprint53_large_physical_cycle_blocks(tmp_path: Path) -> None:
    source = tmp_path / "sprint49"
    rows = [
        _row(time="0.1", K_child="4", child_packet_id="a", parent_packet_id="p", parent_state="minus", child_state="plus", credited_mass="10"),
        _row(time="0.2", K_parent="4", K_child="4", child_packet_id="b", parent_packet_id="a", parent_state="plus", child_state="minus", credited_mass="20"),
        _row(time="0.3", K_child="4", child_packet_id="c", parent_packet_id="b", parent_state="minus", child_state="plus", credited_mass="200"),
    ]
    _write_input(source, rows)

    summary = _run(source, tmp_path / "out", "--physical-amplitude-small-fraction", "0.01")

    assert summary["does_material_source_gate_close"] is True
    assert summary["does_physical_cycle_gate_close"] is False
    assert summary["route_decision"] == "MATERIAL_SOURCE_GATE_CLOSED_PHYSICAL_NO2CYCLE_AMPLITUDE_BLOCKED"
    assert summary["physical_large_cycle_count"] == 1
    assert summary["navier_stokes_promotion"] is False
