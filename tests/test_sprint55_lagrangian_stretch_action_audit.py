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
        "K_parent": "4",
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
        "shell_delta": "0",
        "parent_confidence": "0.5",
        "parent_relation": "cross_shell_parent",
        "classification": "minus_to_plus",
    }
    row.update(overrides)
    return row


def _write_input(path: Path, table_rows: list[dict[str, str]]) -> None:
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
                "source_true_new": "0",
                "source_tracking_uncertain": "0",
                "source_cross_shell": "1",
                "source_low_shell_injection": "0",
                "source_total_material": "1",
                "weighted_true_new": "0",
                "weighted_tracking_uncertain": "0",
                "weighted_cross_shell": "4",
                "weighted_low_shell_injection": "0",
                "weighted_total_material": "4",
                "sigma_true_new_fit": "0",
                "sigma_tracking_uncertain_fit": "0.1",
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


def test_sprint55_lagrangian_action_contract_and_fail_closed_route(tmp_path: Path) -> None:
    source = tmp_path / "sprint49_N32_seed0"
    _write_input(
        source,
        [
            _row(time="0.1", child_packet_id="a", parent_packet_id="p", child_state="plus"),
            _row(time="0.2", child_packet_id="b", parent_packet_id="a", child_state="minus"),
            _row(time="0.3", child_packet_id="c", parent_packet_id="b", child_state="plus"),
        ],
    )
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_sprint55_lagrangian_stretch_action_audit.py"),
            "--inputs",
            str(source),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint55_lagrangian_action_summary.json").read_text(encoding="utf-8"))
    action_rows = list(csv.DictReader((out / "ns_lagrangian_stretch_action.csv").open(newline="", encoding="utf-8")))
    hysteresis_rows = list(csv.DictReader((out / "ns_packet_action_hysteresis.csv").open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_sprint55_lagrangian_action_artifact"
    assert summary["route_decision"] == "PACKET_MASK_JOIN_INSUFFICIENT"
    assert summary["direct_stretch_available_lineage_count"] == 0
    assert summary["packet_local_stretch_action_available"] is False
    assert summary["color_string_proxy_demoted"] is True
    assert summary["clay_promotion"] is False
    assert summary["navier_stokes_promotion"] is False
    assert action_rows
    assert action_rows[0]["lagrangian_label"] == "green"
    assert action_rows[0]["direct_stretch_amplitude"] == "unavailable"
    assert hysteresis_rows
    assert hysteresis_rows[0]["hysteresis_color_state"] == "green"
