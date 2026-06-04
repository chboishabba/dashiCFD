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


def _write_sprint49_dir(path: Path) -> None:
    path.mkdir()
    rows = [
        {
            "time": "0.1",
            "dt": "0.01",
            "K_parent": "2",
            "K_child": "2",
            "child_packet_id": "same_plus",
            "parent_packet_id": "same_plus",
            "child_state": "plus",
            "parent_state": "plus",
            "child_mass": "7",
            "parent_mass": "8",
            "credited_mass": "3",
            "source_true_new": "0",
            "source_tracking_uncertain": "0",
            "source_cross_shell": "0",
            "source_low_shell_injection": "0",
            "advected_overlap": "1",
            "centroid_distance": "0",
            "direction_cosine": "1",
            "shell_delta": "0",
            "parent_confidence": "1",
            "parent_relation": "advected_parent",
            "classification": "plus_to_plus",
        },
        {
            "time": "0.1",
            "dt": "0.01",
            "K_parent": "1",
            "K_child": "4",
            "child_packet_id": "cross_zero_plus",
            "parent_packet_id": "cross_zero",
            "child_state": "plus",
            "parent_state": "zero",
            "child_mass": "11",
            "parent_mass": "12",
            "credited_mass": "5",
            "source_true_new": "0",
            "source_tracking_uncertain": "0",
            "source_cross_shell": "5",
            "source_low_shell_injection": "0",
            "advected_overlap": "0.2",
            "centroid_distance": "9",
            "direction_cosine": "0",
            "shell_delta": "3",
            "parent_confidence": "0.4",
            "parent_relation": "advected_parent",
            "classification": "plus_to_plus",
        },
        {
            "time": "0.1",
            "dt": "0.01",
            "K_parent": "3",
            "K_child": "4",
            "child_packet_id": "adj_minus_plus",
            "parent_packet_id": "adj_minus",
            "child_state": "plus",
            "parent_state": "minus",
            "child_mass": "13",
            "parent_mass": "14",
            "credited_mass": "2",
            "source_true_new": "0",
            "source_tracking_uncertain": "0",
            "source_cross_shell": "0",
            "source_low_shell_injection": "0",
            "advected_overlap": "0.8",
            "centroid_distance": "1",
            "direction_cosine": "0",
            "shell_delta": "1",
            "parent_confidence": "0.7",
            "parent_relation": "split_parent",
            "classification": "minus_to_plus",
        },
        {
            "time": "0.1",
            "dt": "0.01",
            "K_parent": "0",
            "K_child": "4",
            "child_packet_id": "low_minus_plus",
            "parent_packet_id": "low_minus",
            "child_state": "plus",
            "parent_state": "minus",
            "child_mass": "17",
            "parent_mass": "19",
            "credited_mass": "4",
            "source_true_new": "0",
            "source_tracking_uncertain": "0",
            "source_cross_shell": "0",
            "source_low_shell_injection": "4",
            "advected_overlap": "0.1",
            "centroid_distance": "4",
            "direction_cosine": "0",
            "shell_delta": "4",
            "parent_confidence": "0.2",
            "parent_relation": "low_shell_parent",
            "classification": "cross_shell_plus",
        },
    ]
    with (path / "ns_material_parent_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (path / "ns_material_parent_summary.json").write_text(
        json.dumps(
            {
                "contract": "ns_material_parent_artifact",
                "table_row_count": len(rows),
                "summary_row_count": 1,
                "material_parent_route_status": "ADJACENT_PACKET_THEOREM_INSUFFICIENT",
            }
        ),
        encoding="utf-8",
    )


def test_ternary_cross_shell_matrix_contract_and_parent_relation_regression(tmp_path: Path) -> None:
    source = tmp_path / "sprint49"
    _write_sprint49_dir(source)
    out = tmp_path / "sprint50"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_ternary_cross_shell_matrix.py"),
            "--inputs",
            str(source),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    matrix = list(csv.DictReader((out / "ns_full_ternary_transition_matrix.csv").open(newline="", encoding="utf-8")))
    decomp = list(csv.DictReader((out / "ns_cross_shell_source_decomposition.csv").open(newline="", encoding="utf-8")))
    summary = json.loads((out / "ns_ternary_cross_shell_summary.json").read_text(encoding="utf-8"))

    def matrix_mass(kind: str, child: str, parent: str) -> float:
        [row] = [r for r in matrix if r["source_kind"] == kind and r["child_state"] == child and r["parent_state"] == parent]
        return float(row["transition_mass"])

    assert matrix_mass("all", "plus", "plus") == 3.0
    assert matrix_mass("cross_shell", "plus", "zero") == 5.0
    assert matrix_mass("adjacent_shell", "plus", "minus") == 2.0
    assert matrix_mass("low_shell_injection", "plus", "minus") == 4.0

    cross_zero = [
        r
        for r in decomp
        if r["source_kind"] == "cross_shell" and r["parent_state"] == "zero" and r["child_state"] == "plus"
    ][0]
    assert float(cross_zero["weighted_child_mass"]) == 20.0
    assert float(cross_zero["bt_distance_proxy_mass_weighted_mean"]) == 3.0

    assert summary["contract"] == "ns_ternary_cross_shell_artifact"
    assert summary["weighted_cross_plus_from_zero"] == 20.0
    assert summary["weighted_cross_plus_from_minus"] == 0.0
    assert summary["dominant_red_source_state"] == "zero"
    assert summary["route_decision"] == "CROSS_PLUS_FROM_ZERO_DOMINATES"
    assert summary["classification_field_used_for_source_kind"] is False
    assert summary["navier_stokes_promotion"] is False
