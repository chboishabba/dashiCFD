from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_ROUTE_DECISIONS = {
    "CROSS_SHELL_REPLENISHMENT_SOURCE_UNAVAILABLE",
    "CROSS_SHELL_REPLENISHMENT_NO_EDGES",
    "CROSS_SHELL_REPLENISHMENT_CONTRACTIVE_ON_AVAILABLE_DATA",
    "CROSS_SHELL_REPLENISHMENT_MIXED",
    "CROSS_SHELL_REPLENISHMENT_NONCONTRACTIVE_BLOCKED",
}

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

RAW_FIELDS = [
    "run",
    "time",
    "dt",
    "K",
    "packet_id",
    "A_raw_positive",
    "A_raw_negative",
    "A_raw_net",
    "A_raw_total",
    "packet_enstrophy",
    "packet_volume",
    "A_norm_enstrophy_weighted",
    "weighted_A_raw_positive",
    "lagrangian_trit_after_integration",
    "packet_local_mask_available",
    "raw_action_boundary",
]


def _write_sprint49(path: Path) -> None:
    path.mkdir()
    rows = [
        {
            "time": "0.1",
            "dt": "0.1",
            "K_parent": "2",
            "K_child": "3",
            "child_packet_id": "K3_cell1",
            "parent_packet_id": "K2_cell1",
            "child_state": "plus",
            "parent_state": "plus",
            "child_mass": "10",
            "parent_mass": "10",
            "credited_mass": "10",
            "source_true_new": "0",
            "source_tracking_uncertain": "0",
            "source_cross_shell": "1",
            "source_low_shell_injection": "0",
            "advected_overlap": "0.5",
            "centroid_distance": "1",
            "direction_cosine": "0.5",
            "shell_delta": "1",
            "parent_confidence": "0.75",
            "parent_relation": "cross_shell_parent",
            "classification": "plus_to_plus",
        },
        {
            "time": "0.2",
            "dt": "0.1",
            "K_parent": "2",
            "K_child": "3",
            "child_packet_id": "K3_cell2",
            "parent_packet_id": "K2_cell2",
            "child_state": "plus",
            "parent_state": "minus",
            "child_mass": "10",
            "parent_mass": "10",
            "credited_mass": "10",
            "source_true_new": "0",
            "source_tracking_uncertain": "0",
            "source_cross_shell": "1",
            "source_low_shell_injection": "0",
            "advected_overlap": "0.5",
            "centroid_distance": "1",
            "direction_cosine": "0.5",
            "shell_delta": "1",
            "parent_confidence": "0.75",
            "parent_relation": "cross_shell_parent",
            "classification": "minus_to_plus",
        },
        {
            "time": "0.2",
            "dt": "0.1",
            "K_parent": "3",
            "K_child": "3",
            "child_packet_id": "K3_cell3",
            "parent_packet_id": "K3_cell3",
            "child_state": "plus",
            "parent_state": "plus",
            "child_mass": "10",
            "parent_mass": "10",
            "credited_mass": "10",
            "source_true_new": "0",
            "source_tracking_uncertain": "0",
            "source_cross_shell": "0",
            "source_low_shell_injection": "0",
            "advected_overlap": "0.9",
            "centroid_distance": "0",
            "direction_cosine": "1",
            "shell_delta": "0",
            "parent_confidence": "0.9",
            "parent_relation": "advected_parent",
            "classification": "plus_to_plus",
        },
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
                "K_child": "3",
                "M_plus_plus_material": "0.4",
                "source_true_new": "0",
                "source_tracking_uncertain": "0",
                "source_cross_shell": "1",
                "source_low_shell_injection": "0",
                "source_total_material": "1",
                "weighted_true_new": "0",
                "weighted_tracking_uncertain": "0",
                "weighted_cross_shell": "1",
                "weighted_low_shell_injection": "0",
                "weighted_total_material": "1",
                "sigma_true_new_fit": "0",
                "sigma_tracking_uncertain_fit": "0",
                "sigma_cross_shell_fit": "0",
                "sigma_low_shell_fit": "0",
                "sigma_total_material_fit": "0",
                "route_status": "ADJACENT_PACKET_THEOREM_INSUFFICIENT",
            }
        )
    (path / "ns_material_parent_summary.json").write_text(
        json.dumps({"contract": "ns_material_parent_artifact", "table_row_count": len(rows), "summary_row_count": 1}),
        encoding="utf-8",
    )


def _raw_row(run: str, time: str, k: str, packet_id: str, positive: str) -> dict[str, str]:
    return {
        "run": run,
        "time": time,
        "dt": "0.1",
        "K": k,
        "packet_id": packet_id,
        "A_raw_positive": positive,
        "A_raw_negative": "0",
        "A_raw_net": positive,
        "A_raw_total": positive,
        "packet_enstrophy": "1",
        "packet_volume": "1",
        "A_norm_enstrophy_weighted": positive,
        "weighted_A_raw_positive": positive,
        "lagrangian_trit_after_integration": "plus",
        "packet_local_mask_available": "true",
        "raw_action_boundary": "test",
    }


def _write_raw(path: Path, run: str) -> None:
    rows = [
        _raw_row(run, "0.0", "2", "K2_cell1", "4"),
        _raw_row(run, "0.1", "3", "K3_cell1", "2"),
        _raw_row(run, "0.1", "2", "K2_cell2", "1"),
        _raw_row(run, "0.2", "3", "K3_cell2", "5"),
        _raw_row(run, "0.2", "3", "K3_cell3", "7"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_sprint63_cross_shell_replenishment_contractivity_contract(tmp_path: Path) -> None:
    source = tmp_path / "sprint49_material_parent_N4_seed0"
    _write_sprint49(source)
    raw_csv = tmp_path / "raw.csv"
    _write_raw(raw_csv, source.name)
    out = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_sprint63_cross_shell_replenishment_contractivity_audit.py"),
            "--inputs",
            str(source),
            "--raw-action-csv",
            str(raw_csv),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint63_cross_shell_replenishment_contractivity_summary.json").read_text(encoding="utf-8"))
    edges = list(csv.DictReader((out / "ns_cross_shell_replenishment_contractivity.csv").open(newline="", encoding="utf-8")))
    by_k = list(csv.DictReader((out / "ns_cross_shell_replenishment_contractivity_by_k.csv").open(newline="", encoding="utf-8")))
    by_transition = list(csv.DictReader((out / "ns_cross_shell_replenishment_contractivity_by_transition.csv").open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_sprint63_cross_shell_replenishment_contractivity_artifact"
    assert summary["route_decision"] in ALLOWED_ROUTE_DECISIONS
    assert summary["route_decision"] == "CROSS_SHELL_REPLENISHMENT_MIXED"
    assert summary["contractivity_proved"] is False
    assert summary["physical_bridge_proved"] is False
    assert summary["clay_navier_stokes_promoted"] is False
    assert edges
    assert by_k
    assert by_transition
    assert "contractivity_ratio" in edges[0]
    assert "parent_relation" in edges[0]
    assert "shell_delta" in edges[0]
    assert all(float(row["contractivity_ratio"]) >= 0.0 for row in edges)
    assert any(row["contractivity_route_label"] == "CONTRACTIVE" for row in edges)
    assert any(row["contractivity_route_label"].startswith("NONCONTRACTIVE") for row in edges)
