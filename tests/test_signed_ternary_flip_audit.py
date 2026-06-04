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


def _base_row(**overrides: str) -> dict[str, str]:
    row = {
        "time": "0.1",
        "dt": "0.01",
        "K_parent": "1",
        "K_child": "4",
        "child_packet_id": "child",
        "parent_packet_id": "parent",
        "child_state": "plus",
        "parent_state": "minus",
        "child_mass": "10",
        "parent_mass": "10",
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
        "parent_relation": "advected_parent",
        "classification": "same_shell_plus",
    }
    row.update(overrides)
    return row


def _write_sprint49_dir(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir()
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


def _run(source: Path, out: Path, *extra: str) -> dict[str, object]:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_signed_ternary_flip_audit.py"),
            "--inputs",
            str(source),
            "--out-dir",
            str(out),
            *extra,
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads((out / "ns_signed_ternary_flip_summary.json").read_text(encoding="utf-8"))


def test_signed_ternary_flip_balanced_route_and_parent_relation_regression(tmp_path: Path) -> None:
    source = tmp_path / "sprint49"
    out = tmp_path / "sprint51"
    rows = [
        _base_row(
            K_parent="1",
            K_child="4",
            parent_state="minus",
            child_state="plus",
            credited_mass="10",
            child_packet_id="m2p",
            parent_packet_id="a",
            classification="same_shell_plus_to_plus",
        ),
        _base_row(
            K_parent="1",
            K_child="4",
            parent_state="plus",
            child_state="minus",
            credited_mass="10",
            child_packet_id="p2m",
            parent_packet_id="b",
            classification="true_new_plus",
        ),
        _base_row(
            K_parent="4",
            K_child="4",
            parent_state="minus",
            child_state="plus",
            credited_mass="100",
            child_packet_id="same",
            parent_packet_id="same",
            parent_relation="advected_parent",
            classification="cross_shell_parent",
        ),
    ]
    _write_sprint49_dir(source, rows)

    summary = _run(source, out)
    balance = list(csv.DictReader((out / "ns_cross_shell_flip_balance.csv").open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_signed_ternary_flip_artifact"
    assert summary["weighted_cross_minus_to_plus"] == 40.0
    assert summary["weighted_cross_plus_to_minus"] == 40.0
    assert summary["signed_flip_imbalance"] == 0.0
    assert summary["does_signed_flip_balance"] is True
    assert summary["route_decision"] == "SIGNED_FLIP_BALANCED_ROUTE_ALIVE"
    assert summary["classification_field_used_for_source_kind"] is False
    assert summary["clay_promotion"] is False
    assert len(balance) == 1
    assert float(balance[0]["cross_minus_to_plus"]) == 40.0
    assert float(balance[0]["cross_plus_to_minus"]) == 40.0


def test_signed_ternary_flip_unbalanced_route_blocks_raw_minus_to_plus(tmp_path: Path) -> None:
    source = tmp_path / "sprint49"
    out = tmp_path / "sprint51"
    rows = [
        _base_row(
            K_parent="1",
            K_child="4",
            parent_state="minus",
            child_state="plus",
            credited_mass="10",
        ),
        _base_row(
            K_parent="1",
            K_child="4",
            parent_state="plus",
            child_state="minus",
            credited_mass="1",
        ),
    ]
    _write_sprint49_dir(source, rows)

    summary = _run(source, out, "--balance-fraction-threshold", "0.01")

    assert summary["weighted_cross_minus_to_plus"] == 40.0
    assert summary["weighted_cross_plus_to_minus"] == 4.0
    assert summary["does_signed_flip_balance"] is False
    assert summary["raw_minus_to_plus_exceeds_plus_to_minus"] is True
    assert summary["route_decision"] == "RAW_MINUS_TO_PLUS_UNBALANCED_ROUTE_BLOCKED"
