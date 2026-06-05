from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from test_sprint57_vessel_action_reconciliation_audit import _write_input, _write_truth


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_ROUTES = {
    "RAW_RED_DIRECTION_COHERENT_CFM_ROUTE_ALIVE",
    "RAW_RED_DIRECTION_INCOHERENT_CONCENTRATION_BLOCKED",
    "RAW_RED_LOW_CONFIDENCE_ARTIFACT",
    "RAW_RED_DIRECTION_ANATOMY_INCONCLUSIVE",
}


def _run_sprint59(source: Path, out: Path) -> Path:
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
    path = out / "ns_raw_packet_stretch_action.csv"
    assert path.exists()
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert rows
    rows[0]["A_raw_positive"] = "1.0"
    rows[0]["weighted_A_raw_positive"] = "1.0"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_sprint61_direction_coherence_contract(tmp_path: Path) -> None:
    truth = tmp_path / "truth.npz"
    _write_truth(truth)
    source = tmp_path / "sprint49_N4_seed0"
    _write_input(source, truth)
    raw_csv = _run_sprint59(source, tmp_path / "sprint59")
    out = tmp_path / "sprint61"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_sprint61_raw_red_direction_coherence_audit.py"),
            "--inputs",
            str(source),
            "--raw-action-csv",
            str(raw_csv),
            "--out-dir",
            str(out),
            "--top-fraction",
            "1.0",
            "--min-selected",
            "1",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint61_direction_coherence_summary.json").read_text(encoding="utf-8"))
    packet_rows = list(csv.DictReader((out / "ns_raw_red_direction_coherence.csv").open(newline="", encoding="utf-8")))
    by_k_rows = list(csv.DictReader((out / "ns_raw_red_direction_coherence_by_k.csv").open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_sprint61_raw_red_direction_coherence_artifact"
    assert summary["route_decision"] in ALLOWED_ROUTES
    assert summary["clay_navier_stokes_promoted"] is False
    assert summary["cfm_direction_regularity_proved"] is False
    assert packet_rows
    assert by_k_rows
    assert float(packet_rows[0]["direction_coherence_mean"]) >= 0.0
    assert "beltrami_defect_normalized_mean" in packet_rows[0]
    assert "parent_relation" in packet_rows[0]
