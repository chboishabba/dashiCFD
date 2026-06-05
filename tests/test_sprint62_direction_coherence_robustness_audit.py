from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_sprint61_csv(path: Path) -> None:
    rows = [
        {
            "run": "sprint49_material_parent_N32_seed0_gpu",
            "weighted_A_raw_positive": "100",
            "direction_coherence_mean": "0.5",
            "direction_lipschitz_proxy": "3.0",
        },
        {
            "run": "sprint49_material_parent_N32_seed1_gpu",
            "weighted_A_raw_positive": "90",
            "direction_coherence_mean": "0.55",
            "direction_lipschitz_proxy": "3.1",
        },
        {
            "run": "sprint49_material_parent_N64_seed0_gpu",
            "weighted_A_raw_positive": "80",
            "direction_coherence_mean": "0.45",
            "direction_lipschitz_proxy": "4.0",
        },
        {
            "run": "sprint49_material_parent_N64_seed1_gpu",
            "weighted_A_raw_positive": "70",
            "direction_coherence_mean": "0.52",
            "direction_lipschitz_proxy": "3.5",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_sprint62_direction_coherence_robustness_contract(tmp_path: Path) -> None:
    source = tmp_path / "sprint61.csv"
    _write_sprint61_csv(source)
    out = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_sprint62_direction_coherence_robustness_audit.py"),
            "--sprint61-csv",
            str(source),
            "--out-dir",
            str(out),
            "--top-fractions",
            "0.5",
            "1.0",
            "--coherence-thresholds",
            "0.6",
            "0.8",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out / "ns_sprint62_direction_coherence_robustness_summary.json").read_text(encoding="utf-8"))
    sensitivity = list(csv.DictReader((out / "ns_direction_coherence_sensitivity.csv").open(newline="", encoding="utf-8")))
    groups = list(csv.DictReader((out / "ns_direction_coherence_by_run.csv").open(newline="", encoding="utf-8")))

    assert summary["contract"] == "ns_sprint62_direction_coherence_robustness_artifact"
    assert summary["clay_navier_stokes_promoted"] is False
    assert summary["n128_available"] is False
    assert sensitivity
    assert groups
    assert summary["route_decision"] == "DIRECTION_COHERENCE_INCOHERENCE_ROBUST_ON_AVAILABLE_DATA"
