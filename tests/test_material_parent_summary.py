from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ns_material_parent_summary import _route_status_from_summary  # noqa: E402


def _make_synthetic_truth(path: Path) -> None:
    rng = np.random.default_rng(7)
    t, n = 5, 16
    omega = rng.normal(size=(t, n, n, n, 3)).astype(np.float64)
    velocity = 0.05 * rng.normal(size=(t, n, n, n, 3)).astype(np.float64)
    steps = np.arange(t, dtype=np.int64)

    meta = {
        "backend": "cpu",
        "dt": 0.001,
        "domain_length": 2.0 * np.pi,
        "k_star": 3,
        "device": {"device_name": "synthetic"},
        "gpu": {"spv_shaders": []},
        "host_provenance": {},
    }

    payload = {
        "omega_snapshots": omega,
        "velocity_snapshots": velocity,
        "steps": steps,
        "meta_json": json.dumps(meta),
    }
    np.savez_compressed(path, **payload)


def test_material_parent_summary_contract(tmp_path: Path) -> None:
    truth = tmp_path / "synthetic_material_parent_truth.npz"
    _make_synthetic_truth(truth)

    out = tmp_path / "material_parent_out"
    out.mkdir()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ns_material_parent_summary.py"),
            "--truth",
            str(truth),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    table = out / "ns_material_parent_table.csv"
    summary = out / "ns_material_parent_summary.csv"
    manifest = out / "ns_material_parent_summary.json"

    assert table.exists()
    assert summary.exists()
    assert manifest.exists()

    with table.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    required_table = {
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
    }
    assert required_table.issubset(set(header))

    with summary.open(newline="", encoding="utf-8") as handle:
        summ_header = next(csv.reader(handle))
    required_summary = {
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
    }
    assert required_summary.issubset(set(summ_header))

    manifest_text = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_text["contract"] == "ns_material_parent_artifact"
    assert manifest_text["table_row_count"] >= 1
    assert manifest_text["summary_row_count"] >= 1


def test_material_parent_route_uses_dominant_weighted_component() -> None:
    route = _route_status_from_summary(
        sigma_true_new=0.0,
        sigma_tracking_uncertain=0.01,
        sigma_cross_shell=0.46,
        sigma_low_shell=0.25,
        weighted_true_new=0.0,
        weighted_tracking_uncertain=2.6e8,
        weighted_cross_shell=7.8e13,
        weighted_low_shell=2.4e6,
    )
    assert route == "ADJACENT_PACKET_THEOREM_INSUFFICIENT"
