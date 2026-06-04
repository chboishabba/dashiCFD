from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_make_truth_3d_cpu_artifact_contract(tmp_path: Path) -> None:
    out = tmp_path / "truth3d_cpu_smoke.npz"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_truth_3d.py"),
            "--N",
            "16",
            "--steps",
            "0",
            "--save-every",
            "1",
            "--dt",
            "0.001",
            "--nu0",
            "0.001",
            "--seed",
            "0",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    data = np.load(out)
    meta = json.loads(str(data["meta_json"]))
    assert data["omega_snapshots"].shape == (1, 16, 16, 16, 3)
    assert data["velocity_snapshots"].shape == (1, 16, 16, 16, 3)
    assert data["steps"].tolist() == [0]
    assert meta["dimension"] == 3
    assert meta["field"] == "omega"
    assert meta["backend"] == "cpu"
    assert meta["projection"] == "leray"
    assert meta["dealiasing"] == "2/3"
    assert bool(meta["periodic"])
    assert np.isfinite(data["omega_snapshots"]).all()
