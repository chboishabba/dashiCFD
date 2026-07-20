#!/usr/bin/env python3
"""Run the sparse-vortex atom vertical slice against canonical truth snapshots.

The probe performs the narrow integration target:

1. extract signed spatial atoms from one vorticity snapshot;
2. transport the smooth carrier and atoms without further truth access;
3. decode each predicted state;
4. measure the one-step and finite-rollout commutation defects;
5. write genealogy, MDL event accounting, timing, and fidelity metrics.

CPU float64 is the receipt authority.  This script does not replace the existing
LES truth generator and does not promote a Navier--Stokes theorem.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashi_vortex_atoms import (
    AtomCodecConfig,
    decode_proxy,
    defect_metrics,
    extract_vortex_atoms,
    genealogy_rows,
    proxy_step,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, required=True, help="NPZ containing omega_snapshots")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/vortex_atom_rollout"))
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=10)
    p.add_argument("--dt", type=float, default=None, help="override snapshot-to-snapshot dt")
    p.add_argument("--viscosity", type=float, default=None, help="override nu0/nu from metadata")
    p.add_argument("--smooth-k", type=int, default=11)
    p.add_argument("--threshold-sigma", type=float, default=0.75)
    p.add_argument("--max-atoms", type=int, default=32)
    p.add_argument("--peak-candidates", type=int, default=48)
    p.add_argument("--scales", type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--anisotropies", type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--orientations", type=int, default=8)
    p.add_argument("--bits-per-atom", type=float, default=160.0)
    p.add_argument("--q", type=float, default=0.05)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def load_truth(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    data = np.load(path, allow_pickle=True)
    if "omega_snapshots" not in data:
        raise KeyError(f"{path} does not contain omega_snapshots")
    snapshots = np.asarray(data["omega_snapshots"], dtype=np.float64)
    steps = np.asarray(data["steps"], dtype=np.int64) if "steps" in data else np.arange(len(snapshots))
    meta: dict[str, object] = {}
    if "meta_json" in data:
        try:
            meta = json.loads(str(data["meta_json"]))
        except Exception:
            meta = {"meta_json_parse_error": True}
    return snapshots, steps, meta


def resolve_dt(args: argparse.Namespace, steps: np.ndarray, meta: dict[str, object]) -> float:
    if args.dt is not None:
        return float(args.dt)
    base_dt = float(meta.get("dt", 1.0))
    if len(steps) > 1:
        stride = int(steps[1] - steps[0])
        return base_dt * stride
    return base_dt


def resolve_viscosity(args: argparse.Namespace, meta: dict[str, object]) -> float:
    if args.viscosity is not None:
        return float(args.viscosity)
    for key in ("nu0", "nu", "viscosity"):
        if key in meta:
            return float(meta[key])
    return 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, truth: np.ndarray, decoded: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = truth - decoded
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, field, name in (
        (axes[0], truth, "truth omega"),
        (axes[1], decoded, "sparse twist decode"),
        (axes[2], error, "commutation defect"),
    ):
        im = ax.imshow(field, origin="lower")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    snapshots, truth_steps, meta = load_truth(args.truth)
    start = int(args.start_index)
    if start < 0:
        start += len(snapshots)
    horizon = min(max(int(args.rollout_steps), 1), len(snapshots) - start - 1)
    if horizon <= 0:
        raise ValueError("truth artifact does not contain a next snapshot for the requested start")

    dt = resolve_dt(args, truth_steps, meta)
    viscosity = resolve_viscosity(args, meta)
    config = AtomCodecConfig(
        smooth_k=int(args.smooth_k),
        threshold_sigma=float(args.threshold_sigma),
        peak_candidates=int(args.peak_candidates),
        max_atoms=int(args.max_atoms),
        scales=tuple(float(x) for x in args.scales),
        anisotropies=tuple(float(x) for x in args.anisotropies),
        orientations=int(args.orientations),
        bits_per_atom=float(args.bits_per_atom),
        q=float(args.q),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extract_start = perf_counter()
    state = extract_vortex_atoms(snapshots[start], config)
    extraction_seconds = perf_counter() - extract_start

    initial_decode = decode_proxy(state)
    initial_metrics = defect_metrics(snapshots[start], initial_decode)
    metric_rows: list[dict[str, object]] = [
        {
            "rollout_index": 0,
            "truth_index": start,
            "truth_step": int(truth_steps[start]),
            "atom_count": len(state.atoms),
            **initial_metrics.to_json(),
            "update_seconds": 0.0,
            "decode_seconds": 0.0,
        }
    ]
    genealogy: list[dict[str, object]] = genealogy_rows(state.atoms, 0)
    ledgers: list[dict[str, object]] = []

    if args.plot:
        write_plot(args.out_dir / "snapshot_000.png", snapshots[start], initial_decode, "initial codec defect")

    for rollout_index in range(1, horizon + 1):
        update_start = perf_counter()
        candidate_state, provisional_ledger = proxy_step(
            state,
            dt,
            config,
            viscosity=viscosity,
        )
        update_seconds = perf_counter() - update_start

        decode_start = perf_counter()
        decoded = decode_proxy(candidate_state)
        decode_seconds = perf_counter() - decode_start
        truth = snapshots[start + rollout_index]
        metrics = defect_metrics(truth, decoded)

        # Recompute the defect contribution now that the actual one-step/rollout
        # defect is known.  The state transition itself remains truth-free.
        _, ledger = proxy_step(
            state,
            dt,
            config,
            viscosity=viscosity,
            defect_rel_l2=metrics.rel_l2,
        )
        state = candidate_state

        metric_rows.append(
            {
                "rollout_index": rollout_index,
                "truth_index": start + rollout_index,
                "truth_step": int(truth_steps[start + rollout_index]),
                "atom_count": len(state.atoms),
                **metrics.to_json(),
                "update_seconds": update_seconds,
                "decode_seconds": decode_seconds,
            }
        )
        genealogy.extend(genealogy_rows(state.atoms, rollout_index))
        ledgers.append(ledger.to_json())

        if args.plot:
            write_plot(
                args.out_dir / f"snapshot_{rollout_index:03d}.png",
                truth,
                decoded,
                f"rollout {rollout_index}; truth step {truth_steps[start + rollout_index]}",
            )

    write_csv(args.out_dir / "defect_trace.csv", metric_rows)
    write_csv(args.out_dir / "atom_genealogy.csv", genealogy)
    write_csv(args.out_dir / "mdl_event_ledger.csv", ledgers)

    summary = {
        "status": "ok_empirical_bounded_fidelity_no_ns_promotion",
        "truth": str(args.truth),
        "start_index": start,
        "rollout_steps": horizon,
        "dt_per_snapshot": dt,
        "viscosity": viscosity,
        "config": {
            "smooth_k": config.smooth_k,
            "threshold_sigma": config.threshold_sigma,
            "peak_candidates": config.peak_candidates,
            "max_atoms": config.max_atoms,
            "scales": list(config.scales),
            "anisotropies": list(config.anisotropies),
            "orientations": config.orientations,
            "bits_per_atom": config.bits_per_atom,
            "q": config.q,
        },
        "extraction_seconds": extraction_seconds,
        "initial_atom_count": int(metric_rows[0]["atom_count"]),
        "final_atom_count": int(metric_rows[-1]["atom_count"]),
        "one_step_defect": metric_rows[1],
        "final_defect": metric_rows[-1],
        "mean_update_seconds": float(np.mean([float(r["update_seconds"]) for r in metric_rows[1:]])),
        "mean_decode_seconds": float(np.mean([float(r["decode_seconds"]) for r in metric_rows[1:]])),
        "total_mdl_bits_final": float(ledgers[-1]["total_mdl_bits"]) if ledgers else 0.0,
        "boundary": (
            "The atom rollout is an empirical CPU-float64 vertical slice.  It does not prove "
            "uniform frame bounds, a continuum commuting square, runtime speedup, Navier--Stokes "
            "closure, regularity, or Clay authority."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
