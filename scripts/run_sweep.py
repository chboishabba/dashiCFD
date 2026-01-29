#!/usr/bin/env python3
"""
Run kernel-only rollouts against a truth dataset and score error metrics.

Example:
  python scripts/run_sweep.py \
    --truth outputs/truth_2026-01-29T000000.npz \
    --z0 outputs/exp_base_2026-01-29T000000_z0.npz \
    --A outputs/exp_base_2026-01-29T000000_A.npz \
    --noise-levels 0,0.01,0.03 \
    --out outputs/sweep
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from dashi_cfd_operator_v4 import (
    ProxyConfig,
    decode_with_residual,
    energy_from_omega,
    enstrophy,
    make_grid,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, required=True, help="truth NPZ (omega_snapshots + steps)")
    p.add_argument("--z0", type=Path, required=True, help="z0 artifact NPZ")
    p.add_argument("--A", type=Path, required=True, help="A artifact NPZ")
    p.add_argument("--noise-levels", type=str, default="0.0", help="comma-separated z0 noise levels (fraction of z0 std)")
    p.add_argument("--decode-backend", choices=["cpu", "vulkan"], default="cpu", help="decode backend (default: cpu)")
    p.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="FFT backend for GPU decode")
    p.add_argument("--decode-seed", type=int, default=0, help="RNG seed for decode residuals")
    p.add_argument("--out", type=Path, default=Path("outputs/sweep"), help="output prefix")
    p.add_argument("--plot", action="store_true", help="save a summary plot (outputs/..._plot.png)")
    return p.parse_args()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.size == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + 1e-12
    return float(num / den)


def _parse_levels(spec: str) -> List[float]:
    vals = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals or [0.0]


def main() -> None:
    args = parse_args()
    run_ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_prefix = args.out
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(str(out_prefix) + f"_{run_ts}.json")

    def _resolve_single(path: Path) -> Path:
        if path.exists():
            return path
        matches = sorted(Path(path.parent).glob(path.name))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"{path} (matches: {len(matches)})")

    truth_path = _resolve_single(args.truth)
    z0_path = _resolve_single(args.z0)
    A_path = _resolve_single(args.A)

    truth = dict(np.load(truth_path, allow_pickle=True))
    omega_truth = truth["omega_snapshots"]
    truth_steps = truth["steps"].astype(np.int64)
    if omega_truth.ndim != 3:
        raise ValueError(f"truth omega_snapshots must be 3D, got {omega_truth.shape}")

    z0_data = dict(np.load(z0_path, allow_pickle=True))
    A_data = dict(np.load(A_path, allow_pickle=True))
    z0 = z0_data["z"]
    A = A_data["A"]
    mask_low = z0_data["mask_low"].astype(bool)
    anchor_idx = z0_data.get("anchor_idx")
    if anchor_idx is not None and anchor_idx.size == 0:
        anchor_idx = None

    N = int(omega_truth.shape[1])
    if mask_low.ndim == 1 and mask_low.size == N * N:
        mask_low = mask_low.reshape(N, N)
    grid = make_grid(N)

    meta = {}
    if "meta_json" in z0_data:
        try:
            meta = json.loads(str(z0_data["meta_json"]))
        except Exception:
            meta = {}
    cfg = ProxyConfig(
        k_cut=float(meta.get("k_cut", 8.0)),
        resid_mid_cut=float(meta.get("resid_mid_cut", 12.0)),
        dashi_tau=float(meta.get("dashi_tau", 0.35)),
        dashi_smooth_k=int(meta.get("dashi_smooth_k", 11)),
        topk_mid=int(meta.get("topk_mid", 128)),
    )

    noise_levels = _parse_levels(args.noise_levels)
    rng = np.random.default_rng(args.decode_seed)
    z0_std = float(np.std(z0))

    total_steps = int(truth_steps[-1])
    snapshot_map = {int(t): i for i, t in enumerate(truth_steps)}

    runs: List[Dict[str, object]] = []
    for level in noise_levels:
        z_init = z0.astype(np.float64, copy=True)
        if level > 0:
            z_init = z_init + (level * z0_std) * rng.standard_normal(z_init.shape)
        Zhat = np.zeros((2, A.shape[0]), dtype=z_init.dtype)
        Zhat[0] = z_init

        per_snap: List[Dict[str, float]] = []
        for t in range(total_steps):
            Zhat[1] = Zhat[0] @ A
            step = t + 1
            if step in snapshot_map:
                idx = snapshot_map[step]
                omega_true = omega_truth[idx].astype(np.float64)
                omega_hat, _, _, _, _ = decode_with_residual(
                    Zhat[1],
                    grid,
                    cfg,
                    mask_low,
                    anchor_idx,
                    rng,
                    backend=args.decode_backend,
                    fft_backend=args.fft_backend,
                    observer="snapshots",
                )
                if omega_hat is None:
                    raise RuntimeError("decode_with_residual returned None; enable readback or use cpu backend")
                per_snap.append(
                    {
                        "t": float(step),
                        "rel_l2": _rel_l2(omega_hat, omega_true),
                        "corr": _corr(omega_hat, omega_true),
                        "energy_hat": float(energy_from_omega(omega_hat, grid[1], grid[2], grid[3])),
                        "energy_true": float(energy_from_omega(omega_true, grid[1], grid[2], grid[3])),
                        "enstrophy_hat": float(enstrophy(omega_hat)),
                        "enstrophy_true": float(enstrophy(omega_true)),
                    }
                )
            Zhat[0], Zhat[1] = Zhat[1], Zhat[0]

        rel_l2_mean = float(np.mean([r["rel_l2"] for r in per_snap])) if per_snap else 0.0
        corr_mean = float(np.mean([r["corr"] for r in per_snap])) if per_snap else 0.0
        runs.append(
            {
                "noise_level": float(level),
                "rel_l2_mean": rel_l2_mean,
                "corr_mean": corr_mean,
                "per_snapshot": per_snap,
            }
        )

    payload = {
        "run_ts": run_ts,
        "truth": str(args.truth),
        "z0": str(args.z0),
        "A": str(args.A),
        "decode_backend": args.decode_backend,
        "fft_backend": args.fft_backend,
        "noise_levels": noise_levels,
        "total_steps": total_steps,
        "snapshots": len(truth_steps),
        "runs": runs,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[sweep] wrote {out_path}")

    if args.plot and runs:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax_l2, ax_corr = axes
        for run in runs:
            label = f"noise={run['noise_level']}"
            ts = [r["t"] for r in run["per_snapshot"]]
            rel_l2 = [r["rel_l2"] for r in run["per_snapshot"]]
            corr = [r["corr"] for r in run["per_snapshot"]]
            ax_l2.plot(ts, rel_l2, label=label)
            ax_corr.plot(ts, corr, label=label)
        ax_l2.set_title("rel_l2 vs t")
        ax_l2.set_xlabel("t")
        ax_l2.set_ylabel("rel_l2")
        ax_corr.set_title("corr vs t")
        ax_corr.set_xlabel("t")
        ax_corr.set_ylabel("corr")
        ax_corr.legend()
        fig.tight_layout()
        plot_path = Path(str(out_path).replace(".json", "_plot.png"))
        fig.savefig(plot_path, dpi=150)
        print(f"[sweep] wrote {plot_path}")


if __name__ == "__main__":
    main()
