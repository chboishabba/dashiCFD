#!/usr/bin/env python3
"""
Build kernel experiment artifacts (z0/A) from a saved truth NPZ.

Example:
  python scripts/make_exp_base.py --truth outputs/truth_2026-01-29T000000.npz --out-prefix outputs/exp_base
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from dashi_cfd_operator_v4 import (
    ProxyConfig,
    circular_kmask,
    encode_proxy,
    learn_linear_operator,
    make_grid,
)
from vulkan_encode_backend import VulkanEncodeBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=str, required=True, help="truth NPZ with omega_snapshots + steps")
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge regularization (default: 1e-3)")
    p.add_argument("--encode-backend", choices=["cpu", "gpu"], default="cpu", help="encode backend (default: cpu)")
    p.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="FFT backend for GPU encode")
    p.add_argument("--k-cut", type=float, default=8.0, dest="k_cut", help="proxy low-k cutoff (default: 8)")
    p.add_argument("--resid-mid-cut", type=float, default=12.0, dest="resid_mid_cut", help="residual mid-band cutoff (default: 12)")
    p.add_argument("--dashi-tau", type=float, default=0.35, dest="dashi_tau", help="DASHI ternary threshold (default: 0.35)")
    p.add_argument("--dashi-smooth-k", type=int, default=11, dest="dashi_smooth_k", help="DASHI smoothing window (default: 11)")
    p.add_argument("--topk-mid", type=int, default=128, dest="topk_mid", help="mid-band coeffs to preserve (default: 128)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64", help="storage dtype for z/A")
    p.add_argument("--out-prefix", type=Path, default=Path("outputs/exp_base"), help="output prefix")
    return p.parse_args()


def _resolve_truth_path(spec: str) -> Path:
    if spec.startswith("latest:"):
        tag = spec.split(":", 1)[1].strip()
        manifest_path = Path("outputs/truth_manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found for {spec}")
        manifest = json.loads(manifest_path.read_text())
        entry = manifest.get("latest", {}).get(tag)
        if not entry or "npz" not in entry:
            raise FileNotFoundError(f"{spec} not found in {manifest_path}")
        return Path(entry["npz"])
    path = Path(spec)
    if not path.exists():
        matches = sorted(Path(path.parent).glob(path.name))
        if len(matches) == 1:
            path = matches[0]
        else:
            raise FileNotFoundError(f"{path} (matches: {len(matches)})")
    return path


def _load_truth(spec: str) -> Dict[str, np.ndarray]:
    path = _resolve_truth_path(spec)
    return dict(np.load(path, allow_pickle=True))


def main() -> None:
    args = parse_args()
    run_ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")

    truth = _load_truth(args.truth)
    omega_snapshots = truth["omega_snapshots"]
    steps = truth["steps"]
    if omega_snapshots.ndim != 3:
        raise ValueError(f"truth omega_snapshots must be 3D, got {omega_snapshots.shape}")

    N = int(omega_snapshots.shape[1])
    grid = make_grid(N)

    cfg = ProxyConfig(
        k_cut=args.k_cut,
        resid_mid_cut=args.resid_mid_cut,
        dashi_tau=args.dashi_tau,
        dashi_smooth_k=args.dashi_smooth_k,
        topk_mid=args.topk_mid,
    )

    mask_low = circular_kmask(grid[1], grid[2], cfg.k_cut)

    Z = []
    anchor_idx: Optional[np.ndarray] = None
    encoder = None
    if args.encode_backend == "gpu":
        encoder = VulkanEncodeBackend(
            N,
            cfg,
            fft_backend=args.fft_backend,
            timing_enabled=False,
        )

    for frame in omega_snapshots:
        if encoder is None:
            z, mask_low_cpu, anchor_idx = encode_proxy(frame.astype(np.float64), grid, cfg, anchor_idx=anchor_idx)
            if mask_low_cpu is not None:
                mask_low = mask_low_cpu
        else:
            z, anchor_idx = encoder.encode_proxy(frame.astype(np.float32), mask_low, anchor_idx=anchor_idx)
        Z.append(z)

    Z = np.stack(Z, axis=0).astype(args.dtype)
    z0 = Z[0]
    A = learn_linear_operator(Z, ridge=args.ridge).astype(args.dtype)

    meta: Dict[str, object] = {
        "run_ts": run_ts,
        "truth_path": str(_resolve_truth_path(args.truth)),
        "truth_steps": [int(x) for x in steps.tolist()],
        "truth_backend": None,
        "truth_tag": None,
        "N": N,
        "k_cut": float(args.k_cut),
        "resid_mid_cut": float(args.resid_mid_cut),
        "topk_mid": int(args.topk_mid),
        "dashi_tau": float(args.dashi_tau),
        "dashi_smooth_k": int(args.dashi_smooth_k),
        "ridge": float(args.ridge),
        "dtype": args.dtype,
        "encode_backend": args.encode_backend,
        "fft_backend": args.fft_backend,
    }
    if "meta_json" in truth:
        try:
            truth_meta = json.loads(str(truth["meta_json"]))
            meta["truth_backend"] = truth_meta.get("backend")
            meta["truth_tag"] = truth_meta.get("truth_tag")
        except Exception:
            pass

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    base = Path(str(out_prefix) + f"_{run_ts}")
    z0_path = Path(str(base) + "_z0.npz")
    A_path = Path(str(base) + "_A.npz")
    meta_path = Path(str(base) + "_meta.json")

    np.savez(
        z0_path,
        z=z0,
        mask_low=mask_low.reshape(-1),
        anchor_idx=anchor_idx,
        meta_json=json.dumps(meta),
    )
    np.savez(A_path, A=A)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[exp_base] wrote {z0_path}, {A_path}, {meta_path}")


if __name__ == "__main__":
    main()
