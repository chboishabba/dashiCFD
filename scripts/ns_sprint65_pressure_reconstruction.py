#!/usr/bin/env python3
"""Sprint 65 periodic pressure reconstruction for CKN audits.

Sprint 64 can compute the velocity part of a CKN-style local critical
concentration surface, but current truth artifacts do not store pressure.
Sprint 65 reconstructs a zero-mean periodic pressure snapshot from velocity:

    Delta p = - sum_ij (partial_i u_j) (partial_j u_i)

for each saved frame.  The output NPZ copies the source artifact arrays and
adds pressure_snapshots plus pressure_reconstruction_summary_json.  This is a
diagnostic reconstruction only; it does not prove a CKN epsilon theorem or
promote Clay/NS.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-30


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="truth3d NPZ artifacts with velocity_snapshots")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="stored pressure dtype")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _meta(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "meta_json" not in npz.files:
        return {}
    try:
        return json.loads(str(npz["meta_json"]))
    except Exception:
        return {}


def _wavenumbers(n: int, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = float(length) / float(n)
    k = np.fft.fftfreq(n, d=dx) * 2.0 * math.pi
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    return kx, ky, kz, k2


def _grad_u(u: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
    u_hat = np.fft.fftn(np.asarray(u, dtype=np.float64), axes=(0, 1, 2))
    grad = np.empty(u.shape + (3,), dtype=np.float64)
    for comp in range(3):
        grad[..., comp, 0] = np.fft.ifftn(1j * kx * u_hat[..., comp], axes=(0, 1, 2)).real
        grad[..., comp, 1] = np.fft.ifftn(1j * ky * u_hat[..., comp], axes=(0, 1, 2)).real
        grad[..., comp, 2] = np.fft.ifftn(1j * kz * u_hat[..., comp], axes=(0, 1, 2)).real
    return grad


def _pressure_from_velocity(
    u: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    k2: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    grad = _grad_u(u, kx, ky, kz)
    source = np.zeros(u.shape[:3], dtype=np.float64)
    # source = sum_ij partial_i u_j partial_j u_i.
    for j in range(3):
        for i in range(3):
            source += grad[..., j, i] * grad[..., i, j]

    source_hat = np.fft.fftn(source, axes=(0, 1, 2))
    p_hat = np.zeros_like(source_hat)
    np.divide(source_hat, k2, out=p_hat, where=k2 > 0.0)
    p_hat[0, 0, 0] = 0.0
    pressure = np.fft.ifftn(p_hat, axes=(0, 1, 2)).real
    pressure -= float(pressure.mean())

    lap_p = np.fft.ifftn(-k2 * p_hat, axes=(0, 1, 2)).real
    source_zero_mean = source - float(source.mean())
    residual = lap_p + source_zero_mean
    source_rms = float(np.sqrt(np.mean(source_zero_mean * source_zero_mean)))
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    return pressure, {
        "pressure_mean": float(pressure.mean()),
        "pressure_abs_mean": float(np.mean(np.abs(pressure))),
        "pressure_linf": float(np.max(np.abs(pressure))),
        "poisson_source_rms": source_rms,
        "poisson_residual_rms": residual_rms,
        "poisson_relative_residual_rms": residual_rms / (source_rms + EPS),
        "source_mean_removed": float(source.mean()),
    }


def _copy_npz_with_pressure(path: Path, out_path: Path, dtype: np.dtype) -> dict[str, Any]:
    z = np.load(path)
    if "velocity_snapshots" not in z.files:
        raise SystemExit(f"{path} is missing velocity_snapshots")
    velocity = np.asarray(z["velocity_snapshots"])
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise SystemExit(f"{path} velocity_snapshots must have shape (T,N,N,N,3)")

    meta = _meta(z)
    n = int(velocity.shape[1])
    length = float(meta.get("domain_length", 2.0 * math.pi))
    kx, ky, kz, k2 = _wavenumbers(n, length)

    pressure = np.empty(velocity.shape[:4], dtype=dtype)
    frame_stats: list[dict[str, float]] = []
    for idx in range(velocity.shape[0]):
        p, stats = _pressure_from_velocity(velocity[idx], kx, ky, kz, k2)
        pressure[idx] = p.astype(dtype, copy=False)
        stats["frame_index"] = float(idx)
        frame_stats.append(stats)

    max_rel = max((s["poisson_relative_residual_rms"] for s in frame_stats), default=0.0)
    max_abs_mean = max((abs(s["pressure_mean"]) for s in frame_stats), default=0.0)
    summary: dict[str, Any] = {
        "contract": "ns_sprint65_pressure_reconstruction_artifact",
        "input_path": str(path),
        "output_path": str(out_path),
        "N": n,
        "frame_count": int(velocity.shape[0]),
        "domain_length": length,
        "pressure_dtype": str(np.dtype(dtype)),
        "poisson_equation": "Delta p = - sum_ij partial_i u_j partial_j u_i",
        "pressure_gauge": "zero_mean_per_frame",
        "max_poisson_relative_residual_rms": max_rel,
        "mean_poisson_relative_residual_rms": float(np.mean([s["poisson_relative_residual_rms"] for s in frame_stats])) if frame_stats else 0.0,
        "max_pressure_abs_mean": max_abs_mean,
        "max_pressure_linf": max((s["pressure_linf"] for s in frame_stats), default=0.0),
        "pressure_reconstruction_proved": False,
        "ckn_epsilon_regularity_applied": False,
        "clay_navier_stokes_promoted": False,
        "promotion_status": "NO_PROMOTION_SPRINT65_PRESSURE_RECONSTRUCTION_DIAGNOSTIC",
        "frame_stats": frame_stats,
    }

    arrays: dict[str, Any] = {key: z[key] for key in z.files}
    arrays["pressure_snapshots"] = pressure
    arrays["pressure_reconstruction_summary_json"] = json.dumps(summary, allow_nan=True)
    if "meta_json" in arrays:
        augmented_meta = _meta(z)
        augmented_meta["has_pressure_snapshots"] = True
        augmented_meta["pressure_reconstruction"] = {
            "contract": summary["contract"],
            "poisson_equation": summary["poisson_equation"],
            "pressure_gauge": summary["pressure_gauge"],
            "source": str(path),
        }
        arrays["meta_json"] = json.dumps(augmented_meta, allow_nan=True)
    np.savez(out_path, **arrays)
    return summary


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(args.dtype)
    summaries: list[dict[str, Any]] = []
    for path in args.inputs:
        out_path = args.out_dir / f"{path.stem}_pressure.npz"
        if out_path.exists() and not bool(args.overwrite):
            raise SystemExit(f"{out_path} exists; pass --overwrite to replace")
        summary = _copy_npz_with_pressure(path, out_path, dtype)
        summaries.append(summary)
        print(
            "[ns_sprint65_pressure_reconstruction] "
            f"wrote {out_path} max_rel_residual={summary['max_poisson_relative_residual_rms']} "
            f"max_pressure_linf={summary['max_pressure_linf']}"
        )
    manifest = {
        "contract": "ns_sprint65_pressure_reconstruction_manifest",
        "artifact_count": len(summaries),
        "artifacts": summaries,
        "max_poisson_relative_residual_rms": max((s["max_poisson_relative_residual_rms"] for s in summaries), default=0.0),
        "pressure_reconstruction_proved": False,
        "ckn_epsilon_regularity_applied": False,
        "clay_navier_stokes_promoted": False,
    }
    manifest_path = args.out_dir / "ns_sprint65_pressure_reconstruction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[ns_sprint65_pressure_reconstruction] wrote {manifest_path}")


if __name__ == "__main__":
    main()
