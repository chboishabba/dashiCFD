#!/usr/bin/env python3
"""
Probe a signed anisotropic residual-atom codec on an existing dashiCFD truth NPZ.

This is an executable counterpart to the DASHI Agda atom-frame receipts.  It
does not replace the production v4 runner.  It answers one narrow question:
does a deterministic signed anisotropic atom dictionary reduce the residual by
MDL more effectively than leaving the same residual as entropy/noise?
The selected Gram lower eigenvalue is an empirical probe of the
AtomExtendedCarrierFrameReceipt `A > 0` obligation, not a proof of a uniform
lower frame bound.

Input:
  NPZ from scripts/make_truth.py containing omega_snapshots.

Output:
  JSON report with reconstruction metrics, selected atoms, and Gram/frame
  diagnostics for the selected atom family.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashi_les_vorticity_codec_v2 import smooth2d


@dataclass(frozen=True)
class Atom:
    y: int
    x: int
    scale: float
    orientation: float
    anisotropy: float
    sign: int
    amplitude: float
    phase: float
    twist: float
    sse_reduction: float
    mdl_gain: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True, help="truth NPZ with omega_snapshots")
    parser.add_argument("--snapshot-index", type=int, default=-1, help="snapshot index to encode")
    parser.add_argument("--out", type=Path, default=Path("outputs/atom_codec_probe.json"), help="JSON report path")
    parser.add_argument("--smooth-k", type=int, default=11, help="base smoother width")
    parser.add_argument("--max-atoms", type=int, default=24, help="maximum greedy atoms")
    parser.add_argument("--peak-candidates", type=int, default=32, help="residual peaks considered")
    parser.add_argument("--scales", type=float, nargs="+", default=[2.0, 4.0], help="atom major scales")
    parser.add_argument("--anisotropies", type=float, nargs="+", default=[1.0, 3.0], help="atom anisotropy ratios")
    parser.add_argument("--orientations", type=int, default=6, help="number of orientations in [0, pi)")
    parser.add_argument("--bits-per-atom", type=float, default=160.0, help="MDL cost per atom")
    parser.add_argument("--q", type=float, default=0.05, help="distortion quantization scale for MDL")
    parser.add_argument("--min-separation", type=int, default=2, help="minimum candidate peak separation in cells")
    parser.add_argument("--plot", type=Path, default=None, help="optional reconstruction triptych PNG")
    return parser.parse_args()


def _load_snapshot(path: Path, snapshot_index: int) -> tuple[np.ndarray, dict[str, object]]:
    data = np.load(path, allow_pickle=True)
    if "omega_snapshots" not in data:
        raise KeyError(f"{path} does not contain omega_snapshots")
    snapshots = np.asarray(data["omega_snapshots"], dtype=np.float64)
    omega = snapshots[snapshot_index]
    meta: dict[str, object] = {}
    if "meta_json" in data:
        try:
            meta = json.loads(str(data["meta_json"]))
        except Exception:
            meta = {"meta_json_parse_error": True}
    return omega, meta


def _ellipse_kernel(
    shape: tuple[int, int],
    *,
    y0: int,
    x0: int,
    scale: float,
    orientation: float,
    anisotropy: float,
) -> np.ndarray:
    """Periodic signed anisotropic Gaussian atom normalized to unit L2."""
    h, w = shape
    yy, xx = np.indices(shape)
    dy = ((yy - y0 + h // 2) % h) - h // 2
    dx = ((xx - x0 + w // 2) % w) - w // 2

    c = math.cos(orientation)
    s = math.sin(orientation)
    major = c * dx + s * dy
    minor = -s * dx + c * dy
    sigma_major = max(float(scale), 1e-6)
    sigma_minor = max(float(scale) / max(float(anisotropy), 1.0), 1e-6)
    r2 = (major / sigma_major) ** 2 + (minor / sigma_minor) ** 2
    kernel = np.exp(-0.5 * r2)
    kernel -= float(np.mean(kernel))
    norm = float(np.linalg.norm(kernel))
    if norm <= 1e-12:
        return np.zeros(shape, dtype=np.float64)
    return kernel / norm


def _candidate_peaks(residual: np.ndarray, count: int, min_separation: int) -> list[tuple[int, int]]:
    flat_order = np.argsort(np.abs(residual).ravel())[::-1]
    h, w = residual.shape
    chosen: list[tuple[int, int]] = []
    sep2 = max(int(min_separation), 0) ** 2
    for idx in flat_order:
        y = int(idx // w)
        x = int(idx % w)
        ok = True
        for cy, cx in chosen:
            dy = min(abs(y - cy), h - abs(y - cy))
            dx = min(abs(x - cx), w - abs(x - cx))
            if dy * dy + dx * dx < sep2:
                ok = False
                break
        if ok:
            chosen.append((y, x))
            if len(chosen) >= count:
                break
    return chosen


def _orientation_grid(n: int) -> list[float]:
    n = max(int(n), 1)
    return [math.pi * i / n for i in range(n)]


def _best_atom(
    residual: np.ndarray,
    peaks: Iterable[tuple[int, int]],
    *,
    scales: list[float],
    anisotropies: list[float],
    orientations: list[float],
    bits_per_atom: float,
    q: float,
) -> tuple[Atom | None, np.ndarray | None]:
    best_atom: Atom | None = None
    best_kernel: np.ndarray | None = None
    q2 = max(float(q) ** 2, 1e-18)

    for y, x in peaks:
        phase = float(math.atan2(float(residual[y, x]), abs(float(residual[y, x])) + 1e-12))
        for scale in scales:
            for anisotropy in anisotropies:
                for orientation in orientations:
                    kernel = _ellipse_kernel(
                        residual.shape,
                        y0=y,
                        x0=x,
                        scale=scale,
                        orientation=orientation,
                        anisotropy=anisotropy,
                    )
                    denom = float(np.vdot(kernel, kernel).real)
                    if denom <= 1e-12:
                        continue
                    amplitude = float(np.vdot(residual, kernel).real / denom)
                    signed_kernel = amplitude * kernel
                    sse_reduction = float(
                        2.0 * amplitude * np.vdot(residual, kernel).real
                        - amplitude * amplitude * denom
                    )
                    mdl_gain = sse_reduction / q2 - float(bits_per_atom)
                    if best_atom is None or mdl_gain > best_atom.mdl_gain:
                        best_atom = Atom(
                            y=y,
                            x=x,
                            scale=float(scale),
                            orientation=float(orientation),
                            anisotropy=float(anisotropy),
                            sign=1 if amplitude >= 0.0 else -1,
                            amplitude=amplitude,
                            phase=phase,
                            twist=0.0,
                            sse_reduction=sse_reduction,
                            mdl_gain=mdl_gain,
                        )
                        best_kernel = signed_kernel
    return best_atom, best_kernel


def _frame_diagnostics(kernels: list[np.ndarray]) -> dict[str, float | int | bool]:
    if not kernels:
        return {
            "atoms": 0,
            "lower_frame_bound_selected": 0.0,
            "upper_frame_bound_selected": 0.0,
            "condition_number_selected": math.inf,
            "positive_lower_bound_selected": False,
        }
    mat = np.stack([k.ravel() for k in kernels], axis=0)
    gram = mat @ mat.T
    eigvals = np.linalg.eigvalsh(gram)
    lower = float(np.min(eigvals))
    upper = float(np.max(eigvals))
    return {
        "atoms": len(kernels),
        "lower_frame_bound_selected": lower,
        "upper_frame_bound_selected": upper,
        "condition_number_selected": float(upper / max(lower, 1e-18)),
        "positive_lower_bound_selected": bool(lower > 1e-10),
    }


def _metrics(original: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    err = original - reconstruction
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(original) + 1e-12))
    corr = float(np.corrcoef(original.ravel(), reconstruction.ravel())[0, 1])
    return {
        "rel_l2": rel_l2,
        "correlation": corr,
        "sse": float(np.sum(err * err)),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def _write_plot(path: Path, omega: np.ndarray, reconstruction: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = omega - reconstruction
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, field, title in [
        (axes[0], omega, "omega"),
        (axes[1], reconstruction, "atom reconstruction"),
        (axes[2], err, "error"),
    ]:
        im = ax.imshow(field, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    t0 = perf_counter()
    omega, meta = _load_snapshot(args.truth, args.snapshot_index)
    base = smooth2d(omega, args.smooth_k)
    residual = omega - base
    initial_sse = float(np.sum(residual * residual))

    peaks = _candidate_peaks(residual, args.peak_candidates, args.min_separation)
    orientations = _orientation_grid(args.orientations)

    reconstruction = base.copy()
    current_residual = residual.copy()
    atoms: list[Atom] = []
    kernels: list[np.ndarray] = []

    for _ in range(max(int(args.max_atoms), 0)):
        atom, kernel = _best_atom(
            current_residual,
            peaks,
            scales=[float(v) for v in args.scales],
            anisotropies=[float(v) for v in args.anisotropies],
            orientations=orientations,
            bits_per_atom=args.bits_per_atom,
            q=args.q,
        )
        if atom is None or kernel is None or atom.mdl_gain <= 0.0:
            break
        atoms.append(atom)
        kernels.append(kernel / (np.linalg.norm(kernel) + 1e-12))
        reconstruction += kernel
        current_residual -= kernel
        peaks = _candidate_peaks(current_residual, args.peak_candidates, args.min_separation)

    atom_sse = float(np.sum(current_residual * current_residual))
    noise_energy_baseline = base.copy()
    baseline_metrics = _metrics(omega, noise_energy_baseline)
    atom_metrics = _metrics(omega, reconstruction)
    frame = _frame_diagnostics(kernels)
    elapsed = perf_counter() - t0

    report = {
        "status": "ok",
        "truth": str(args.truth),
        "snapshot_index": int(args.snapshot_index),
        "meta": meta,
        "config": {
            "smooth_k": int(args.smooth_k),
            "max_atoms": int(args.max_atoms),
            "peak_candidates": int(args.peak_candidates),
            "scales": [float(v) for v in args.scales],
            "anisotropies": [float(v) for v in args.anisotropies],
            "orientations": int(args.orientations),
            "bits_per_atom": float(args.bits_per_atom),
            "q": float(args.q),
            "min_separation": int(args.min_separation),
        },
        "atom_count": len(atoms),
        "initial_residual_sse": initial_sse,
        "final_residual_sse": atom_sse,
        "residual_sse_reduction": float(initial_sse - atom_sse),
        "residual_sse_reduction_frac": float((initial_sse - atom_sse) / max(initial_sse, 1e-18)),
        "baseline_smooth_only": baseline_metrics,
        "atom_reconstruction": atom_metrics,
        "selected_frame_diagnostics": frame,
        "atoms": [asdict(atom) for atom in atoms],
        "elapsed_seconds": elapsed,
        "governance": {
            "receipt_alignment": "DASHI.Physics.Closure.AtomExtendedCarrierFrameReceipt",
            "random_phase_replaced": True,
            "phase_field_recorded": True,
            "frame_bound_is_selected_dictionary_only": True,
            "finite_atom_dictionary_is_frame_candidate": True,
            "selected_lower_gram_eigenvalue_is_empirical_A_probe": frame[
                "positive_lower_bound_selected"
            ],
            "lower_frame_bound_A_positive_obligation_proved": False,
            "upper_frame_bound_uniform_B_computed": False,
            "gate3_norm_comparison_proved": False,
            "gate3_proved": False,
            "ns_regularity_claimed": False,
            "clay_claimed": False,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.plot is not None:
        _write_plot(args.plot, omega, reconstruction)
    print(json.dumps({
        "out": str(args.out),
        "atom_count": len(atoms),
        "rel_l2_baseline": baseline_metrics["rel_l2"],
        "rel_l2_atom": atom_metrics["rel_l2"],
        "residual_sse_reduction_frac": report["residual_sse_reduction_frac"],
        "lower_frame_bound_selected": frame["lower_frame_bound_selected"],
        "elapsed_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
