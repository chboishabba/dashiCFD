#!/usr/bin/env python3
"""Extract dyadic shell enstrophy and an EV5 candidate trace.

This adapter is intentionally evidence-only.  It consumes a truth artifact
written by ``scripts/make_truth.py`` (NPZ keys: ``omega_snapshots`` and
``steps``) and writes the files named by the DASHI Agda receipt:

  * ``manifest.json``
  * ``shell_enstrophy.csv``
  * ``ev5_trace.csv``
  * ``checks.json``

The corrected NS->EV5 diagnostic lane semantics are:

  * v2: enstrophy-weighted mean dyadic shell
  * v3: adjacent cascade-flux ratio diagnostic only
  * v5: secondary shell occupancy among the top shells
  * v7: dissipative enstrophy tail above K*(nu)
  * v11: phase/coherence proxy outside the canonical FRACTRAN rules

The checks are diagnostics only.  A reproducible failure falsifies this
encoding or its bucketization for the supplied trace family; it is not a
mathematical theorem, an NS transfer theorem, or Clay evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EV5Row:
    step: int
    time: float
    mean_shell: float
    lane2: int
    lane3: int
    lane5: int
    lane7: int
    lane11: int
    q_log: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, default=None, help="truth NPZ from scripts/make_truth.py")
    p.add_argument("--out-dir", type=Path, required=True, help="directory for manifest/checks/csv outputs")
    p.add_argument("--dt", type=float, default=None, help="override timestep; defaults to meta_json dt or 1")
    p.add_argument("--nu", type=float, default=None, help="override viscosity for K*(nu); defaults to meta_json nu0/nu")
    p.add_argument("--bucket-scale", type=float, default=1000.0, help="quantization scale for EV5 lanes")
    p.add_argument(
        "--lane-mode",
        choices=["ns-ev5", "legacy-jmax-peak"],
        default="ns-ev5",
        help="lane dictionary: corrected ns-ev5 or legacy j_max/peak mapping for comparison",
    )
    p.add_argument(
        "--tail-k-star",
        type=float,
        default=None,
        help="override dissipative cutoff K*(nu); otherwise derived from nu",
    )
    p.add_argument(
        "--tail-k-star-mode",
        choices=["ev5-log", "sqrt-inv-nu", "inv-nu"],
        default="ev5-log",
        help="fallback K*(nu) rule when --tail-k-star is not supplied",
    )
    p.add_argument("--top-shells", type=int, default=5, help="number of nonzero shells used by secondary occupancy")
    p.add_argument("--smoke", action="store_true", help="run against a small synthetic truth trace instead of --truth")
    p.add_argument("--smoke-n", type=int, default=16, help="grid size for --smoke")
    p.add_argument("--smoke-samples", type=int, default=4, help="snapshot count for --smoke")
    p.add_argument("--burn-in", type=int, default=0, help="initial adjacent EV5 transitions ignored by Q checks")
    p.add_argument("--q-tolerance", type=float, default=0.0, help="allowed one-step Q increase before counting a violation")
    p.add_argument(
        "--allowed-q-increases",
        type=int,
        default=0,
        help="allowed number of post-burn-in Q violations before this trace is falsified",
    )
    p.add_argument(
        "--cycle-window",
        type=int,
        default=8,
        help="recent EV5 states to treat as a resonant-cycle warning window",
    )
    p.add_argument(
        "--phase-delta-fraction",
        type=float,
        default=0.5,
        help=(
            "fraction of the observed lane7 drop required before t_star; "
            "if lane7 has no positive drop, t_star is the first checked sample"
        ),
    )
    p.add_argument(
        "--weighted-alpha",
        type=float,
        default=math.log(7.0) / math.log(2.0),
        help="weight alpha in the phase-2 Lyapunov score lane2 + alpha * lane7",
    )
    return p.parse_args()


def _synthetic_truth(samples: int, n: int, dt_override: float | None, nu_override: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    samples = max(1, int(samples))
    n = max(8, int(n))
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    frames = []
    enstrophy = []
    for t in range(samples):
        amp = math.exp(-0.15 * t)
        frame = amp * (np.sin(2.0 * xx + yy) + 0.35 * np.sin(4.0 * yy + 0.5 * t))
        frames.append(frame.astype(np.float64))
        enstrophy.append(0.5 * float(np.mean(frame * frame)))
    meta = {
        "dt": float(1.0 if dt_override is None else dt_override),
        "nu0": float(0.25 if nu_override is None else nu_override),
        "source": "synthetic --smoke trace",
    }
    return (
        np.stack(frames, axis=0),
        np.arange(samples, dtype=np.int64),
        np.asarray(enstrophy, dtype=np.float64),
        meta,
    )


def _load_meta(data: np.lib.npyio.NpzFile, dt_override: float | None) -> dict:
    meta = {}
    if "meta_json" in data.files:
        raw = data["meta_json"]
        try:
            meta = json.loads(str(raw.item() if hasattr(raw, "item") else raw))
        except Exception:
            meta = {}
    if dt_override is not None:
        meta["dt"] = float(dt_override)
    meta.setdefault("dt", 1.0)
    return meta


def _effective_nu(meta: dict, nu_override: float | None) -> float | None:
    if nu_override is not None:
        return float(nu_override)
    for key in ("nu0", "nu", "viscosity"):
        value = meta.get(key)
        if value is not None:
            return float(value)
    return None


def _tail_k_star(nu: float | None, override: float | None, mode: str) -> int:
    if override is not None:
        k_star = int(math.floor(float(override)))
    else:
        if nu is None or not math.isfinite(float(nu)) or float(nu) <= 0.0:
            raise SystemExit(
                "corrected --lane-mode ns-ev5 requires --nu, meta_json nu0/nu/viscosity, "
                "or explicit --tail-k-star for dissipative-tail lane7"
            )
        if mode == "ev5-log":
            k_star = int(math.floor(0.75 * math.log2(1.0 / float(nu))))
        elif mode == "inv-nu":
            k_star = int(math.floor(1.0 / float(nu)))
        else:
            k_star = int(math.floor(1.0 / math.sqrt(float(nu))))
    if k_star < 0:
        raise SystemExit(f"K*(nu) must be a finite nonnegative shell index, got {k_star!r}")
    return k_star


def _tail_shell_start(k_star: int) -> int:
    return max(0, int(k_star))


def _dyadic_shells(n: int) -> np.ndarray:
    freqs = np.fft.fftfreq(n) * n
    ky, kx = np.meshgrid(freqs, freqs, indexing="ij")
    radius = np.sqrt(kx * kx + ky * ky)
    shells = np.zeros_like(radius, dtype=np.int64)
    mask = radius >= 1.0
    shells[mask] = np.floor(np.log2(radius[mask])).astype(np.int64)
    return shells


def shell_enstrophy(omega: np.ndarray, shells: np.ndarray) -> np.ndarray:
    omega_hat = np.fft.fft2(omega)
    # Parseval-normalized shell energy for vorticity.  The constant scale is
    # irrelevant for EV5 bucket comparisons but keeps values grid-comparable.
    spectral_weight = (np.abs(omega_hat) ** 2) / float(omega.size * omega.size)
    max_shell = int(shells.max())
    out = np.zeros(max_shell + 1, dtype=np.float64)
    for j in range(max_shell + 1):
        out[j] = float(spectral_weight[shells == j].sum())
    return out


def _bucket(value: float, scale: float) -> int:
    if not math.isfinite(value) or value <= 0.0:
        return 0
    return int(math.floor(value * scale))


def encode_ev5(
    step: int,
    time: float,
    weights: np.ndarray,
    scale: float,
    top_shells: int,
    *,
    lane_mode: str,
    tail_shell_start: int,
) -> EV5Row:
    nonzero = weights.copy()
    if len(nonzero) > 0:
        nonzero[0] = 0.0
    total = float(nonzero.sum())
    if total <= 0.0:
        return EV5Row(step, time, 0.0, 0, 0, 0, 0, 0, 0)

    j_max = int(np.argmax(nonzero))
    peak = float(nonzero[j_max])
    next_weight = float(nonzero[j_max + 1]) if j_max + 1 < len(nonzero) else 0.0
    ratio = next_weight / max(peak, 1e-300)

    top_count = max(1, min(int(top_shells), len(nonzero)))
    top_indices = np.argpartition(nonzero, -top_count)[-top_count:]
    secondary = float(nonzero[top_indices].sum() - peak) / max(total, 1e-300)

    omega = np.exp(2j * np.pi / 3.0)
    phases = np.array([omega ** (j % 3) for j in range(len(nonzero))])
    coherence = abs(complex(np.sum(nonzero * phases))) / max(total, 1e-300)

    mean_shell = float(np.dot(np.arange(len(nonzero), dtype=np.float64), nonzero) / max(total, 1e-300))
    tail = float(nonzero[tail_shell_start:].sum()) / max(total, 1e-300) if tail_shell_start < len(nonzero) else 0.0

    if lane_mode == "legacy-jmax-peak":
        lane2 = max(j_max, 0)
        lane7 = _bucket(peak / max(total, 1e-300), scale)
        q_log = lane2 + lane7
    else:
        lane2 = _bucket(mean_shell, scale)
        lane7 = _bucket(tail, scale)
        q_log = lane2 + lane7

    lane3 = _bucket(ratio, scale)
    lane5 = _bucket(secondary, scale)
    lane11 = _bucket(coherence, scale)
    return EV5Row(step, time, mean_shell, lane2, lane3, lane5, lane7, lane11, q_log)


def _state(row: EV5Row) -> tuple[int, int, int, int, int]:
    return (row.lane2, row.lane3, row.lane5, row.lane7, row.lane11)


def _nonincreasing_violations(values: list[float], tolerance: float) -> int:
    tol = max(float(tolerance), 0.0)
    return sum(1 for a, b in zip(values, values[1:]) if b > a + tol)


def _vector_ev5_status(rows: list[EV5Row], k_star: int) -> dict:
    lane7_values = [int(row.lane7) for row in rows]
    mean_shell_values = [float(row.mean_shell) for row in rows]
    lane7_increases = sum(1 for a, b in zip(lane7_values, lane7_values[1:]) if b > a)
    bound = float(int(k_star) + 1)
    mean_shell_exceedances = sum(1 for value in mean_shell_values if value > bound)
    v7_decreasing = lane7_increases == 0
    v2_bounded = mean_shell_exceedances == 0
    return {
        "K_star": int(k_star),
        "K_star_definition": "floor(3/4 * log2(1/nu)) unless explicitly overridden",
        "v7_decreasing": bool(v7_decreasing),
        "v7_decreasing_definition": "all(diff(lane7) <= 0)",
        "v2_bounded": bool(v2_bounded),
        "v2_bounded_definition": "all(mean_shell <= K_star + 1)",
        "v2_bound": bound,
        "ev5_admissible": bool(v7_decreasing and v2_bounded),
        "lane7_increases": int(lane7_increases),
        "mean_shell_exceedances": int(mean_shell_exceedances),
        "mean_shell_max": float(max(mean_shell_values)) if mean_shell_values else None,
        "lane7_first": int(lane7_values[0]) if lane7_values else None,
        "lane7_last": int(lane7_values[-1]) if lane7_values else None,
    }


def _two_phase_diagnostic(
    rows: list[EV5Row],
    *,
    burn_in: int,
    q_tolerance: float,
    phase_delta_fraction: float,
    weighted_alpha: float,
) -> dict:
    start = min(max(int(burn_in), 0), max(len(rows) - 1, 0))
    alpha = float(weighted_alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise SystemExit(f"--weighted-alpha must be finite and positive, got {weighted_alpha!r}")

    delta_fraction = float(phase_delta_fraction)
    if not math.isfinite(delta_fraction):
        raise SystemExit(f"--phase-delta-fraction must be finite, got {phase_delta_fraction!r}")
    delta_fraction = min(max(delta_fraction, 0.0), 1.0)

    if not rows:
        return {
            "phase_delta_fraction": delta_fraction,
            "weighted_alpha": alpha,
            "t_star_index": None,
            "t_star_step": None,
            "t_star_time": None,
            "phase1_lane7_ok": True,
            "phase2_weighted_ok": True,
            "two_phase_ok": True,
            "phase1_lane7_violations": 0,
            "phase2_weighted_increases": 0,
            "weighted_definition": "lane2 + weighted_alpha * lane7",
        }

    lane7 = [float(row.lane7) for row in rows]
    weighted = [float(row.lane2) + alpha * float(row.lane7) for row in rows]

    phase_lane7 = lane7[start:]
    if phase_lane7:
        initial_lane7 = phase_lane7[0]
        min_lane7 = min(phase_lane7)
        observed_drop = max(initial_lane7 - min_lane7, 0.0)
    else:
        initial_lane7 = 0.0
        observed_drop = 0.0

    if observed_drop <= 0.0:
        t_star_index = start
        target_lane7 = initial_lane7
    else:
        target_lane7 = initial_lane7 - delta_fraction * observed_drop
        t_star_index = next(
            (
                idx
                for idx in range(start, len(rows))
                if lane7[idx] <= target_lane7
            ),
            len(rows) - 1,
        )

    phase1_values = lane7[start : t_star_index + 1]
    phase2_values = weighted[t_star_index:]
    phase1_violations = _nonincreasing_violations(phase1_values, 0.0)
    phase2_increases = _nonincreasing_violations(phase2_values, q_tolerance)
    phase1_ok = phase1_violations == 0
    phase2_ok = phase2_increases == 0

    return {
        "phase_delta_fraction": delta_fraction,
        "weighted_alpha": alpha,
        "weighted_definition": "lane2 + weighted_alpha * lane7",
        "t_star_index": int(t_star_index),
        "t_star_step": int(rows[t_star_index].step),
        "t_star_time": float(rows[t_star_index].time),
        "t_star_lane7_target": float(target_lane7),
        "t_star_rule": (
            "first post-burn-in sample whose lane7 reaches the requested fraction "
            "of the observed lane7 drop; if no positive lane7 drop is observed, "
            "the first post-burn-in sample is used"
        ),
        "phase1_lane7_ok": phase1_ok,
        "phase2_weighted_ok": phase2_ok,
        "two_phase_ok": bool(phase1_ok and phase2_ok),
        "phase1_lane7_violations": int(phase1_violations),
        "phase2_weighted_increases": int(phase2_increases),
        "phase1_index_range": [int(start), int(t_star_index)],
        "phase2_index_range": [int(t_star_index), int(len(rows) - 1)],
        "weighted_first": float(phase2_values[0]) if phase2_values else None,
        "weighted_last": float(phase2_values[-1]) if phase2_values else None,
    }


def _normalization_guard(shell_rows: list[np.ndarray], truth_enstrophy: np.ndarray | None) -> dict:
    if truth_enstrophy is None or len(truth_enstrophy) == 0:
        return {
            "available": False,
            "passed": None,
            "max_abs_error": None,
            "max_rel_error": None,
            "comparison": "truth artifact has no enstrophy array",
        }

    n = min(len(shell_rows), int(len(truth_enstrophy)))
    if n == 0:
        return {
            "available": False,
            "passed": None,
            "max_abs_error": None,
            "max_rel_error": None,
            "comparison": "empty shell or truth enstrophy rows",
        }

    # shell_enstrophy uses sum |fft2(omega)|^2 / N^4 = mean(omega^2).
    # dashi_cfd_operator_v4.enstrophy uses 0.5 * mean(omega^2).
    shell_total = np.asarray([0.5 * float(row.sum()) for row in shell_rows[:n]], dtype=np.float64)
    truth = np.asarray(truth_enstrophy[:n], dtype=np.float64)
    abs_err = np.abs(shell_total - truth)
    rel_err = abs_err / np.maximum(np.abs(truth), 1e-300)
    max_abs = float(abs_err.max(initial=0.0))
    max_rel = float(rel_err.max(initial=0.0))
    return {
        "available": True,
        "passed": bool(max_rel <= 1e-8 or max_abs <= 1e-12),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "comparison": "0.5 * sum_j shell_enstrophy[j] vs truth enstrophy",
    }


def checks(
    rows: list[EV5Row],
    shell_rows: list[np.ndarray],
    *,
    lane_mode: str,
    cycle_window: int,
    burn_in: int,
    q_tolerance: float,
    allowed_q_increases: int,
    truth_enstrophy: np.ndarray | None,
    phase_delta_fraction: float,
    weighted_alpha: float,
    k_star: int,
) -> dict:
    normalization_guard = _normalization_guard(shell_rows, truth_enstrophy)
    vector_status = _vector_ev5_status(rows, k_star)
    phase_checks = _two_phase_diagnostic(
        rows,
        burn_in=burn_in,
        q_tolerance=q_tolerance,
        phase_delta_fraction=phase_delta_fraction,
        weighted_alpha=weighted_alpha,
    )
    q_definition = "lane2 + lane7; lane3 is cascade-flux diagnostic only"
    if len(rows) < 2:
        normalization_failed = normalization_guard["passed"] is False
        return {
            "samples": len(rows),
            "lane_mode": lane_mode,
            "q_definition": q_definition,
            "burn_in": int(burn_in),
            "q_tolerance": float(q_tolerance),
            "allowed_q_increases": int(allowed_q_increases),
            "vector_ev5": vector_status,
            "ev5_admissible": vector_status["ev5_admissible"],
            "monotone_q_log": True,
            "monotone_energy": True,
            "q_increases": 0,
            "r17_tail_nonincreasing": True,
            "lane7_increases": 0,
            "cycle_warnings": 0,
            "shell_contradictions": 0,
            "normalization_guard": normalization_guard,
            **phase_checks,
            "evaluation_status": "falsified" if normalization_failed else "passed_or_inconclusive",
            "falsified": bool(normalization_failed),
        }

    start = max(0, int(burn_in))
    transitions = list(zip(rows, rows[1:]))
    checked_transitions = transitions[start:]
    q_tol = max(float(q_tolerance), 0.0)
    q_increases = sum(1 for a, b in checked_transitions if b.q_log > a.q_log + q_tol)
    lane7_increases = sum(1 for a, b in checked_transitions if b.lane7 > a.lane7)
    shell_contradictions = sum(
        1
        for a, b in checked_transitions
        if b.q_log <= a.q_log + q_tol and b.lane2 > a.lane2
    )

    window = max(0, int(cycle_window))
    seen: dict[tuple[int, int, int, int, int], int] = {}
    cycle_warnings = 0
    for idx, row in enumerate(rows):
        state = _state(row)
        prev = seen.get(state)
        if prev is not None and (window == 0 or idx - prev <= window):
            cycle_warnings += 1
        seen[state] = idx

    normalization_failed = normalization_guard["passed"] is False
    falsified = normalization_failed or (
        q_increases > max(int(allowed_q_increases), 0)
        or shell_contradictions > 0
    )
    return {
        "samples": len(rows),
        "lane_mode": lane_mode,
        "q_definition": q_definition,
        "burn_in": int(burn_in),
        "q_tolerance": q_tol,
        "allowed_q_increases": int(allowed_q_increases),
        "vector_ev5": vector_status,
        "ev5_admissible": vector_status["ev5_admissible"],
        "monotone_q_log": q_increases == 0,
        "monotone_energy": q_increases == 0,
        "q_increases": q_increases,
        "r17_tail_nonincreasing": lane7_increases == 0,
        "lane7_increases": lane7_increases,
        "cycle_window": window,
        "cycle_warnings": cycle_warnings,
        "cycle_warning_status": (
            "diagnostic only in a single trace; falsifying resonant-cycle status requires "
            "a forced/no-forcing comparison manifest"
        ),
        "shell_contradictions": shell_contradictions,
        "normalization_guard": normalization_guard,
        **phase_checks,
        "evaluation_status": "falsified" if falsified else "passed_or_inconclusive",
        "falsified": bool(falsified),
        "falsification_rule": (
            "Preserve failures.  For a reproducible trace family whose shell-normalization guard "
            "fails, the report fails closed.  Otherwise this candidate encoding is falsified if "
            "post-burn-in Q_log has more than the allowed tolerated increases, if v2 shell movement "
            "contradicts encoded descent, or if "
            "a forced-run EV5 cycle survives the no-forcing/control comparison.  This single-trace "
            "adapter flags cycle warnings but does not promote them to resonant-cycle falsifiers "
            "without the comparison manifest.  This is evidence-only and does not prove or "
            "disprove Navier-Stokes regularity."
        ),
    }


def write_shell_csv(path: Path, steps: Iterable[int], dt: float, shell_rows: list[np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "j", "enstrophy_weight"])
        for step, weights in zip(steps, shell_rows):
            time = float(step) * dt
            for j, value in enumerate(weights):
                writer.writerow([int(step), f"{time:.12g}", int(j), f"{float(value):.17g}"])


def write_ev5_csv(path: Path, rows: list[EV5Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [
            "step", "time", "mean_shell", "lane2", "lane3", "lane5", "lane7", "lane11", "q_log"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        snapshots, steps, truth_enstrophy, meta = _synthetic_truth(args.smoke_samples, args.smoke_n, args.dt, args.nu)
        source_truth = "synthetic --smoke trace"
    else:
        if args.truth is None:
            raise SystemExit("provide --truth or use --smoke")
        data = np.load(args.truth, allow_pickle=False)
        if "omega_snapshots" not in data.files or "steps" not in data.files:
            raise SystemExit("--truth must contain omega_snapshots and steps arrays")
        snapshots = data["omega_snapshots"]
        steps = data["steps"].astype(np.int64)
        meta = _load_meta(data, args.dt)
        truth_enstrophy = data["enstrophy"] if "enstrophy" in data.files else None
        source_truth = str(args.truth)
    if snapshots.ndim != 3:
        raise SystemExit(f"omega_snapshots must be rank 3, got shape {snapshots.shape}")

    dt = float(meta.get("dt", 1.0))
    nu = _effective_nu(meta, args.nu)
    tail_k_star = _tail_k_star(nu, args.tail_k_star, args.tail_k_star_mode)
    tail_start = _tail_shell_start(tail_k_star)
    shells = _dyadic_shells(int(snapshots.shape[1]))
    shell_rows = [shell_enstrophy(frame.astype(np.float64), shells) for frame in snapshots]
    ev5_rows = [
        encode_ev5(
            int(step),
            float(step) * dt,
            weights,
            args.bucket_scale,
            args.top_shells,
            lane_mode=args.lane_mode,
            tail_shell_start=tail_start,
        )
        for step, weights in zip(steps, shell_rows)
    ]

    shell_csv = args.out_dir / "shell_enstrophy.csv"
    ev5_csv = args.out_dir / "ev5_trace.csv"
    checks_json = args.out_dir / "checks.json"
    manifest_json = args.out_dir / "manifest.json"

    write_shell_csv(shell_csv, steps, dt, shell_rows)
    write_ev5_csv(ev5_csv, ev5_rows)

    result_checks = checks(
        ev5_rows,
        shell_rows,
        lane_mode=args.lane_mode,
        cycle_window=args.cycle_window,
        burn_in=args.burn_in,
        q_tolerance=args.q_tolerance,
        allowed_q_increases=args.allowed_q_increases,
        truth_enstrophy=truth_enstrophy,
        phase_delta_fraction=args.phase_delta_fraction,
        weighted_alpha=args.weighted_alpha,
        k_star=tail_k_star,
    )
    checks_json.write_text(json.dumps(result_checks, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "source_truth": source_truth,
        "snapshot_count": int(len(snapshots)),
        "grid_size": int(snapshots.shape[1]),
        "dt": dt,
        "nu": nu,
        "bucket_scale": float(args.bucket_scale),
        "lane_mode": args.lane_mode,
        "tail_k_star": float(tail_k_star),
        "tail_k_star_mode": args.tail_k_star_mode,
        "tail_shell_start": int(tail_start),
        "tail_unresolved": bool(tail_start >= len(shell_rows[0]) if shell_rows else True),
        "top_shells": int(args.top_shells),
        "burn_in": int(args.burn_in),
        "q_tolerance": float(args.q_tolerance),
        "allowed_q_increases": int(args.allowed_q_increases),
        "cycle_window": int(args.cycle_window),
        "phase_delta_fraction": float(args.phase_delta_fraction),
        "weighted_alpha": float(args.weighted_alpha),
        "normalization": "Parseval-normalized |fft2(omega)|^2 / N^4 per dyadic shell",
        "ev5_semantics": {
            "v2": "enstrophy-weighted mean dyadic shell; lane2 is its bucket (legacy mode: active j_max)",
            "v3": "adjacent cascade-flux ratio E[j_max+1]/E[j_max], diagnostic only and excluded from Q_log",
            "v5": "secondary top-shell occupancy",
            "v7": "dissipative enstrophy tail fraction above K*(nu)=floor(3/4*log2(1/nu)) (legacy mode: peak-shell fraction)",
            "v11": "Z/3 phase-coherence proxy outside canonical FRACTRAN rules",
            "Q_log": "v2 + v7",
            "vector_criterion": "ev5_admissible = all(diff(lane7)<=0) and all(mean_shell <= K_star + 1)",
        },
        "lane_boundary": (
            "Projection dictionary and numerical diagnostics only; not an EV5 transfer theorem, "
            "not an actual-flow Navier-Stokes estimate, and not Clay evidence."
        ),
        "outputs": {
            "shell_enstrophy_csv": str(shell_csv),
            "ev5_trace_csv": str(ev5_csv),
            "checks_json": str(checks_json),
        },
        "evidence_boundary": "2D CFD evidence only; no 3D NS proof or Clay promotion.",
        "source_meta": meta,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[ns-ev5] wrote {manifest_json}, {shell_csv}, {ev5_csv}, {checks_json}")
    vector_status = result_checks["vector_ev5"]
    print(
        "[ns-ev5] vector "
        f"K_star={vector_status['K_star']} "
        f"v7_decreasing={vector_status['v7_decreasing']} "
        f"v2_bounded={vector_status['v2_bounded']} "
        f"ev5_admissible={vector_status['ev5_admissible']}"
    )


if __name__ == "__main__":
    main()
