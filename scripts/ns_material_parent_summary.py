#!/usr/bin/env python3
"""Sprint 49 material-parent lineage summary producer.

This is a lightweight producer for the CLI-facing artifact contract required by
`dashi_agda` replay. It extracts coarse material packets at each
`(time, shell)` and computes parent links from one snapshot to the next using
advected centroid prediction, overlap/mass similarity, direction cosine, and
shell adjacency.

The producer writes three files under `--out-dir`:

* `ns_material_parent_table.csv`
* `ns_material_parent_summary.csv`
* `ns_material_parent_summary.json`

The implementation intentionally keeps packets coarse (voxel-bin packets) so it
can run on medium grids (N32/N64) and provide a falsification signal before
committing to a denser lineage reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-30


@dataclass(frozen=True)
class TableRow:
    time: float
    dt: float
    K_parent: int
    K_child: int
    child_packet_id: str
    parent_packet_id: str
    child_state: str
    parent_state: str
    child_mass: float
    parent_mass: float
    credited_mass: float
    source_true_new: float
    source_tracking_uncertain: float
    source_cross_shell: float
    source_low_shell_injection: float
    advected_overlap: float
    centroid_distance: float
    direction_cosine: float
    shell_delta: int
    parent_confidence: float
    parent_relation: str
    classification: str


@dataclass(frozen=True)
class SummaryRow:
    time: float
    K_child: int
    M_plus_plus_material: float
    source_true_new: float
    source_tracking_uncertain: float
    source_cross_shell: float
    source_low_shell_injection: float
    source_total_material: float
    weighted_true_new: float
    weighted_tracking_uncertain: float
    weighted_cross_shell: float
    weighted_low_shell_injection: float
    weighted_total_material: float
    sigma_true_new_fit: float
    sigma_tracking_uncertain_fit: float
    sigma_cross_shell_fit: float
    sigma_low_shell_fit: float
    sigma_total_material_fit: float
    route_status: str


@dataclass(frozen=True)
class Packet:
    time: float
    K: int
    packet_id: str
    state: str
    mass: float
    centroid: np.ndarray
    velocity: np.ndarray
    plus_mass: float
    minus_mass: float
    zero_mass: float


@dataclass(frozen=True)
class MatchCandidate:
    parent: Packet
    child: Packet
    score: float
    overlap: float
    centroid_distance: float
    direction_cosine: float
    shell_delta: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, required=True, help="truth NPZ with omega_snapshots and steps")
    p.add_argument("--out-dir", type=Path, required=True, help="output directory")
    p.add_argument("--dt", type=float, default=None, help="override dt in seconds")
    p.add_argument("--packet-grid", type=int, default=8, help="coarse packet lattice per axis")
    p.add_argument("--packet-active-quantile", type=float, default=0.90, help="mass quantile for active bins")
    p.add_argument("--min-packet-mass", type=float, default=1e-16, help="minimum bin mass floor")
    p.add_argument(
        "--state-threshold",
        type=float,
        default=0.0,
        help="stretching-threshold to classify packet state (plus/zero/minus)",
    )
    p.add_argument("--match-distance-scale", type=float, default=0.15, help="distance scale in units of domain")
    p.add_argument("--match-score-floor", type=float, default=0.10, help="minimum score to keep parent candidate")
    p.add_argument("--match-tracking-threshold", type=float, default=0.45, help="below this score is tracking_uncertain")
    p.add_argument("--match-secondary-ratio", type=float, default=0.80, help="secondary candidates kept above this fraction")
    p.add_argument("--parent-overlap-weight", type=float, default=0.55, help="score weight for centroid overlap")
    p.add_argument("--parent-mass-weight", type=float, default=0.25, help="score weight for parent/child mass matching")
    p.add_argument("--parent-direction-weight", type=float, default=0.10, help="score weight for advection-direction cosine")
    p.add_argument("--parent-shell-weight", type=float, default=0.10, help="score bonus for shell adjacency")
    p.add_argument(
        "--backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help="requested backend for material-parent producer",
    )
    p.add_argument(
        "--fft-backend",
        type=str,
        default="vkfft-vulkan",
        help="GPU FFT backend used when --backend=gpu",
    )
    p.add_argument(
        "--diagnostic-precision",
        choices=["float32", "float64"],
        default="float64",
        help="GPU spectral derivative precision used when --backend=gpu",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print progress every N snapshots/transitions; use 0 to disable",
    )
    p.add_argument(
        "--low-shell-k-offset",
        type=int,
        default=0,
        help="classification treats child shell <= k_star + offset as low-shell injection",
    )
    return p.parse_args()


def _load_meta(data: np.lib.npyio.NpzFile, dt_override: float | None) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if "meta_json" in data.files:
        raw = data["meta_json"]
        try:
            meta = json.loads(str(raw.item() if hasattr(raw, "item") else raw))
        except Exception:
            meta = {"meta_json_parse_error": True}
    if dt_override is not None:
        meta["dt"] = float(dt_override)
    meta.setdefault("dt", 1.0)
    return meta


def _load_truth(path: Path, dt_override: float | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(path)
    if "omega_snapshots" not in data.files:
        raise SystemExit(f"truth file {path} missing omega_snapshots")
    omega = np.asarray(data["omega_snapshots"], dtype=np.float64)
    if omega.ndim != 5:
        raise SystemExit("omega_snapshots must be [T,N,N,N,3]")
    if omega.shape[-1] != 3:
        raise SystemExit("omega_snapshots must have 3 vector components")
    if "steps" in data.files:
        steps = np.asarray(data["steps"], dtype=np.int64)
    else:
        steps = np.arange(omega.shape[0], dtype=np.int64)
    meta = _load_meta(data, dt_override)
    return omega, steps, meta


def _load_velocity(path: Path) -> np.ndarray | None:
    data = np.load(path)
    if "velocity_snapshots" not in data.files:
        return None
    velocity = np.asarray(data["velocity_snapshots"], dtype=np.float64)
    if velocity.shape != np.load(path)["omega_snapshots"].shape:
        raise SystemExit("velocity_snapshots shape must match omega_snapshots")
    return velocity


def _periodic_distance(a: np.ndarray, b: np.ndarray, L: float) -> float:
    d = np.abs(a - b)
    d = np.minimum(d, L - d)
    return float(np.linalg.norm(d))


def _periodic_centroid(indices: np.ndarray, L: float) -> float:
    if indices.size == 0:
        return 0.0
    angles = 2.0 * np.pi * (indices + 0.5) / float(indices.max() + 1)
    # For consistency with axis lengths that may differ in practice, use periodic
    # averaging through sine/cosine and then normalize to [0, L).
    c = float(np.mean(np.cos(angles)))
    s = float(np.mean(np.sin(angles)))
    if c == 0.0 and s == 0.0:
        return float((float(indices.mean()) + 0.5) * L / float(indices.max() + 1))
    ang = math.atan2(s, c)
    if ang < 0.0:
        ang += 2.0 * math.pi
    return float(ang * L / (2.0 * math.pi))


def _periodic_weighted_centroid(coords: np.ndarray, weights: np.ndarray, n: int, L: float) -> np.ndarray:
    if coords.size == 0:
        return np.zeros(3, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if np.allclose(weights.sum(), 0.0):
        return (coords + 0.5) * (L / float(n))
    out = np.empty(3, dtype=np.float64)
    for axis in range(3):
        vals = coords[:, axis]
        theta = 2.0 * np.pi * (vals + 0.5) / float(n)
        w = weights
        c = float(np.dot(w, np.cos(theta)) / np.dot(w, np.ones_like(w)))
        s = float(np.dot(w, np.sin(theta)) / np.dot(w, np.ones_like(w)))
        if c == 0.0 and s == 0.0:
            out[axis] = float((float(np.dot(w, vals)) / max(float(np.dot(w, np.ones_like(w))), EPS) + 0.5) * L / float(n))
        else:
            ang = math.atan2(s, c)
            if ang < 0.0:
                ang += 2.0 * math.pi
            out[axis] = float(ang * L / (2.0 * math.pi))
    return out


def _direction_cosine(parent_velocity: np.ndarray, parent_to_child: np.ndarray) -> float:
    parent_speed = float(np.linalg.norm(parent_velocity))
    disp_norm = float(np.linalg.norm(parent_to_child))
    if parent_speed <= 0.0 or disp_norm <= 0.0:
        return 0.0
    pc = parent_velocity / max(parent_speed, EPS)
    dc = parent_to_child / max(disp_norm, EPS)
    return float(np.clip(np.dot(pc, dc), -1.0, 1.0))


def _build_shell_map(n: int, L: float) -> np.ndarray:
    dk = np.fft.fftfreq(n, d=L / float(n)) * 2.0 * math.pi
    kz, ky, kx = np.meshgrid(dk, dk, dk, indexing="ij")
    radius = np.sqrt(kx * kx + ky * ky + kz * kz)
    shells = np.zeros_like(radius, dtype=np.int64)
    mask = radius >= 1.0
    shells[mask] = np.floor(radius[mask]).astype(np.int64)
    return shells


def _build_velocity_gradient(u: np.ndarray, L: float) -> np.ndarray:
    # shape: (N,N,N,3,3), grad[...,i,j] = ∂u_i/∂x_j
    h = L / float(u.shape[0])
    inv_two_h = 0.5 / h
    grad = np.empty(u.shape + (3,), dtype=np.float64)
    for comp in range(3):
        grad_comp = grad[..., comp, :]
        for axis in range(3):
            grad_comp[..., axis] = (np.roll(u[..., comp], -1, axis=axis) - np.roll(u[..., comp], 1, axis=axis)) * inv_two_h
        grad[..., comp, :] = grad_comp
    return grad


def _make_gpu_diagnostic_backend(n: int, L: float, fft_backend: str, precision: str) -> Any:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from vulkan_truth3d_backend import VulkanSpectralDiagnostic3DBackend  # type: ignore
    except Exception as exc:
        raise SystemExit(f"--backend=gpu could not import Vulkan diagnostic backend: {exc}") from exc

    backend = VulkanSpectralDiagnostic3DBackend(
        n,
        length=L,
        fft_backend=fft_backend,
        precision=precision,
        timing_enabled=True,
    )
    runtime = dict(backend.runtime_info())
    if runtime.get("fft_plan_backend") != "vulkan" or runtime.get("ifft_plan_backend") != "vulkan":
        backend.close()
        raise SystemExit(f"--backend=gpu fell back from Vulkan vkFFT: {runtime}")
    return backend


def _build_velocity_gradient_gpu(backend: Any, u: np.ndarray) -> np.ndarray:
    backend.set_velocity_real(u, project=False, dealias=False)
    derivs = backend.read_velocity_derivatives()
    grad = np.empty(u.shape + (3,), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            grad[..., i, j] = np.asarray(derivs[(i, j)], dtype=np.float64)
    return grad


def _packet_state_counts(mass: np.ndarray, sign_mask_plus: np.ndarray, sign_mask_minus: np.ndarray) -> tuple[float, float, float]:
    pm = mass[sign_mask_plus].sum()
    mm = mass[sign_mask_minus].sum()
    zm = mass[~(sign_mask_plus | sign_mask_minus)].sum()
    return float(pm), float(mm), float(zm)


def _classify_packet_state(pm: float, mm: float, zm: float) -> str:
    if pm >= zm and pm >= mm:
        return "plus"
    if mm >= pm and mm >= zm:
        return "minus"
    if zm >= pm and zm >= mm:
        return "zero"
    return "zero"


def _extract_packets(
    omega: np.ndarray,
    stretch_state: np.ndarray | None,
    velocity: np.ndarray | None,
    shell_map: np.ndarray,
    time: float,
    packet_grid: int,
    packet_active_quantile: float,
    min_packet_mass: float,
    state_threshold: float,
    n: int,
    L: float,
) -> list[Packet]:
    del state_threshold
    v2 = np.sum(omega * omega, axis=-1)
    if stretch_state is None:
        stretch_state = np.zeros_like(v2, dtype=np.int8)

    packets: list[Packet] = []
    stride = max(1, int(n / packet_grid))
    max_cell = int(packet_grid ** 3)

    # Precompute for each shell to avoid touching absent shells repeatedly.
    shell_ids = np.unique(shell_map)
    for K in shell_ids:
        shell_mask = shell_map == K
        if not np.any(shell_mask):
            continue
        idx = np.nonzero(shell_mask)
        if len(idx[0]) == 0:
            continue
        flat_mass = v2[shell_mask]
        if not np.any(flat_mass > 0.0):
            continue

        i = idx[0]
        j = idx[1]
        k = idx[2]
        cell = (i // stride) * packet_grid * packet_grid + (j // stride) * packet_grid + (k // stride)
        flat_cell_mass = np.zeros(max_cell, dtype=np.float64)
        np.add.at(flat_cell_mass, cell, flat_mass)

        nonzero_cells = flat_cell_mass > 0.0
        if not np.any(nonzero_cells):
            continue
        nz = flat_cell_mass[nonzero_cells]
        q = np.quantile(nz, packet_active_quantile)
        threshold = max(float(q), float(min_packet_mass))

        active_cells = np.where((flat_cell_mass >= threshold) & nonzero_cells)[0]
        if active_cells.size == 0:
            # keep the strongest support if quantile is too aggressive
            top = int(np.argmax(flat_cell_mass))
            active_cells = np.array([top], dtype=np.int64)

        coords = np.stack([i, j, k], axis=1)
        for cidx in active_cells:
            sel = cell == cidx
            if not bool(np.any(sel)):
                continue
            packet_mass = float(flat_cell_mass[cidx])
            if packet_mass <= 0.0:
                continue
            point_mass = flat_mass[sel]
            point_coords = coords[sel]
            state = stretch_state[shell_mask][sel]

            plus_mask = state > 0
            minus_mask = state < 0
            pm, mm, zm = _packet_state_counts(point_mass, plus_mask, minus_mask)
            st = _classify_packet_state(pm, mm, zm)

            centroid = _periodic_weighted_centroid(point_coords.astype(np.float64), point_mass, n, L)
            if velocity is None:
                vel = np.zeros(3, dtype=np.float64)
            else:
                vel_pts = velocity[shell_mask][sel]
                vel_w = np.asarray(vel_pts, dtype=np.float64)
                vel_mass = np.dot(point_mass, np.ones((vel_w.shape[0],), dtype=np.float64))
                vel = np.dot(point_mass, vel_w) / max(vel_mass, EPS)

            packet = Packet(
                time=time,
                K=int(K),
                packet_id=f"K{int(K)}_cell{int(cidx)}",
                state=st,
                mass=packet_mass,
                centroid=centroid,
                velocity=vel,
                plus_mass=pm,
                minus_mass=mm,
                zero_mass=zm,
            )
            packets.append(packet)
    return packets


def _match_candidates(
    parent_packets: list[Packet],
    child_packets: list[Packet],
    dt: float,
    L: float,
    min_score: float,
    secondary_ratio: float,
    weights: dict[str, float],
) -> tuple[list[MatchCandidate], dict[str, int], dict[str, int]]:
    if not parent_packets or not child_packets:
        return [], {}, {}

    candidates: list[MatchCandidate] = []
    parent_to_children: dict[str, int] = {p.packet_id: 0 for p in parent_packets}
    child_to_parents: dict[str, int] = {c.packet_id: 0 for c in child_packets}

    dist_scale = max(EPS, float(L) * float(weights["match_distance_scale"]))
    parent_centroids = np.asarray([p.centroid for p in parent_packets], dtype=np.float64)
    parent_velocities = np.asarray([p.velocity for p in parent_packets], dtype=np.float64)
    parent_masses = np.asarray([p.mass for p in parent_packets], dtype=np.float64)
    parent_shells = np.asarray([p.K for p in parent_packets], dtype=np.int64)
    parent_speeds = np.linalg.norm(parent_velocities, axis=1)
    for child in child_packets:
        advected = np.mod(parent_centroids + parent_velocities * float(dt), float(L))
        delta_xyz = np.abs(advected - child.centroid)
        delta_xyz = np.minimum(delta_xyz, float(L) - delta_xyz)
        dist = np.linalg.norm(delta_xyz, axis=1)
        finite = np.isfinite(dist)
        if not bool(np.any(finite)):
            continue

        overlap = np.exp(-((dist / max(dist_scale, EPS)) ** 2))
        mass_ratio = np.minimum(parent_masses, child.mass) / np.maximum(np.maximum(parent_masses, child.mass), EPS)

        parent_to_child = child.centroid - advected
        disp_norm = np.linalg.norm(parent_to_child, axis=1)
        direction = np.zeros_like(dist)
        direction_mask = (parent_speeds > 0.0) & (disp_norm > 0.0)
        if bool(np.any(direction_mask)):
            direction[direction_mask] = np.einsum(
                "ij,ij->i",
                parent_velocities[direction_mask] / parent_speeds[direction_mask, None],
                parent_to_child[direction_mask] / disp_norm[direction_mask, None],
            )
            direction = np.clip(direction, -1.0, 1.0)

        shell_delta = np.abs(parent_shells - int(child.K))
        shell_pref = np.where(shell_delta == 0, 1.0, np.where(shell_delta == 1, 0.75, np.where(shell_delta == 2, 0.45, 0.2)))
        score = (
            weights["overlap"] * overlap
            + weights["mass"] * mass_ratio
            + weights["direction"] * np.maximum(direction, 0.0)
            + weights["shell"] * shell_pref
        )
        score = np.clip(score, 0.0, 1.0)
        keep = finite & (score >= min_score)
        if not bool(np.any(keep)):
            continue

        kept_indices = np.nonzero(keep)[0]
        kept_order = kept_indices[np.argsort(score[kept_indices])[::-1]]
        best = float(score[kept_order[0]])
        threshold = max(float(min_score), float(secondary_ratio) * best)
        selected_indices = kept_order[score[kept_order] >= threshold]
        if selected_indices.size == 0:
            selected_indices = kept_order[:1]
        for idx in selected_indices:
            parent = parent_packets[int(idx)]
            c = MatchCandidate(
                parent=parent,
                child=child,
                score=float(score[idx]),
                overlap=float(overlap[idx]),
                centroid_distance=float(dist[idx]),
                direction_cosine=float(direction[idx]),
                shell_delta=int(shell_delta[idx]),
            )
            candidates.append(c)
            parent_to_children[c.parent.packet_id] = parent_to_children.get(c.parent.packet_id, 0) + 1
            child_to_parents[c.child.packet_id] = child_to_parents.get(c.child.packet_id, 0) + 1

    return candidates, parent_to_children, child_to_parents


def _route_status_from_summary(
    sigma_true_new: float,
    sigma_tracking_uncertain: float,
    sigma_cross_shell: float,
    sigma_low_shell: float,
    weighted_true_new: float,
    weighted_tracking_uncertain: float,
    weighted_cross_shell: float,
    weighted_low_shell: float,
) -> str:
    weighted = {
        "true_new": weighted_true_new,
        "tracking_uncertain": weighted_tracking_uncertain,
        "cross_shell": weighted_cross_shell,
        "low_shell": weighted_low_shell,
    }
    dominant = max(weighted, key=weighted.get)
    if dominant == "tracking_uncertain" and weighted_tracking_uncertain > weighted_true_new:
        return "TRACKING_UNCERTAIN_NEEDS_DENSER_SNAPSHOTS"
    if dominant in {"cross_shell", "low_shell"} and weighted[dominant] > weighted_true_new:
        return "ADJACENT_PACKET_THEOREM_INSUFFICIENT"
    if sigma_true_new <= 0.5:
        return "TRUE_NEW_SOURCE_SUBCRITICAL"
    if sigma_true_new > 0.5 and weighted_tracking_uncertain <= weighted_true_new:
        return "MATERIAL_PARENT_REPAIRS_NEW_SOURCE_DIAGNOSTIC_ONLY"
    return "TRACKING_UNCERTAIN_NEEDS_DENSER_SNAPSHOTS"


def _build_table_rows(
    parent_packets: list[Packet],
    child_packets: list[Packet],
    dt: float,
    L: float,
    k_star: int,
    low_shell_k_offset: int,
    min_score: float,
    secondary_ratio: float,
    match_tracking_threshold: float,
    weights: dict[str, float],
) -> list[TableRow]:
    rows: list[TableRow] = []
    candidates, parent_to_children, child_to_parents = _match_candidates(
        parent_packets,
        child_packets,
        dt,
        L,
        min_score,
        secondary_ratio,
        {
            "overlap": float(weights["parent_overlap_weight"]),
            "mass": float(weights["parent_mass_weight"]),
            "direction": float(weights["parent_direction_weight"]),
            "shell": float(weights["parent_shell_weight"]),
            "match_distance_scale": float(weights["match_distance_scale"]),
        },
    )

    child_ids_with_match = {c.parent.packet_id for c in candidates}
    # build quick lookup for unmatched children
    matched_children = {c.child.packet_id: [] for c in candidates}
    for c in candidates:
        matched_children[c.child.packet_id].append(c)

    for child in child_packets:
        cands = matched_children.get(child.packet_id, [])
        if not cands:
            source_true_new = child.mass if child.state == "plus" and child.K > k_star + low_shell_k_offset else 0.0
            source_low = child.mass if child.state == "plus" and child.K <= k_star + low_shell_k_offset else 0.0
            relation = "low_shell_parent" if source_low > 0.0 else "true_new"
            classification = "low_shell_injection_plus" if source_low > 0.0 else (
                "true_new_plus" if child.state == "plus" else "nonplus_transition"
            )
            rows.append(
                TableRow(
                    time=child.time,
                    dt=float(dt),
                    K_parent=-1,
                    K_child=child.K,
                    child_packet_id=child.packet_id,
                    parent_packet_id="none",
                    child_state=child.state,
                    parent_state="none",
                    child_mass=child.mass,
                    parent_mass=0.0,
                    credited_mass=0.0,
                    source_true_new=source_true_new,
                    source_tracking_uncertain=0.0,
                    source_cross_shell=0.0,
                    source_low_shell_injection=source_low,
                    advected_overlap=0.0,
                    centroid_distance=0.0,
                    direction_cosine=0.0,
                    shell_delta=0,
                    parent_confidence=0.0,
                    parent_relation=relation,
                    classification=classification,
                )
            )
            continue

        denom = max(1, len(cands))
        parent_to_child_mass_share = child.mass / float(denom)
        for cand in cands:
            relation = "advected_parent"
            if cand.score < match_tracking_threshold:
                relation = "tracking_uncertain"
            if relation == "advected_parent":
                if abs(cand.shell_delta) > 0:
                    if cand.parent.K < k_star + low_shell_k_offset:
                        relation = "low_shell_parent"
                    else:
                        relation = "cross_shell_parent"
                elif child_to_parents.get(child.packet_id, 0) > 1:
                    relation = "merge_parent"
                elif parent_to_children.get(cand.parent.packet_id, 0) > 1:
                    relation = "split_parent"

            if child.state == "plus" and cand.parent.state == "plus":
                classification = "plus_to_plus"
            elif child.state == "plus" and cand.parent.state == "zero":
                classification = "zero_to_plus"
            elif child.state == "plus" and cand.parent.state == "minus":
                classification = "minus_to_plus"
            elif child.state == "plus" and relation == "tracking_uncertain":
                classification = "tracking_uncertain_plus"
            elif child.state == "plus" and relation in {"cross_shell_parent", "low_shell_parent"}:
                classification = "cross_shell_plus" if relation == "cross_shell_parent" else "low_shell_injection_plus"
            elif child.state == "plus":
                classification = "nonplus_transition"
            else:
                classification = "nonplus_transition"

            source_true_new = 0.0
            source_tracking_uncertain = 0.0
            source_cross_shell = 0.0
            source_low_shell = 0.0
            if child.state == "plus":
                if relation == "tracking_uncertain":
                    source_tracking_uncertain = parent_to_child_mass_share
                elif relation == "cross_shell_parent":
                    source_cross_shell = parent_to_child_mass_share
                elif relation == "low_shell_parent":
                    source_low_shell = parent_to_child_mass_share
                elif relation == "true_new":
                    source_true_new = parent_to_child_mass_share

            rows.append(
                TableRow(
                    time=child.time,
                    dt=float(dt),
                    K_parent=cand.parent.K,
                    K_child=child.K,
                    child_packet_id=child.packet_id,
                    parent_packet_id=cand.parent.packet_id,
                    child_state=child.state,
                    parent_state=cand.parent.state,
                    child_mass=child.mass,
                    parent_mass=cand.parent.mass,
                    credited_mass=parent_to_child_mass_share,
                    source_true_new=source_true_new,
                    source_tracking_uncertain=source_tracking_uncertain,
                    source_cross_shell=source_cross_shell,
                    source_low_shell_injection=source_low_shell,
                    advected_overlap=cand.overlap,
                    centroid_distance=cand.centroid_distance,
                    direction_cosine=cand.direction_cosine,
                    shell_delta=cand.shell_delta,
                    parent_confidence=cand.score,
                    parent_relation=relation,
                    classification=classification,
                )
            )

    # Ensure every child has at least one row; cands may have empty packet lists in weird degenerate cases
    # already handled above.
    return rows


def _table_csv_rows(rows: list[TableRow]) -> list[dict[str, str]]:
    return [
        {
            "time": f"{r.time:.17g}",
            "dt": f"{r.dt:.17g}",
            "K_parent": str(r.K_parent),
            "K_child": str(r.K_child),
            "child_packet_id": r.child_packet_id,
            "parent_packet_id": r.parent_packet_id,
            "child_state": r.child_state,
            "parent_state": r.parent_state,
            "child_mass": f"{r.child_mass:.17g}",
            "parent_mass": f"{r.parent_mass:.17g}",
            "credited_mass": f"{r.credited_mass:.17g}",
            "source_true_new": f"{r.source_true_new:.17g}",
            "source_tracking_uncertain": f"{r.source_tracking_uncertain:.17g}",
            "source_cross_shell": f"{r.source_cross_shell:.17g}",
            "source_low_shell_injection": f"{r.source_low_shell_injection:.17g}",
            "advected_overlap": f"{r.advected_overlap:.17g}",
            "centroid_distance": f"{r.centroid_distance:.17g}",
            "direction_cosine": f"{r.direction_cosine:.17g}",
            "shell_delta": str(r.shell_delta),
            "parent_confidence": f"{r.parent_confidence:.17g}",
            "parent_relation": r.parent_relation,
            "classification": r.classification,
        }
        for r in rows
    ]


def _summary_csv_rows(rows: list[TableRow]) -> list[dict[str, str]]:
    by_time_k: dict[tuple[float, int], list[TableRow]] = {}
    for row in rows:
        by_time_k.setdefault((row.time, int(row.K_child)), []).append(row)

    out: list[dict[str, str]] = []
    for (time, K), items in sorted(by_time_k.items()):
        plus_children = [r for r in items if r.child_state == "plus"]
        plus_mass = sum(r.child_mass for r in plus_children)
        plus_to_plus = sum(r.credited_mass for r in plus_children if r.classification == "plus_to_plus")
        source_true_new = sum(r.source_true_new for r in items)
        source_tracking = sum(r.source_tracking_uncertain for r in items)
        source_cross = sum(r.source_cross_shell for r in items)
        source_low = sum(r.source_low_shell_injection for r in items)
        source_total = source_true_new + source_tracking + source_cross + source_low

        # Weighted mass uses a standard 2^(K/2) envelope to expose tail behavior.
        w = 2.0 ** (0.5 * float(K))
        weighted_true = source_true_new * w
        weighted_tracking = source_tracking * w
        weighted_cross = source_cross * w
        weighted_low = source_low * w
        weighted_total = weighted_true + weighted_tracking + weighted_cross + weighted_low

        parent_plus = sum(r.parent_mass for r in plus_children if r.classification == "plus_to_plus")
        M_plus_plus_material = plus_to_plus / max(parent_plus, EPS)

        sigma_true_new = source_true_new / max(plus_mass, EPS)
        sigma_tracking = source_tracking / max(plus_mass, EPS)
        sigma_cross = source_cross / max(plus_mass, EPS)
        sigma_low = source_low / max(plus_mass, EPS)
        sigma_total = source_total / max(plus_mass, EPS)

        s = SummaryRow(
            time=float(time),
            K_child=int(K),
            M_plus_plus_material=float(M_plus_plus_material),
            source_true_new=float(source_true_new),
            source_tracking_uncertain=float(source_tracking),
            source_cross_shell=float(source_cross),
            source_low_shell_injection=float(source_low),
            source_total_material=float(source_total),
            weighted_true_new=float(weighted_true),
            weighted_tracking_uncertain=float(weighted_tracking),
            weighted_cross_shell=float(weighted_cross),
            weighted_low_shell_injection=float(weighted_low),
            weighted_total_material=float(weighted_total),
            sigma_true_new_fit=float(sigma_true_new),
            sigma_tracking_uncertain_fit=float(sigma_tracking),
            sigma_cross_shell_fit=float(sigma_cross),
            sigma_low_shell_fit=float(sigma_low),
            sigma_total_material_fit=float(sigma_total),
            route_status=_route_status_from_summary(
                sigma_true_new=sigma_true_new,
                sigma_tracking_uncertain=sigma_tracking,
                sigma_cross_shell=sigma_cross,
                sigma_low_shell=sigma_low,
                weighted_true_new=weighted_true,
                weighted_tracking_uncertain=weighted_tracking,
                weighted_cross_shell=weighted_cross,
                weighted_low_shell=weighted_low,
            ),
        )
        out.append(
            {
                "time": f"{s.time:.17g}",
                "K_child": str(s.K_child),
                "M_plus_plus_material": f"{s.M_plus_plus_material:.17g}",
                "source_true_new": f"{s.source_true_new:.17g}",
                "source_tracking_uncertain": f"{s.source_tracking_uncertain:.17g}",
                "source_cross_shell": f"{s.source_cross_shell:.17g}",
                "source_low_shell_injection": f"{s.source_low_shell_injection:.17g}",
                "source_total_material": f"{s.source_total_material:.17g}",
                "weighted_true_new": f"{s.weighted_true_new:.17g}",
                "weighted_tracking_uncertain": f"{s.weighted_tracking_uncertain:.17g}",
                "weighted_cross_shell": f"{s.weighted_cross_shell:.17g}",
                "weighted_low_shell_injection": f"{s.weighted_low_shell_injection:.17g}",
                "weighted_total_material": f"{s.weighted_total_material:.17g}",
                "sigma_true_new_fit": f"{s.sigma_true_new_fit:.17g}",
                "sigma_tracking_uncertain_fit": f"{s.sigma_tracking_uncertain_fit:.17g}",
                "sigma_cross_shell_fit": f"{s.sigma_cross_shell_fit:.17g}",
                "sigma_low_shell_fit": f"{s.sigma_low_shell_fit:.17g}",
                "sigma_total_material_fit": f"{s.sigma_total_material_fit:.17g}",
                "route_status": s.route_status,
            }
        )

    return out


def main() -> None:
    args = _parse_args()
    if args.packet_grid <= 0:
        raise SystemExit("--packet-grid must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be nonnegative")
    if args.packet_active_quantile < 0.0 or args.packet_active_quantile > 1.0:
        raise SystemExit("--packet-active-quantile must be in [0, 1]")

    omega, steps, meta = _load_truth(args.truth, args.dt)
    velocity = _load_velocity(args.truth)

    if args.backend == "gpu" and velocity is None:
        raise SystemExit("--backend=gpu requires velocity_snapshots for advection-aware matching")

    dt = float(meta.get("dt", 1.0))
    n = int(omega.shape[1])
    L = float(meta.get("domain_length", 2.0 * math.pi))
    k_star = int(meta.get("k_star", 0))

    shell_map = _build_shell_map(n, L)

    # state by stretch sign, positive means plus, negative minus, otherwise zero.
    velfield = velocity if velocity is not None else None
    gpu_backend = None
    gpu_runtime: dict[str, Any] = {}
    gpu_device: dict[str, Any] = {}
    actual_backend = "cpu"
    if args.backend == "gpu":
        print(
            "[ns_material_parent_summary] backend=gpu requested "
            f"fft_backend={args.fft_backend} precision={args.diagnostic_precision}"
        )
        print(f"[ns_material_parent_summary] VK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES', '<unset>')}")
        gpu_backend = _make_gpu_diagnostic_backend(n, L, args.fft_backend, args.diagnostic_precision)
        gpu_runtime = dict(gpu_backend.runtime_info())
        gpu_device = dict(gpu_backend.device_info())
        actual_backend = "gpu_spectral_gradient_cpu_packets"
        print(
            "[ns_material_parent_summary] backend=gpu active "
            f"fft_plan_backend={gpu_runtime.get('fft_plan_backend')} "
            f"device={gpu_device.get('device_name', '<unknown>')}"
        )

    all_time_packets: list[list[Packet]] = []
    t0 = time.perf_counter()
    try:
        for t, frame in enumerate(omega):
            snapshot_time = float(steps[t] * dt) if t < len(steps) else float(t * dt)
            if args.backend == "cpu":
                stretch_state = None
            else:
                if velfield is None:
                    stretch_state = None
                else:
                    u_frame = velfield[t].astype(np.float64)
                    if gpu_backend is None:
                        grad_u = _build_velocity_gradient(u_frame, L)
                    else:
                        grad_u = _build_velocity_gradient_gpu(gpu_backend, u_frame)
                    # stretch_scalar = omega . ((omega . grad) u)
                    stretch = np.einsum("...i,...ij,...j->...", frame.astype(np.float64), grad_u, frame.astype(np.float64))
                    stretch_state = np.zeros(stretch.shape, dtype=np.int8)
                    stretch_state[stretch > args.state_threshold] = 1
                    stretch_state[stretch < -args.state_threshold] = -1

            packets = _extract_packets(
                omega=frame.astype(np.float64),
                stretch_state=stretch_state,
                velocity=velfield[t].astype(np.float64) if velfield is not None else None,
                shell_map=shell_map,
                time=snapshot_time,
                packet_grid=args.packet_grid,
                packet_active_quantile=args.packet_active_quantile,
                min_packet_mass=args.min_packet_mass,
                state_threshold=args.state_threshold,
                n=n,
                L=L,
            )
            all_time_packets.append(packets)
            if args.progress_every and ((t + 1) % args.progress_every == 0 or t + 1 == len(omega)):
                elapsed = time.perf_counter() - t0
                print(
                    "[ns_material_parent_summary] "
                    f"snapshots {t + 1}/{len(omega)} packets={len(packets)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        if gpu_backend is not None:
            gpu_runtime = dict(gpu_backend.runtime_info())
            gpu_backend.close()

    table_rows: list[TableRow] = []
    for t in range(len(all_time_packets) - 1):
        parent_packets = all_time_packets[t]
        child_packets = all_time_packets[t + 1]
        if not parent_packets and not child_packets:
            continue
        table_rows.extend(
            _build_table_rows(
                parent_packets=parent_packets,
                child_packets=child_packets,
                dt=dt,
                L=L,
                k_star=k_star,
                low_shell_k_offset=args.low_shell_k_offset,
                min_score=args.match_score_floor,
                secondary_ratio=args.match_secondary_ratio,
                match_tracking_threshold=args.match_tracking_threshold,
                weights={
                    "parent_overlap_weight": args.parent_overlap_weight,
                    "parent_mass_weight": args.parent_mass_weight,
                    "parent_direction_weight": args.parent_direction_weight,
                    "parent_shell_weight": args.parent_shell_weight,
                    "match_distance_scale": args.match_distance_scale,
                },
            )
        )
        if args.progress_every and ((t + 1) % args.progress_every == 0 or t + 1 == len(all_time_packets) - 1):
            elapsed = time.perf_counter() - t0
            print(
                "[ns_material_parent_summary] "
                f"transitions {t + 1}/{max(len(all_time_packets) - 1, 0)} table_rows={len(table_rows)} elapsed={elapsed:.1f}s",
                flush=True,
            )

    # If there are no transitions (single snapshot) still emit one-time summary rows.
    if not table_rows and all_time_packets:
        table_rows = _build_table_rows(
            parent_packets=[],
            child_packets=all_time_packets[0],
            dt=dt,
            L=L,
            k_star=k_star,
            low_shell_k_offset=args.low_shell_k_offset,
            min_score=args.match_score_floor,
            secondary_ratio=args.match_secondary_ratio,
            match_tracking_threshold=args.match_tracking_threshold,
            weights={
                "parent_overlap_weight": args.parent_overlap_weight,
                "parent_mass_weight": args.parent_mass_weight,
                "parent_direction_weight": args.parent_direction_weight,
                "parent_shell_weight": args.parent_shell_weight,
                "match_distance_scale": args.match_distance_scale,
            },
        )

    summary_rows = _summary_csv_rows(
        table_rows,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "ns_material_parent_table.csv"
    summary_path = out_dir / "ns_material_parent_summary.csv"
    manifest_path = out_dir / "ns_material_parent_summary.json"

    with open(table_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TableRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in _table_csv_rows(table_rows):
            writer.writerow(row)

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SummaryRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    route_counts: dict[str, int] = {}
    for row in table_rows:
        route_counts[row.parent_relation] = route_counts.get(row.parent_relation, 0) + 1

    # Global route status by time-K maxima follows harness replay semantics.
    per_k_time_weighted = {
        "weighted_true_new": float(sum(float(r["weighted_true_new"]) for r in summary_rows)),
        "weighted_tracking_uncertain": float(sum(float(r["weighted_tracking_uncertain"]) for r in summary_rows)),
        "weighted_cross_shell": float(sum(float(r["weighted_cross_shell"]) for r in summary_rows)),
        "weighted_low_shell": float(sum(float(r["weighted_low_shell_injection"]) for r in summary_rows)),
    }
    route_status = _route_status_from_summary(
        sigma_true_new=(
            sum(float(r["sigma_true_new_fit"]) for r in summary_rows) / max(len(summary_rows), 1)
        ),
        sigma_tracking_uncertain=(
            sum(float(r["sigma_tracking_uncertain_fit"]) for r in summary_rows) / max(len(summary_rows), 1)
        ),
        sigma_cross_shell=(
            sum(float(r["sigma_cross_shell_fit"]) for r in summary_rows) / max(len(summary_rows), 1)
        ),
        sigma_low_shell=(
            sum(float(r["sigma_low_shell_fit"]) for r in summary_rows) / max(len(summary_rows), 1)
        ),
        weighted_true_new=per_k_time_weighted["weighted_true_new"],
        weighted_tracking_uncertain=per_k_time_weighted["weighted_tracking_uncertain"],
        weighted_cross_shell=per_k_time_weighted["weighted_cross_shell"],
        weighted_low_shell=per_k_time_weighted["weighted_low_shell"],
    )

    prov = meta.get("host_provenance", {})
    gpu_path = os.environ.get("VK_ICD_FILENAMES")
    vulkan_info = prov.get("vulkan", {}) if isinstance(prov, dict) else {}
    manifest = {
        "contract": "ns_material_parent_artifact",
        "source_truth": str(args.truth),
        "requested_backend": args.backend,
        "actual_backend": actual_backend,
        "fft_backend": args.fft_backend if args.backend == "gpu" else "numpy",
        "diagnostic_precision": args.diagnostic_precision if args.backend == "gpu" else "float64_cpu",
        "icd_path": gpu_path,
        "device_name": str(
            gpu_device.get(
                "device_name",
                (vulkan_info.get("device") or {}).get("device_name", meta.get("device", {}).get("device_name", "not_probed")),
            )
        ),
        "gpu_runtime": gpu_runtime,
        "gpu_device": gpu_device,
        "shader_list": meta.get("gpu", {}).get("spv_shaders", []),
        "dtype": str(omega.dtype),
        "packet_grid": int(args.packet_grid),
        "packet_active_quantile": float(args.packet_active_quantile),
        "match_thresholds": {
            "match_score_floor": float(args.match_score_floor),
            "match_tracking_threshold": float(args.match_tracking_threshold),
            "match_secondary_ratio": float(args.match_secondary_ratio),
            "match_distance_scale": float(args.match_distance_scale),
        },
        "parent_score_weights": {
            "overlap": float(args.parent_overlap_weight),
            "mass": float(args.parent_mass_weight),
            "direction": float(args.parent_direction_weight),
            "shell": float(args.parent_shell_weight),
        },
        "ns_material_parent_table_path": str(table_path),
        "ns_material_parent_summary_path": str(summary_path),
        "table_row_count": len(table_rows),
        "summary_row_count": len(summary_rows),
        "manifest_time_range": {"first_time": float(steps[0] * dt) if len(steps) else 0.0, "last_time": float(steps[-1] * dt) if len(steps) else 0.0},
        "material_parent_route_status": route_status,
        "route_relation_counts": route_counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[ns_material_parent_summary] wrote {table_path}")
    print(f"[ns_material_parent_summary] wrote {summary_path}")
    print(f"[ns_material_parent_summary] wrote {manifest_path}")


if __name__ == "__main__":
    main()
