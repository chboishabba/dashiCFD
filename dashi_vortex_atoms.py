"""Deterministic sparse-vorticity atom extraction, transport, decode, and ledgers.

This module is the numerical vertical slice corresponding to the Agda
``SparseTwistLES`` contracts.  It is deliberately CPU/NumPy and deterministic:
CPU float64 remains the receipt authority, while GPU implementations may later
be checked against this surface.

The model is a bounded-fidelity proxy, not a Navier--Stokes theorem.  A proxy
state contains a transported smooth background plus sparse signed anisotropic
vortex atoms.  Aggregate/random-phase reconstruction is intentionally absent
from the faithful path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class VortexAtom:
    atom_id: int
    parent_id: int | None
    y: float
    x: float
    sign: int
    amplitude: float
    circulation: float
    core_scale: float
    orientation: float
    anisotropy: float
    lifetime: int = 0

    def to_json(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class AtomCodecConfig:
    smooth_k: int = 11
    threshold_sigma: float = 0.75
    peak_candidates: int = 48
    max_atoms: int = 32
    min_separation: float = 2.0
    scales: tuple[float, ...] = (2.0, 4.0)
    anisotropies: tuple[float, ...] = (1.0, 3.0)
    orientations: int = 8
    min_mdl_gain: float = 0.0
    bits_per_atom: float = 160.0
    q: float = 0.05
    prune_amplitude: float = 1e-5
    merge_distance: float = 1.5
    merge_orientation_tolerance: float = math.pi / 6.0


@dataclass(frozen=True)
class ProxyState:
    background: Array
    atoms: tuple[VortexAtom, ...]
    next_atom_id: int
    step: int = 0


@dataclass(frozen=True)
class EventLedger:
    step: int
    transported: int
    pruned: int
    merged: int
    split: int
    born: int
    atom_count_before: int
    atom_count_after: int
    carrier_bits: float
    operator_bits: float
    residual_bits: float
    defect_bits: float
    total_mdl_bits: float

    def to_json(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class DefectMetrics:
    rel_l2: float
    correlation: float
    max_abs: float
    circulation_error: float
    enstrophy_error: float
    signed_support_iou: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


def _odd_width(k: int) -> int:
    k = max(int(k), 1)
    return k if k % 2 else k + 1


def periodic_box_smooth(field: Array, k: int) -> Array:
    """Periodic separable box smoother; unlike ``mode='same'`` it respects the torus."""
    k = _odd_width(k)
    radius = k // 2
    out = np.zeros_like(field, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        out += np.roll(field, dy, axis=0)
    tmp = out / float(k)
    out = np.zeros_like(field, dtype=np.float64)
    for dx in range(-radius, radius + 1):
        out += np.roll(tmp, dx, axis=1)
    return out / float(k)


def spectral_velocity(omega: Array, length: float = 2.0 * math.pi) -> tuple[Array, Array]:
    """Recover periodic incompressible velocity from scalar vorticity."""
    h, w = omega.shape
    if h != w:
        raise ValueError("spectral_velocity currently requires a square periodic grid")
    n = h
    dx = length / n
    k = np.fft.fftfreq(n, d=dx) * 2.0 * math.pi
    ky, kx = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0
    oh = np.fft.fft2(omega)
    psi_hat = oh / k2
    psi_hat[0, 0] = 0.0
    u = np.fft.ifft2(1j * ky * psi_hat).real
    v = np.fft.ifft2(-1j * kx * psi_hat).real
    return u, v


def periodic_bilinear(field: Array, y: float, x: float) -> float:
    h, w = field.shape
    y %= h
    x %= w
    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    y1 = (y0 + 1) % h
    x1 = (x0 + 1) % w
    fy = y - y0
    fx = x - x0
    return float(
        (1.0 - fy) * (1.0 - fx) * field[y0, x0]
        + fy * (1.0 - fx) * field[y1, x0]
        + (1.0 - fy) * fx * field[y0, x1]
        + fy * fx * field[y1, x1]
    )


def periodic_gradient(field: Array) -> tuple[Array, Array]:
    dy = 0.5 * (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0))
    dx = 0.5 * (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1))
    return dy, dx


def _periodic_delta(a: float, b: float, extent: int) -> float:
    d = a - b
    return (d + extent / 2.0) % extent - extent / 2.0


def _orientation_distance(a: float, b: float) -> float:
    d = abs((a - b) % math.pi)
    return min(d, math.pi - d)


def atom_kernel(shape: tuple[int, int], atom: VortexAtom, *, unit_l2: bool = True) -> Array:
    """Periodic signed anisotropic Gaussian atom."""
    h, w = shape
    yy, xx = np.indices(shape, dtype=np.float64)
    dy = ((yy - atom.y + h / 2.0) % h) - h / 2.0
    dx = ((xx - atom.x + w / 2.0) % w) - w / 2.0
    c = math.cos(atom.orientation)
    s = math.sin(atom.orientation)
    major = c * dx + s * dy
    minor = -s * dx + c * dy
    sigma_major = max(float(atom.core_scale), 1e-6)
    sigma_minor = max(sigma_major / max(float(atom.anisotropy), 1.0), 1e-6)
    kernel = np.exp(-0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2))
    kernel -= float(kernel.mean())
    if unit_l2:
        norm = float(np.linalg.norm(kernel))
        if norm > 1e-12:
            kernel /= norm
    return atom.amplitude * kernel


def decode_atoms(shape: tuple[int, int], atoms: Sequence[VortexAtom]) -> Array:
    field = np.zeros(shape, dtype=np.float64)
    for atom in atoms:
        field += atom_kernel(shape, atom)
    return field


def decode_proxy(state: ProxyState) -> Array:
    return np.asarray(state.background, dtype=np.float64) + decode_atoms(state.background.shape, state.atoms)


def _candidate_peaks(residual: Array, config: AtomCodecConfig) -> list[tuple[int, int]]:
    threshold = config.threshold_sigma * float(np.std(residual))
    order = np.argsort(np.abs(residual).ravel())[::-1]
    h, w = residual.shape
    selected: list[tuple[int, int]] = []
    for flat in order:
        y, x = divmod(int(flat), w)
        if abs(float(residual[y, x])) < threshold:
            break
        if all(
            _periodic_delta(y, py, h) ** 2 + _periodic_delta(x, px, w) ** 2
            >= config.min_separation**2
            for py, px in selected
        ):
            selected.append((y, x))
            if len(selected) >= config.peak_candidates:
                break
    return selected


def _orientation_grid(count: int) -> tuple[float, ...]:
    count = max(int(count), 1)
    return tuple(math.pi * i / count for i in range(count))


def extract_vortex_atoms(
    omega: Array,
    config: AtomCodecConfig = AtomCodecConfig(),
    *,
    start_atom_id: int = 0,
) -> ProxyState:
    """Greedy deterministic MDL extraction of signed, placed, oriented atoms."""
    omega = np.asarray(omega, dtype=np.float64)
    background = periodic_box_smooth(omega, config.smooth_k)
    residual = omega - background
    atoms: list[VortexAtom] = []
    next_id = int(start_atom_id)
    q2 = max(config.q * config.q, 1e-18)
    orientations = _orientation_grid(config.orientations)

    for _ in range(max(config.max_atoms, 0)):
        peaks = _candidate_peaks(residual, config)
        best: VortexAtom | None = None
        best_kernel: Array | None = None
        best_gain = -math.inf
        for y, x in peaks:
            for scale in config.scales:
                for anisotropy in config.anisotropies:
                    for orientation in orientations:
                        template_atom = VortexAtom(
                            atom_id=next_id,
                            parent_id=None,
                            y=float(y),
                            x=float(x),
                            sign=1,
                            amplitude=1.0,
                            circulation=0.0,
                            core_scale=float(scale),
                            orientation=float(orientation),
                            anisotropy=float(anisotropy),
                        )
                        template = atom_kernel(omega.shape, template_atom)
                        denom = float(np.vdot(template, template).real)
                        if denom <= 1e-12:
                            continue
                        amplitude = float(np.vdot(residual, template).real / denom)
                        kernel = amplitude * template
                        reduction = float(
                            2.0 * amplitude * np.vdot(residual, template).real
                            - amplitude * amplitude * denom
                        )
                        gain = reduction / q2 - config.bits_per_atom
                        if gain > best_gain:
                            circulation = float(np.sum(kernel))
                            best_gain = gain
                            best_kernel = kernel
                            best = replace(
                                template_atom,
                                sign=1 if amplitude >= 0.0 else -1,
                                amplitude=amplitude,
                                circulation=circulation,
                            )
        if best is None or best_kernel is None or best_gain <= config.min_mdl_gain:
            break
        atoms.append(best)
        residual -= best_kernel
        next_id += 1

    return ProxyState(background=background, atoms=tuple(atoms), next_atom_id=next_id, step=0)


def advect_background(background: Array, u: Array, v: Array, dt: float, viscosity: float = 0.0) -> Array:
    """Periodic semi-Lagrangian background transport plus spectral diffusion."""
    h, w = background.shape
    yy, xx = np.indices(background.shape, dtype=np.float64)
    departure_y = (yy - dt * v) % h
    departure_x = (xx - dt * u) % w
    y0 = np.floor(departure_y).astype(np.int64)
    x0 = np.floor(departure_x).astype(np.int64)
    y1 = (y0 + 1) % h
    x1 = (x0 + 1) % w
    fy = departure_y - y0
    fx = departure_x - x0
    advected = (
        (1.0 - fy) * (1.0 - fx) * background[y0, x0]
        + fy * (1.0 - fx) * background[y1, x0]
        + (1.0 - fy) * fx * background[y0, x1]
        + fy * fx * background[y1, x1]
    )
    if viscosity <= 0.0:
        return advected
    ky = np.fft.fftfreq(h) * 2.0 * math.pi
    kx = np.fft.fftfreq(w) * 2.0 * math.pi
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    damping = np.exp(-viscosity * dt * (KX * KX + KY * KY))
    return np.fft.ifft2(np.fft.fft2(advected) * damping).real


def transport_atoms(
    atoms: Sequence[VortexAtom],
    u: Array,
    v: Array,
    dt: float,
    *,
    viscosity: float = 0.0,
) -> tuple[VortexAtom, ...]:
    """Advect positions and twist orientations by the sampled velocity gradient."""
    h, w = u.shape
    du_dy, du_dx = periodic_gradient(u)
    dv_dy, dv_dx = periodic_gradient(v)
    transported: list[VortexAtom] = []
    for atom in atoms:
        ua = periodic_bilinear(u, atom.y, atom.x)
        va = periodic_bilinear(v, atom.y, atom.x)
        y_new = (atom.y + dt * va) % h
        x_new = (atom.x + dt * ua) % w

        tx = math.cos(atom.orientation)
        ty = math.sin(atom.orientation)
        gx = periodic_bilinear(du_dx, atom.y, atom.x) * tx + periodic_bilinear(du_dy, atom.y, atom.x) * ty
        gy = periodic_bilinear(dv_dx, atom.y, atom.x) * tx + periodic_bilinear(dv_dy, atom.y, atom.x) * ty
        tx_new = tx + dt * gx
        ty_new = ty + dt * gy
        orientation = math.atan2(ty_new, tx_new) % math.pi

        decay = math.exp(-max(viscosity, 0.0) * dt / max(atom.core_scale * atom.core_scale, 1e-12))
        amplitude = atom.amplitude * decay
        transported.append(
            replace(
                atom,
                y=y_new,
                x=x_new,
                orientation=orientation,
                amplitude=amplitude,
                circulation=atom.circulation * decay,
                lifetime=atom.lifetime + 1,
            )
        )
    return tuple(transported)


def apply_mdl_events(
    atoms: Sequence[VortexAtom],
    config: AtomCodecConfig,
) -> tuple[tuple[VortexAtom, ...], dict[str, int]]:
    """Deterministic prune/merge policy; split and birth remain explicit zeros."""
    pruned_atoms = [a for a in atoms if abs(a.amplitude) >= config.prune_amplitude]
    pruned = len(atoms) - len(pruned_atoms)
    used = [False] * len(pruned_atoms)
    merged_atoms: list[VortexAtom] = []
    merged_count = 0
    for i, atom in enumerate(pruned_atoms):
        if used[i]:
            continue
        group = [atom]
        used[i] = True
        for j in range(i + 1, len(pruned_atoms)):
            other = pruned_atoms[j]
            if used[j] or other.sign != atom.sign:
                continue
            dy = _periodic_delta(other.y, atom.y, 10**9)  # overwritten below for ordinary coordinates
            dx = _periodic_delta(other.x, atom.x, 10**9)
            distance = math.hypot(dy, dx)
            if distance <= config.merge_distance and _orientation_distance(atom.orientation, other.orientation) <= config.merge_orientation_tolerance:
                group.append(other)
                used[j] = True
        if len(group) == 1:
            merged_atoms.append(atom)
            continue
        weights = np.array([abs(a.amplitude) for a in group], dtype=np.float64)
        weights /= float(weights.sum()) + 1e-12
        base = min(group, key=lambda a: a.atom_id)
        merged_atoms.append(
            replace(
                base,
                parent_id=base.atom_id,
                y=float(sum(w * a.y for w, a in zip(weights, group))),
                x=float(sum(w * a.x for w, a in zip(weights, group))),
                amplitude=float(sum(a.amplitude for a in group)),
                circulation=float(sum(a.circulation for a in group)),
                core_scale=float(max(a.core_scale for a in group)),
                lifetime=max(a.lifetime for a in group),
            )
        )
        merged_count += len(group) - 1
    return tuple(merged_atoms), {
        "pruned": pruned,
        "merged": merged_count,
        "split": 0,
        "born": 0,
    }


def proxy_step(
    state: ProxyState,
    dt: float,
    config: AtomCodecConfig = AtomCodecConfig(),
    *,
    viscosity: float = 0.0,
    defect_rel_l2: float = 0.0,
) -> tuple[ProxyState, EventLedger]:
    field = decode_proxy(state)
    u, v = spectral_velocity(field)
    background = advect_background(state.background, u, v, dt, viscosity)
    transported = transport_atoms(state.atoms, u, v, dt, viscosity=viscosity)
    atoms, events = apply_mdl_events(transported, config)

    # Explicit, simple two-part MDL ledger.  Constants are a runtime policy,
    # not a coding theorem; the ledger makes that approximation visible.
    carrier_bits = config.bits_per_atom * len(atoms)
    operator_bits = 4.0 * 64.0 * len(atoms)  # position delta + twist/orientation update
    residual_bits = 64.0 * float(state.background.size)
    q = max(config.q, 1e-12)
    defect_bits = state.background.size * math.log2(1.0 + max(defect_rel_l2, 0.0) / q)
    total = carrier_bits + operator_bits + residual_bits + defect_bits
    ledger = EventLedger(
        step=state.step + 1,
        transported=len(state.atoms),
        pruned=events["pruned"],
        merged=events["merged"],
        split=events["split"],
        born=events["born"],
        atom_count_before=len(state.atoms),
        atom_count_after=len(atoms),
        carrier_bits=carrier_bits,
        operator_bits=operator_bits,
        residual_bits=residual_bits,
        defect_bits=defect_bits,
        total_mdl_bits=total,
    )
    return ProxyState(background, atoms, state.next_atom_id, state.step + 1), ledger


def signed_support_iou(a: Array, b: Array, threshold: float | None = None) -> float:
    if threshold is None:
        threshold = 0.5 * min(float(np.std(a)), float(np.std(b)))
    sa = np.where(a > threshold, 1, np.where(a < -threshold, -1, 0))
    sb = np.where(b > threshold, 1, np.where(b < -threshold, -1, 0))
    union = (sa != 0) | (sb != 0)
    if not np.any(union):
        return 1.0
    intersection = (sa == sb) & union
    return float(np.count_nonzero(intersection) / np.count_nonzero(union))


def defect_metrics(reference: Array, candidate: Array) -> DefectMetrics:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    error = reference - candidate
    ref_norm = float(np.linalg.norm(reference))
    rel_l2 = float(np.linalg.norm(error) / (ref_norm + 1e-12))
    if float(np.std(reference)) <= 1e-14 or float(np.std(candidate)) <= 1e-14:
        correlation = 1.0 if rel_l2 <= 1e-14 else 0.0
    else:
        correlation = float(np.corrcoef(reference.ravel(), candidate.ravel())[0, 1])
    return DefectMetrics(
        rel_l2=rel_l2,
        correlation=correlation,
        max_abs=float(np.max(np.abs(error))),
        circulation_error=float(abs(np.sum(reference) - np.sum(candidate))),
        enstrophy_error=float(abs(0.5 * np.mean(reference * reference) - 0.5 * np.mean(candidate * candidate))),
        signed_support_iou=signed_support_iou(reference, candidate),
    )


def genealogy_rows(atoms: Iterable[VortexAtom], step: int) -> list[dict[str, int | float | None]]:
    rows: list[dict[str, int | float | None]] = []
    for atom in atoms:
        row = atom.to_json()
        row["step"] = int(step)
        rows.append(row)
    return rows
