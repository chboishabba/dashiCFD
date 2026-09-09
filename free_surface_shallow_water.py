"""Deterministic 2D conservative shallow-water producer for bounded VFX shots.

This is a reduced free-surface hydrodynamics model, not a 3D VOF/RANS/LES solver.
It exists to produce physically dimensioned height/velocity/load fields with explicit
conservation diagnostics and a clear fidelity boundary for downstream DASHI/VFX use.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np

G = 9.80665
RHO_WATER = 1025.0
H_EPS = 1e-6


@dataclass(frozen=True)
class Grid:
    nx: int = 160
    ny: int = 80
    lx: float = 160.0
    ly: float = 80.0

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny


@dataclass(frozen=True)
class RunConfig:
    depth: float = 8.0
    inflow_u: float = 7.0
    cfl: float = 0.28
    seconds: float = 8.0
    disturbance_height: float = 1.0
    disturbance_radius: float = 7.5


def primitive(U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.maximum(U[0], H_EPS)
    return h, U[1] / h, U[2] / h


def flux_x(U: np.ndarray) -> np.ndarray:
    h, u, v = primitive(U)
    return np.stack((U[1], U[1] * u + 0.5 * G * h * h, U[1] * v))


def flux_y(U: np.ndarray) -> np.ndarray:
    h, u, v = primitive(U)
    return np.stack((U[2], U[2] * u, U[2] * v + 0.5 * G * h * h))


def rusanov_x(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    hl, ul, _ = primitive(left)
    hr, ur, _ = primitive(right)
    a = np.maximum(np.abs(ul) + np.sqrt(G * hl), np.abs(ur) + np.sqrt(G * hr))
    return 0.5 * (flux_x(left) + flux_x(right)) - 0.5 * a[None, ...] * (right - left)


def rusanov_y(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    hl, _, vl = primitive(lower)
    hr, _, vr = primitive(upper)
    a = np.maximum(np.abs(vl) + np.sqrt(G * hl), np.abs(vr) + np.sqrt(G * hr))
    return 0.5 * (flux_y(lower) + flux_y(upper)) - 0.5 * a[None, ...] * (upper - lower)


def make_hull_mask(grid: Grid) -> np.ndarray:
    x = (np.arange(grid.nx) + 0.5) * grid.dx - grid.lx / 2.0
    y = (np.arange(grid.ny) + 0.5) * grid.dy - grid.ly / 2.0
    X, Y = np.meshgrid(x, y)
    # Frigate-like slender ellipse: intentionally geometry-proxy, not a naval CAD hull.
    return (X / 28.0) ** 2 + (Y / 5.0) ** 2 <= 1.0


def initial_state(grid: Grid, cfg: RunConfig, solid: np.ndarray) -> np.ndarray:
    x = (np.arange(grid.nx) + 0.5) * grid.dx - grid.lx / 2.0
    y = (np.arange(grid.ny) + 0.5) * grid.dy - grid.ly / 2.0
    X, Y = np.meshgrid(x, y)
    r2 = (X + 18.0) ** 2 + (Y - 8.0) ** 2
    bump = cfg.disturbance_height * np.exp(-r2 / (2.0 * cfg.disturbance_radius**2))
    h = cfg.depth + bump
    hu = h * cfg.inflow_u
    hv = np.zeros_like(h)
    U = np.stack((h, hu, hv))
    U[:, solid] = 0.0
    return U


def _reflect_x(U: np.ndarray) -> np.ndarray:
    R = U.copy()
    R[1] *= -1.0
    return R


def _reflect_y(U: np.ndarray) -> np.ndarray:
    R = U.copy()
    R[2] *= -1.0
    return R


def step(U: np.ndarray, solid: np.ndarray, grid: Grid, dt: float) -> np.ndarray:
    # Transmissive outer boundary through edge replication.
    P = np.pad(U, ((0, 0), (1, 1), (1, 1)), mode="edge")
    S = np.pad(solid, ((1, 1), (1, 1)), mode="constant", constant_values=False)

    L = P[:, 1:-1, :-1]
    R = P[:, 1:-1, 1:]
    sl = S[1:-1, :-1]
    sr = S[1:-1, 1:]
    # Reflect fluid state at fluid/solid x faces; solid/solid flux is zero.
    Lx = np.where(sr[None, ...] & ~sl[None, ...], _reflect_x(L), L)
    Rx = np.where(sl[None, ...] & ~sr[None, ...], _reflect_x(R), R)
    Fx = rusanov_x(Lx, Rx)
    Fx[:, sl & sr] = 0.0

    B = P[:, :-1, 1:-1]
    T = P[:, 1:, 1:-1]
    sb = S[:-1, 1:-1]
    st = S[1:, 1:-1]
    By = np.where(st[None, ...] & ~sb[None, ...], _reflect_y(B), B)
    Ty = np.where(sb[None, ...] & ~st[None, ...], _reflect_y(T), T)
    Fy = rusanov_y(By, Ty)
    Fy[:, sb & st] = 0.0

    out = U - dt / grid.dx * (Fx[:, :, 1:] - Fx[:, :, :-1]) - dt / grid.dy * (Fy[:, 1:, :] - Fy[:, :-1, :])
    out[0] = np.maximum(out[0], H_EPS)
    out[:, solid] = 0.0
    return out


def stable_dt(U: np.ndarray, solid: np.ndarray, grid: Grid, cfl: float) -> float:
    h, u, v = primitive(U)
    wet = ~solid
    sx = np.max((np.abs(u) + np.sqrt(G * h))[wet])
    sy = np.max((np.abs(v) + np.sqrt(G * h))[wet])
    return cfl * min(grid.dx / max(sx, 1e-12), grid.dy / max(sy, 1e-12))


def diagnostics(U: np.ndarray, U0: np.ndarray, solid: np.ndarray, grid: Grid, cfg: RunConfig) -> dict[str, float | str | bool]:
    wet = ~solid
    cell_area = grid.dx * grid.dy
    h, u, v = primitive(U)
    h0, _, _ = primitive(U0)
    mass = float(RHO_WATER * np.sum(h[wet]) * cell_area)
    mass0 = float(RHO_WATER * np.sum(h0[wet]) * cell_area)
    kinetic = float(0.5 * RHO_WATER * np.sum(h[wet] * (u[wet] ** 2 + v[wet] ** 2)) * cell_area)
    potential = float(0.5 * RHO_WATER * G * np.sum(h[wet] ** 2) * cell_area)
    eta = np.where(wet, h - cfg.depth, 0.0)

    # Hydrostatic wall-load proxy from fluid cells adjacent to the hull mask.
    fluid = ~solid
    left_solid = np.roll(solid, 1, axis=1)
    right_solid = np.roll(solid, -1, axis=1)
    down_solid = np.roll(solid, 1, axis=0)
    up_solid = np.roll(solid, -1, axis=0)
    p = 0.5 * RHO_WATER * G * h * h
    fx = float(np.sum(p[fluid & left_solid]) * grid.dy - np.sum(p[fluid & right_solid]) * grid.dy)
    fy = float(np.sum(p[fluid & down_solid]) * grid.dx - np.sum(p[fluid & up_solid]) * grid.dx)

    receipt_basis = {
        "model": "2d_saint_venant_rusanov_v1",
        "nx": grid.nx,
        "ny": grid.ny,
        "lx_m": grid.lx,
        "ly_m": grid.ly,
        "depth_m": cfg.depth,
        "inflow_u_m_s": cfg.inflow_u,
        "seconds": cfg.seconds,
        "mass_relative_error": (mass - mass0) / max(abs(mass0), 1.0),
        "max_abs_eta_m": float(np.max(np.abs(eta))),
        "hull_force_x_N": fx,
        "hull_force_y_N": fy,
    }
    digest = hashlib.sha256(json.dumps(receipt_basis, sort_keys=True).encode()).hexdigest()
    return {
        **receipt_basis,
        "mass_kg": mass,
        "mass_initial_kg": mass0,
        "kinetic_energy_J": kinetic,
        "hydrostatic_potential_J": potential,
        "artifact_sha256": digest,
        "fidelity": "reduced_free_surface_shallow_water_not_3d_vof_rans_les",
        "physically_calibrated_hull": False,
    }


def run(grid: Grid = Grid(), cfg: RunConfig = RunConfig()) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | bool]]:
    solid = make_hull_mask(grid)
    U = initial_state(grid, cfg, solid)
    U0 = U.copy()
    t = 0.0
    while t < cfg.seconds:
        dt = min(stable_dt(U, solid, grid, cfg.cfl), cfg.seconds - t)
        U = step(U, solid, grid, dt)
        t += dt
    return U, solid, diagnostics(U, U0, solid, grid, cfg)
