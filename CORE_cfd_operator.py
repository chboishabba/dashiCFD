#!/usr/bin/env python3
"""
CORE_cfd_operator.py

Purpose: Run a lightweight 2D vorticity rollout using the dashiCORE backend
abstractions (Carrier + backend registry) and compare performance against the
legacy numpy-only gating path. Designed for quick, headless benchmarking rather
than plotting.

Usage (CPU backend, headless):
    MPLBACKEND=Agg python CORE_cfd_operator.py

Optional env vars:
    CORE_BACKEND=cpu|accelerated   # selects dashi_core backend
    FUSED_MASK=1                   # bypass Carrier/Backend and run fused numpy mask
    STEPS=200                      # timestep count (default 120)
    N=64                           # grid size (default 64)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

# Ensure dashiCORE package is importable
ROOT = Path(__file__).resolve().parent
CORE_ROOT = ROOT / "dashiCORE"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from dashi_core.carrier import Carrier
from dashi_core.backend import use_backend, get_backend


# -----------------------------
# Spectral utilities (periodic box)
# -----------------------------

def fft2(a): return np.fft.fft2(a)
def ifft2(a): return np.fft.ifft2(a).real


def make_grid(N: int, L: float = 2 * np.pi):
    dx = L / N
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    return dx, KX, KY, K2


def deriv_x(a, KX): return ifft2(1j * KX * fft2(a))
def deriv_y(a, KY): return ifft2(1j * KY * fft2(a))
def laplacian(a, K2): return ifft2(-K2 * fft2(a))


def poisson_solve_minus_lap(omega, K2):
    oh = fft2(omega)
    psih = oh / K2
    psih[0, 0] = 0.0
    return ifft2(psih)


def velocity_from_psi(psi, KX, KY):
    u = deriv_y(psi, KY)
    v = -deriv_x(psi, KX)
    return u, v


# -----------------------------
# Legacy ternary ops (baseline)
# -----------------------------

def ternary_sym(X, tau):
    s = np.zeros_like(X, dtype=np.int8)
    s[X >= tau] = +1
    s[X <= -tau] = -1
    return s


def majority_kernel_3x3(s: np.ndarray) -> np.ndarray:
    acc = (
        s +
        np.roll(s, 1, axis=0) + np.roll(s, -1, axis=0) +
        np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1) +
        np.roll(np.roll(s, 1, axis=0), 1, axis=1) +
        np.roll(np.roll(s, 1, axis=0), -1, axis=1) +
        np.roll(np.roll(s, -1, axis=0), 1, axis=1) +
        np.roll(np.roll(s, -1, axis=0), -1, axis=1)
    ).astype(np.int16)
    out = np.zeros_like(s, dtype=np.int8)
    out[acc > 0] = 1
    out[acc < 0] = -1
    return out


def saturate_ternary(s0: np.ndarray, iters: int = 6) -> np.ndarray:
    s = s0
    for _ in range(iters):
        sn = majority_kernel_3x3(s)
        if np.array_equal(sn, s):
            break
        s = sn
    return s


def legacy_mask(X: np.ndarray, tau: float) -> np.ndarray:
    """Baseline ternary gating used in earlier scripts."""
    s0 = ternary_sym(X, tau)
    s_star = saturate_ternary(s0)
    return (s_star != 0).astype(np.float64)


# -----------------------------
# CORE-backed ternary ops
# -----------------------------

def core_mask(X: np.ndarray, tau: float, backend_name: str, fused: bool) -> Tuple[np.ndarray, dict]:
    """
    Convert ternary field into a Carrier using dashiCORE backends.
    Returns support mask and backend metrics.
    """
    if fused:
        signed = ternary_sym(X, tau)
        s_arr = saturate_ternary(signed)
        mask = (s_arr != 0).astype(np.float64)
        return mask, {"ops": 0}

    with use_backend(backend_name):
        backend = get_backend()
        backend.reset_metrics()

        signed = ternary_sym(X, tau)
        carrier = Carrier.from_signed(signed)

        # Majority using signed view; could be swapped with accelerated kernels later.
        s_arr = carrier.to_signed()
        s_arr = saturate_ternary(s_arr)
        carrier_sat = Carrier.from_signed(s_arr)

        mask = carrier_sat.support.astype(np.float64)
        metrics = dict(backend.metrics)
    return mask, metrics


# -----------------------------
# LES utilities
# -----------------------------

def strain_mag(u, v, KX, KY):
    du_dx = deriv_x(u, KX)
    du_dy = deriv_y(u, KY)
    dv_dx = deriv_x(v, KX)
    dv_dy = deriv_y(v, KY)
    Sxy = 0.5 * (du_dy + dv_dx)
    return np.sqrt(2 * (du_dx**2 + dv_dy**2 + 2 * Sxy**2) + 1e-30)


def smagorinsky_nu(u, v, KX, KY, Cs, Delta):
    return (Cs * Delta) ** 2 * strain_mag(u, v, KX, KY)


def rhs_vorticity(omega, nu, KX, KY, K2):
    psi = poisson_solve_minus_lap(omega, K2)
    u, v = velocity_from_psi(psi, KX, KY)
    adv = u * deriv_x(omega, KX) + v * deriv_y(omega, KY)
    diff = nu * laplacian(omega, K2)
    return -adv + diff


def step_rk2(omega, nu, dt, KX, KY, K2):
    k1 = rhs_vorticity(omega, nu, KX, KY, K2)
    k2 = rhs_vorticity(omega + dt * k1, nu, KX, KY, K2)
    return omega + 0.5 * dt * (k1 + k2)


# -----------------------------
# Rollout driver
# -----------------------------

@dataclass
class RunResult:
    label: str
    backend: str
    steps: int
    ms_per_step: float
    final_enstrophy: float
    mask_mean: float
    mask_cost_ops: int


def run_sim(mask_fn: Callable[[np.ndarray], Tuple[np.ndarray, dict]], label: str, backend: str, steps: int, N: int) -> RunResult:
    dx, KX, KY, K2 = make_grid(N)
    np.random.seed(0)
    omega = np.random.randn(N, N)
    omega = (omega - omega.mean()) / (omega.std() + 1e-12)

    nu0 = 1e-4
    dt = 0.01
    Cs = 0.17

    t0 = time.perf_counter()
    mask_mean_accum = 0.0
    mask_ops_accum = 0

    for _ in range(steps):
        psi = poisson_solve_minus_lap(omega, K2)
        u, v = velocity_from_psi(psi, KX, KY)
        nu_t_base = smagorinsky_nu(u, v, KX, KY, Cs=Cs, Delta=dx)

        X = omega - (omega.mean())
        X = X / (np.max(np.abs(X)) + 1e-12)

        mask, metrics = mask_fn(X)
        g = (1.0 - 0.7 * mask)  # suppress SGS where structure is present
        nu_eff = nu0 + np.maximum(0.0, nu_t_base * g)

        omega = step_rk2(omega, nu_eff, dt, KX, KY, K2)
        mask_mean_accum += float(mask.mean())
        mask_ops_accum += int(metrics.get("ops", 0))

    total_s = time.perf_counter() - t0
    enstrophy = 0.5 * float(np.mean(omega * omega))
    return RunResult(
        label=label,
        backend=backend,
        steps=steps,
        ms_per_step=1000 * total_s / steps,
        final_enstrophy=enstrophy,
        mask_mean=mask_mean_accum / steps,
        mask_cost_ops=mask_ops_accum // max(steps, 1),
    )


# -----------------------------
# Perf harness
# -----------------------------

def main():
    backend_env = os.environ.get("CORE_BACKEND", "cpu")
    fused = os.environ.get("FUSED_MASK", "0") == "1"
    steps = int(os.environ.get("STEPS", "120"))
    N = int(os.environ.get("N", "64"))

    # Legacy path: pure NumPy mask
    def mask_legacy(X):
        return legacy_mask(X, tau=0.35), {"ops": 0}

    res_legacy = run_sim(mask_legacy, label="legacy", backend="numpy", steps=steps, N=N)

    def mask_core(X):
        return core_mask(X, tau=0.35, backend_name=backend_env, fused=fused)

    res_core = run_sim(mask_core, label="core", backend=backend_env, steps=steps, N=N)

    speedup = res_legacy.ms_per_step / res_core.ms_per_step if res_core.ms_per_step else float("inf")

    print(f"\n=== CORE vs Legacy (N={N}, steps={steps}, backend={backend_env}, fused={fused}) ===")
    print(f"{res_legacy.label:8s}  ms/step={res_legacy.ms_per_step:7.3f}  enstrophy={res_legacy.final_enstrophy:8.4f}  mask_mean={res_legacy.mask_mean:6.3f}")
    print(f"{res_core.label:8s}  ms/step={res_core.ms_per_step:7.3f}  enstrophy={res_core.final_enstrophy:8.4f}  mask_mean={res_core.mask_mean:6.3f}  mask_ops≈{res_core.mask_cost_ops}")
    print(f"speedup (legacy/core): {speedup:0.3f}x")


if __name__ == "__main__":
    main()
