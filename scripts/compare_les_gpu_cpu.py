#!/usr/bin/env python3
"""Compare GPU LES vs CPU baseline on a small run (sanity check)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from dashi_cfd_operator_v4 import (
    make_grid,
    poisson_solve_minus_lap,
    velocity_from_psi,
    smagorinsky_nu,
    step_rk2,
    enstrophy as enstrophy_cpu,
)
from vulkan_les_backend import VulkanLESBackend, init_random_omega


@dataclass
class CompareResult:
    rel_l2: float
    enstrophy_cpu: float
    enstrophy_gpu: float
    mean_abs_delta: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=64, help="grid size (default: 64)")
    p.add_argument("--steps", type=int, default=50, help="number of steps (default: 50)")
    p.add_argument("--dt", type=float, default=0.01, help="time step")
    p.add_argument("--nu0", type=float, default=1e-4, help="base viscosity")
    p.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant")
    p.add_argument("--seed", type=int, default=0, help="initial condition seed")
    p.add_argument("--stats-every", type=int, default=10, help="print enstrophy every K steps")
    p.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="FFT backend for GPU")
    return p.parse_args()


def cpu_step(omega: np.ndarray, *, nu0: float, Cs: float, dt: float, KX, KY, K2, dx: float) -> np.ndarray:
    psi = poisson_solve_minus_lap(omega, K2)
    u, v = velocity_from_psi(psi, KX, KY)
    nu_t = np.maximum(0.0, smagorinsky_nu(u, v, KX, KY, Cs, dx))
    return step_rk2(omega, nu0 + nu_t, dt, KX, KY, K2)


def compare_run(
    N: int,
    steps: int,
    dt: float,
    nu0: float,
    Cs: float,
    seed: int,
    stats_every: int,
    fft_backend: str,
) -> CompareResult:
    omega0 = init_random_omega(N, seed=seed)

    dx, KX, KY, K2 = make_grid(N)
    omega_cpu = omega0.astype(np.float64, copy=True)

    gpu = VulkanLESBackend(N, dt=dt, nu0=nu0, Cs=Cs, fft_backend=fft_backend)
    gpu.set_initial_omega(omega0)

    t0 = perf_counter()
    for step in range(1, steps + 1):
        omega_cpu = cpu_step(omega_cpu, nu0=nu0, Cs=Cs, dt=dt, KX=KX, KY=KY, K2=K2, dx=dx)
        gpu.step()
        if stats_every and (step % stats_every == 0):
            Zc = enstrophy_cpu(omega_cpu)
            Zg = gpu.enstrophy()
            print(f"[stats] t={step:05d} cpu={Zc:.6e} gpu={Zg:.6e}")
    dt_total = perf_counter() - t0

    omega_gpu = gpu.read_omega().astype(np.float64, copy=False)
    diff = omega_gpu - omega_cpu
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(omega_cpu) + 1e-12))
    mean_abs_delta = float(np.mean(np.abs(diff)))
    return CompareResult(
        rel_l2=rel_l2,
        enstrophy_cpu=float(enstrophy_cpu(omega_cpu)),
        enstrophy_gpu=float(enstrophy_cpu(omega_gpu)),
        mean_abs_delta=mean_abs_delta,
    )


def main() -> None:
    args = parse_args()
    result = compare_run(
        N=args.N,
        steps=args.steps,
        dt=args.dt,
        nu0=args.nu0,
        Cs=args.Cs,
        seed=args.seed,
        stats_every=args.stats_every,
        fft_backend=args.fft_backend,
    )
    print(
        "[compare] rel_l2={:.3e}  mean_abs_delta={:.3e}  enstrophy_cpu={:.6e}  enstrophy_gpu={:.6e}".format(
            result.rel_l2,
            result.mean_abs_delta,
            result.enstrophy_cpu,
            result.enstrophy_gpu,
        )
    )


if __name__ == "__main__":
    main()
