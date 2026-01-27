#!/usr/bin/env python3
"""GPU-only LES run with vkFFT + Vulkan kernels; emits enstrophy and optional visuals."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt

from vulkan_les_backend import VulkanLESBackend, init_random_omega


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=256, help="grid size (default: 256)")
    p.add_argument("--steps", type=int, default=5000, help="number of steps")
    p.add_argument("--dt", type=float, default=0.01, help="time step")
    p.add_argument("--nu0", type=float, default=1e-4, help="base viscosity")
    p.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant")
    p.add_argument("--seed", type=int, default=0, help="initial condition seed")
    p.add_argument("--stats-every", type=int, default=200, help="emit enstrophy every K steps")
    p.add_argument("--viz-every", type=int, default=0, help="save omega PNG every K steps (0 disables)")
    p.add_argument("--progress-every", type=int, default=0, help="print progress every K steps (0 disables)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"), help="output directory")
    p.add_argument("--prefix", type=str, default="les_gpu", help="output filename prefix")
    p.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="FFT backend")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    backend = VulkanLESBackend(args.N, dt=args.dt, nu0=args.nu0, Cs=args.Cs, fft_backend=args.fft_backend)
    omega0 = init_random_omega(args.N, seed=args.seed)
    backend.set_initial_omega(omega0)

    enstrophy_log = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        backend.step()
        if args.stats_every and (step % args.stats_every == 0):
            Z = backend.enstrophy()
            enstrophy_log.append((step, Z))
            print(f"[stats] t={step} enstrophy={Z:.6e}")
        if args.progress_every and (step % args.progress_every == 0):
            print(f"[progress] t={step}/{args.steps}")
        if args.viz_every and (step % args.viz_every == 0):
            omega = backend.read_omega()
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
            im = ax.imshow(omega, origin="lower", cmap="viridis")
            ax.set_title(f"omega t={step}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            out_path = args.out_dir / f"{args.prefix}_t{step:06d}.png"
            fig.savefig(out_path, dpi=140)
            plt.close(fig)
            print(f"[viz] wrote {out_path}")

    dt_total = time.perf_counter() - t0
    if enstrophy_log:
        out_csv = args.out_dir / f"{args.prefix}_enstrophy.csv"
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("step,enstrophy\n")
            for step, val in enstrophy_log:
                f.write(f"{step},{val}\n")
        print(f"[stats] wrote {out_csv}")
    print(f"[done] steps={args.steps} total_s={dt_total:.3f}")


if __name__ == "__main__":
    main()
