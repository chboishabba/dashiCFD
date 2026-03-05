#!/usr/bin/env python3
"""
Interactive CLI wrapper for common GPU/CPU workflows.

Examples:
  python dashi_cli.py les --interactive
  python dashi_cli.py kernel --interactive
  python dashi_cli.py plot --interactive
  python dashi_cli.py compare --interactive
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    resp = input(f"{text}{suffix}: ").strip()
    return resp if resp else (default or "")


def _prompt_int(text: str, default: int) -> int:
    while True:
        resp = _prompt(text, str(default))
        try:
            return int(resp)
        except ValueError:
            print("Enter an integer.")


def _prompt_float(text: str, default: float) -> float:
    while True:
        resp = _prompt(text, str(default))
        try:
            return float(resp)
        except ValueError:
            print("Enter a number.")


def _prompt_choice(text: str, choices: list[str], default: str) -> str:
    while True:
        resp = _prompt(f"{text} ({', '.join(choices)})", default)
        if resp in choices:
            return resp
        print(f"Choose one of: {', '.join(choices)}")


def _prompt_bool(text: str, default: bool) -> bool:
    default_str = "y" if default else "n"
    while True:
        resp = _prompt(f"{text} (y/n)", default_str).lower()
        if resp in {"y", "yes"}:
            return True
        if resp in {"n", "no"}:
            return False
        print("Enter y or n.")


def _run(cmd: list[str], *, headless: bool) -> int:
    env = os.environ.copy()
    if headless:
        env.setdefault("MPLBACKEND", "Agg")
    print("[run] " + " ".join(cmd))
    return subprocess.call(cmd, env=env)


def _script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def cmd_les(args: argparse.Namespace) -> int:
    if args.interactive:
        args.N = _prompt_int("Grid size N", args.N)
        args.steps = _prompt_int("Steps", args.steps)
        args.dt = _prompt_float("dt", args.dt)
        args.nu0 = _prompt_float("nu0", args.nu0)
        args.Cs = _prompt_float("Cs", args.Cs)
        args.stats_every = _prompt_int("stats-every", args.stats_every)
        args.viz_every = _prompt_int("viz-every (0 disables)", args.viz_every)
        args.progress_every = _prompt_int("progress-every (0 disables)", args.progress_every)
        args.out_dir = Path(_prompt("out-dir", str(args.out_dir)))
        args.prefix = _prompt("prefix", args.prefix)
        args.fft_backend = _prompt_choice("fft-backend", ["vkfft-vulkan", "vkfft-opencl", "vkfft", "numpy"], args.fft_backend)
        args.spectral_truncation = _prompt_choice("spectral-truncation", ["none", "exp"], args.spectral_truncation)
        args.trunc_alpha = _prompt_float("trunc-alpha", args.trunc_alpha)
        args.trunc_power = _prompt_float("trunc-power", args.trunc_power)
        args.headless = _prompt_bool("Headless (MPLBACKEND=Agg)", args.headless)

    cmd = [
        sys.executable,
        _script_path("run_les_gpu.py"),
        "--N",
        str(args.N),
        "--steps",
        str(args.steps),
        "--dt",
        str(args.dt),
        "--nu0",
        str(args.nu0),
        "--Cs",
        str(args.Cs),
        "--stats-every",
        str(args.stats_every),
        "--viz-every",
        str(args.viz_every),
        "--progress-every",
        str(args.progress_every),
        "--out-dir",
        str(args.out_dir),
        "--prefix",
        args.prefix,
        "--fft-backend",
        args.fft_backend,
        "--spectral-truncation",
        args.spectral_truncation,
        "--trunc-alpha",
        str(args.trunc_alpha),
        "--trunc-power",
        str(args.trunc_power),
    ]
    return _run(cmd, headless=args.headless)


def cmd_kernel(args: argparse.Namespace) -> int:
    if args.interactive:
        args.z0_npz = _prompt("z0 npz path", args.z0_npz)
        args.A_npz = _prompt("A npz path", args.A_npz)
        args.steps = _prompt_int("Steps", args.steps)
        args.decode_every = _prompt_int("decode-every", args.decode_every)
        args.observer = _prompt_choice("observer", ["metrics", "snapshots", "visualize", "none"], args.observer)
        args.backend = _prompt_choice("backend", ["vulkan", "accelerated", "cpu"], args.backend)
        args.fft_backend = _prompt_choice("fft-backend", ["vkfft-vulkan", "vkfft-opencl", "vkfft", "numpy"], args.fft_backend)
        args.op_backend = _prompt_choice("op-backend", ["vulkan", "accelerated", "cpu"], args.op_backend)
        args.require_gpu = _prompt_bool("require-gpu", args.require_gpu)
        args.metrics_json = _prompt("metrics-json", args.metrics_json)
        args.headless = _prompt_bool("Headless (MPLBACKEND=Agg)", args.headless)

    cmd = [
        sys.executable,
        _script_path("perf_kernel.py"),
        "--z0-npz",
        args.z0_npz,
        "--A-npz",
        args.A_npz,
        "--steps",
        str(args.steps),
        "--decode-every",
        str(args.decode_every),
        "--decode-backend",
        "vulkan",
        "--observer",
        args.observer,
        "--backend",
        args.backend,
        "--fft-backend",
        args.fft_backend,
        "--op-backend",
        args.op_backend,
        "--metrics-json",
        args.metrics_json,
    ]
    if args.require_gpu:
        cmd.append("--require-gpu")
    return _run(cmd, headless=args.headless)


def cmd_plot(args: argparse.Namespace) -> int:
    if args.interactive:
        args.input = _prompt("input (JSON/CSV)", args.input)
        args.output = _prompt("output PNG", args.output)
        args.format = _prompt_choice("format", ["json", "csv"], args.format)
        args.title = _prompt("title (optional)", args.title)
        args.headless = _prompt_bool("Headless (MPLBACKEND=Agg)", args.headless)

    cmd = [
        sys.executable,
        _script_path("scripts/plot_enstrophy.py"),
        "--input",
        args.input,
        "--output",
        args.output,
        "--format",
        args.format,
    ]
    if args.title:
        cmd += ["--title", args.title]
    return _run(cmd, headless=args.headless)


def cmd_compare(args: argparse.Namespace) -> int:
    if args.interactive:
        args.N = _prompt_int("Grid size N", args.N)
        args.steps = _prompt_int("Steps", args.steps)
        args.dt = _prompt_float("dt", args.dt)
        args.nu0 = _prompt_float("nu0", args.nu0)
        args.Cs = _prompt_float("Cs", args.Cs)
        args.seed = _prompt_int("seed", args.seed)
        args.stats_every = _prompt_int("stats-every", args.stats_every)
        args.fft_backend = _prompt_choice("fft-backend", ["vkfft-vulkan", "vkfft-opencl", "vkfft", "numpy"], args.fft_backend)
        args.headless = _prompt_bool("Headless (MPLBACKEND=Agg)", args.headless)

    cmd = [
        sys.executable,
        _script_path("scripts/compare_les_gpu_cpu.py"),
        "--N",
        str(args.N),
        "--steps",
        str(args.steps),
        "--dt",
        str(args.dt),
        "--nu0",
        str(args.nu0),
        "--Cs",
        str(args.Cs),
        "--seed",
        str(args.seed),
        "--stats-every",
        str(args.stats_every),
        "--fft-backend",
        args.fft_backend,
    ]
    return _run(cmd, headless=args.headless)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    les = sub.add_parser(
        "les",
        help="GPU LES run (vkFFT + Vulkan)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    les.add_argument("--interactive", action="store_true", help="prompt for params")
    les.add_argument("--N", type=int, default=512, help="grid size")
    les.add_argument("--steps", type=int, default=200000, help="number of steps")
    les.add_argument("--dt", type=float, default=0.01, help="time step")
    les.add_argument("--nu0", type=float, default=1e-4, help="base viscosity")
    les.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant")
    les.add_argument("--stats-every", type=int, default=200, help="emit enstrophy every K steps")
    les.add_argument("--viz-every", type=int, default=0, help="save PNG every K steps (0 disables)")
    les.add_argument("--progress-every", type=int, default=2000, help="print progress every K steps")
    les.add_argument("--out-dir", type=Path, default=Path("outputs"), help="output directory")
    les.add_argument("--prefix", type=str, default="les_gpu", help="output filename prefix")
    les.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="FFT backend")
    les.add_argument("--spectral-truncation", type=str, default="none", help="spectral truncation filter")
    les.add_argument("--trunc-alpha", type=float, default=36.0, help="exp truncation alpha")
    les.add_argument("--trunc-power", type=float, default=8.0, help="exp truncation power")
    les.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    les.set_defaults(func=cmd_les)

    kernel = sub.add_parser(
        "kernel",
        help="Kernel rollout + decode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    kernel.add_argument("--interactive", action="store_true", help="prompt for params")
    kernel.add_argument("--z0-npz", dest="z0_npz", type=str, default="outputs/kernel_N128_z0.npz", help="z0 npz with proxy state")
    kernel.add_argument("--A-npz", dest="A_npz", type=str, default="outputs/kernel_N128_A.npz", help="operator npz with key A")
    kernel.add_argument("--steps", type=int, default=20000, help="rollout steps")
    kernel.add_argument("--decode-every", type=int, default=200, help="decode every K steps")
    kernel.add_argument("--observer", type=str, default="snapshots", help="observer policy")
    kernel.add_argument("--backend", type=str, default="vulkan", help="dashiCORE backend")
    kernel.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="FFT backend for decode")
    kernel.add_argument("--op-backend", type=str, default="vulkan", help="operator backend")
    kernel.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    kernel.add_argument("--metrics-json", type=str, default="outputs/perf_snapshots_gpu.json", help="optional metrics output file")
    kernel.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    kernel.set_defaults(func=cmd_kernel)

    plot = sub.add_parser(
        "plot",
        help="Plot enstrophy from JSON/CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plot.add_argument("--interactive", action="store_true", help="prompt for params")
    plot.add_argument("--input", type=str, default="outputs/perf_snapshots_gpu.json", help="input JSON/CSV file")
    plot.add_argument("--output", type=str, default="outputs/enstrophy.png", help="output PNG path")
    plot.add_argument("--format", type=str, default="json", help="input format")
    plot.add_argument("--title", type=str, default="", help="optional plot title")
    plot.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    plot.set_defaults(func=cmd_plot)

    compare = sub.add_parser(
        "compare",
        help="Compare GPU LES vs CPU baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    compare.add_argument("--interactive", action="store_true", help="prompt for params")
    compare.add_argument("--N", type=int, default=64, help="grid size")
    compare.add_argument("--steps", type=int, default=50, help="number of steps")
    compare.add_argument("--dt", type=float, default=0.01, help="time step")
    compare.add_argument("--nu0", type=float, default=1e-4, help="base viscosity")
    compare.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant")
    compare.add_argument("--seed", type=int, default=0, help="initial condition seed")
    compare.add_argument("--stats-every", type=int, default=10, help="print enstrophy every K steps")
    compare.add_argument("--fft-backend", type=str, default="vkfft-vulkan", help="FFT backend for GPU")
    compare.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    compare.set_defaults(func=cmd_compare)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
