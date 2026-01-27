#!/usr/bin/env python3
"""Plot enstrophy vs time from perf_kernel metrics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("outputs/perf_snapshots_gpu.json"), help="metrics JSON path")
    p.add_argument("--output", type=Path, default=Path("outputs/enstrophy_kernel_only.png"), help="output PNG path")
    p.add_argument("--title", type=str, default="Kernel-only decode enstrophy", help="plot title")
    p.add_argument("--dpi", type=int, default=160, help="output DPI")
    p.add_argument("--figsize", type=str, default="8,4", help="figure size inches as W,H")
    p.add_argument("--format", type=str, choices=["auto", "json", "csv"], default="auto", help="input format")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fmt = args.format
    if fmt == "auto":
        fmt = "csv" if args.input.suffix.lower() == ".csv" else "json"

    pts = []
    if fmt == "json":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = [(m["t"], m["enstrophy"]) for m in data.get("decode_metrics", []) if "enstrophy" in m]
        if not pts:
            raise SystemExit("No enstrophy in metrics JSON. Run perf_kernel.py with --observer snapshots.")
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            if len(header) < 2 or header[0] != "step":
                raise SystemExit("CSV must start with header: step,enstrophy")
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                pts.append((int(parts[0]), float(parts[1])))
        if not pts:
            raise SystemExit("No enstrophy in CSV.")

    t, z = zip(*pts)
    w_str, h_str = args.figsize.split(",")
    figsize = (float(w_str), float(h_str))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize)
    plt.plot(t, z, lw=1.5)
    plt.xlabel("t")
    plt.ylabel("Enstrophy")
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
