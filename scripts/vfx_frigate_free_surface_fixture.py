#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from free_surface_shallow_water import Grid, RunConfig, primitive, run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the reduced dashiCFD free-surface VFX fixture.")
    p.add_argument("--nx", type=int, default=160)
    p.add_argument("--ny", type=int, default=80)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--inflow-u", type=float, default=7.0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/vfx_frigate_free_surface"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grid = Grid(nx=args.nx, ny=args.ny)
    cfg = RunConfig(seconds=args.seconds, inflow_u=args.inflow_u)
    U, solid, receipt = run(grid, cfg)
    h, u, v = primitive(U)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "free_surface_state.npz",
        h=h,
        u=u,
        v=v,
        solid=solid,
        dx=np.array(grid.dx),
        dy=np.array(grid.dy),
    )
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
