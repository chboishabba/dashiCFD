#!/usr/bin/env python3
"""Lightweight CPU/GPU sampler for quick bottleneck sanity checks."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _find_gpu_busy_path() -> Optional[Path]:
    for card in (Path("/sys/class/drm")).glob("card*"):
        busy = card / "device" / "gpu_busy_percent"
        if busy.is_file():
            return busy
    return None


def _find_vram_paths() -> Tuple[Optional[Path], Optional[Path]]:
    for card in (Path("/sys/class/drm")).glob("card*"):
        used = card / "device" / "mem_info_vram_used"
        total = card / "device" / "mem_info_vram_total"
        if used.is_file() and total.is_file():
            return used, total
    return None, None


def _read_cpu_stat() -> Tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as f:
        line = f.readline()
    parts = line.split()
    if parts[0] != "cpu":
        return 0, 0
    vals = [int(x) for x in parts[1:]]
    total = sum(vals)
    idle = vals[3] + vals[4] if len(vals) > 4 else vals[3]
    return total, idle


def _read_proc_cpu(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
            parts = f.read().split()
        utime = int(parts[13])
        stime = int(parts[14])
        return utime + stime
    except OSError:
        return None


@dataclass
class Sample:
    ts: float
    cpu_pct: float
    proc_cpu_pct: Optional[float]
    gpu_busy_pct: Optional[float]
    vram_used_mb: Optional[float]
    vram_total_mb: Optional[float]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", type=float, default=0.5, help="sample interval seconds (default: 0.5)")
    p.add_argument("--duration", type=float, default=0.0, help="total duration seconds (0 = run forever)")
    p.add_argument("--pid", type=int, default=None, help="optional PID to report process CPU%%")
    p.add_argument("--csv", type=Path, default=None, help="optional CSV output path")
    args = p.parse_args()

    busy_path = _find_gpu_busy_path()
    vram_used_path, vram_total_path = _find_vram_paths()
    clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    ncpu = os.cpu_count() or 1

    last_total, last_idle = _read_cpu_stat()
    last_proc = _read_proc_cpu(args.pid) if args.pid else None
    start_ts = time.time()
    last_ts = start_ts

    header = "ts,cpu_pct,proc_cpu_pct,gpu_busy_pct,vram_used_mb,vram_total_mb"
    print(header)
    csv_f = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(args.csv, "w", encoding="utf-8")
        csv_f.write(header + "\n")

    try:
        while True:
            time.sleep(max(args.interval, 0.01))
            now = time.time()
            total, idle = _read_cpu_stat()
            dt = max(now - last_ts, 1e-9)
            delta_total = total - last_total
            delta_idle = idle - last_idle
            cpu_pct = 100.0 * (1.0 - (delta_idle / max(delta_total, 1)))

            proc_pct = None
            if args.pid:
                proc_now = _read_proc_cpu(args.pid)
                if proc_now is not None and last_proc is not None:
                    delta_proc = proc_now - last_proc
                    proc_pct = 100.0 * (delta_proc / (clk_tck * dt * ncpu))
                last_proc = proc_now

            gpu_busy = None
            if busy_path is not None:
                raw = _read_text(busy_path)
                if raw is not None:
                    try:
                        gpu_busy = float(raw)
                    except ValueError:
                        gpu_busy = None

            vram_used = None
            vram_total = None
            if vram_used_path is not None and vram_total_path is not None:
                u = _read_text(vram_used_path)
                t = _read_text(vram_total_path)
                try:
                    if u is not None:
                        vram_used = float(u) / (1024 * 1024)
                    if t is not None:
                        vram_total = float(t) / (1024 * 1024)
                except ValueError:
                    vram_used = None
                    vram_total = None

            sample = Sample(
                ts=now,
                cpu_pct=cpu_pct,
                proc_cpu_pct=proc_pct,
                gpu_busy_pct=gpu_busy,
                vram_used_mb=vram_used,
                vram_total_mb=vram_total,
            )
            line = "{:.3f},{:.2f},{},{},{},{}".format(
                sample.ts,
                sample.cpu_pct,
                "" if sample.proc_cpu_pct is None else f"{sample.proc_cpu_pct:.2f}",
                "" if sample.gpu_busy_pct is None else f"{sample.gpu_busy_pct:.1f}",
                "" if sample.vram_used_mb is None else f"{sample.vram_used_mb:.1f}",
                "" if sample.vram_total_mb is None else f"{sample.vram_total_mb:.1f}",
            )
            print(line)
            if csv_f is not None:
                csv_f.write(line + "\n")
                csv_f.flush()

            last_total, last_idle = total, idle
            last_ts = now
            if args.duration > 0 and (now - start_ts) >= args.duration:
                break
    finally:
        if csv_f is not None:
            csv_f.close()


if __name__ == "__main__":
    main()
