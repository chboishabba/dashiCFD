#!/usr/bin/env python3
"""
Create an H.264 MP4 from a glob of PNG frames using VAAPI acceleration.

Example:
    python host_ffmpeg.py "outputs/v4_*_compare.png" --output outputs/v4_compare.mp4
"""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pattern",
        help="glob pattern for input frames (quote to avoid shell expansion)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.mp4",
        help="output video path (default: output.mp4)",
    )
    parser.add_argument(
        "--framerate",
        "-r",
        type=int,
        default=60,
        help="input frame rate in FPS (default: 60)",
    )
    parser.add_argument(
        "--qp",
        type=int,
        default=20,
        help="H.264 quantizer parameter (smaller is higher quality)",
    )
    parser.add_argument(
        "--vaapi-device",
        default="/dev/dri/renderD128",
        help="VAAPI render node (default: /dev/dri/renderD128)",
    )
    parser.add_argument(
        "--libva-driver",
        default="radeonsi",
        help="LIBVA_DRIVER_NAME value (default: radeonsi)",
    )
    parser.add_argument(
        "--codec",
        default="h264_vaapi",
        help="VAAPI encoder (e.g., h264_vaapi, hevc_vaapi)",
    )
    parser.add_argument(
        "--profile",
        default="high",
        help="H.264 profile (default: high)",
    )
    parser.add_argument(
        "--overwrite",
        "-y",
        action="store_true",
        help="allow overwriting the output file",
    )
    parser.add_argument(
        "--movflags",
        default="+faststart",
        help="ffmpeg -movflags value (default: +faststart)",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="additional ffmpeg args appended verbatim (use multiple times)",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = ["ffmpeg"]
    if args.overwrite:
        cmd.append("-y")
    cmd.extend(
        [
            "-framerate",
            str(args.framerate),
            "-pattern_type",
            "glob",
            "-i",
            args.pattern,
            "-vaapi_device",
            args.vaapi_device,
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            args.codec,
            "-profile:v",
            args.profile,
            "-qp",
            str(args.qp),
            "-movflags",
            args.movflags,
        ]
    )
    cmd.extend(args.extra)
    cmd.append(args.output)
    return cmd


def main() -> None:
    args = parse_args()

    matches = sorted(glob.glob(args.pattern))
    if not matches:
        print(f"No files matched pattern: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args)
    env = os.environ.copy()
    env["LIBVA_DRIVER_NAME"] = args.libva_driver

    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    try:
        subprocess.run(cmd, env=env, check=True)
    except FileNotFoundError:
        print("ffmpeg not found on PATH", file=sys.stderr)
        sys.exit(127)
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
