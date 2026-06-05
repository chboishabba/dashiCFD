#!/usr/bin/env python3
"""Visualize Sprint 49 red/green/blue packet threads in a 3D volume.

The visualizer consumes a Sprint 49 material-parent output directory, rebuilds
the deterministic ``K{K}_cell{cell}`` packet masks from the source truth volume,
and renders the selected snapshot as:

* red:   ``plus``
* green: ``zero``
* blue:  ``minus``

Empty/nil cells are stored as NaN in the exported volume and are not plotted.
The rendered thread points use configurable opacity, defaulting to 30%.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import ns_sprint54_no2cycle_resolution_cadence_audit as sprint54
import ns_sprint55_lagrangian_stretch_action_audit as sprint55
import ns_sprint56_packet_local_stretch_action_audit as sprint56


STATE_TO_LABEL = {"minus": -1.0, "zero": 0.0, "plus": 1.0}
STATE_TO_COLOR = {
    "minus": (0.10, 0.35, 1.00),
    "zero": (0.10, 0.85, 0.25),
    "plus": (1.00, 0.12, 0.10),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Sprint 49 material-parent output directory")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--truth-root", type=Path, default=Path("."), help="root used to resolve source_truth")
    p.add_argument("--time", default="latest", help="'latest', 'first', or an exact table time")
    p.add_argument("--state-column", choices=["child_state", "parent_state"], default="child_state")
    p.add_argument("--trit-source", choices=["child_state", "parent_state", "raw_action"], default=None)
    p.add_argument("--raw-action-csv", type=Path, default=None, help="Sprint 59 ns_raw_packet_stretch_action.csv")
    p.add_argument("--raw-action-threshold", type=float, default=0.0)
    p.add_argument("--all-times", action="store_true", help="render every available time and optionally animate")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--animation-format", choices=["none", "gif", "webm", "both"], default="none")
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--camera-elev", type=float, default=24.0)
    p.add_argument("--camera-azim", type=float, default=-56.0)
    p.add_argument("--alpha", type=float, default=0.30)
    p.add_argument("--max-points", type=int, default=80000, help="max plotted voxels after stratified sampling")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--background", choices=["transparent", "black", "white"], default="transparent")
    return p.parse_args()


def _read_table(input_dir: Path) -> list[dict[str, str]]:
    path = input_dir / "ns_material_parent_table.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"time", "child_packet_id", "child_state", "parent_state"}
    missing = sorted(required.difference(rows[0].keys() if rows else []))
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    return rows


def _choose_time(rows: list[dict[str, str]], requested: str) -> float:
    times = sorted({float(row["time"]) for row in rows})
    if not times:
        raise SystemExit("material-parent table has no rows")
    if requested == "latest":
        return times[-1]
    if requested == "first":
        return times[0]
    target = float(requested)
    if target not in times:
        nearest = min(times, key=lambda t: abs(t - target))
        raise SystemExit(f"time {target} not present; nearest table time is {nearest}")
    return target


def _available_times(rows: list[dict[str, str]], args: argparse.Namespace) -> list[float]:
    times = sorted({float(row["time"]) for row in rows})
    if not times:
        raise SystemExit("material-parent table has no rows")
    if bool(args.all_times):
        stride = max(1, int(args.frame_stride))
        return times[::stride]
    return [_choose_time(rows, str(args.time))]


def _trit_source(args: argparse.Namespace) -> str:
    return str(args.trit_source or args.state_column)


def _truth_meta(input_dir: Path, truth_root: Path) -> dict[str, Any]:
    summary = json.loads((input_dir / "ns_material_parent_summary.json").read_text(encoding="utf-8"))
    source_truth = str(summary.get("source_truth") or "")
    truth_path = Path(source_truth)
    if source_truth and not truth_path.is_absolute():
        truth_path = truth_root / truth_path
    return {
        "run": input_dir.name,
        "truth_path": str(truth_path),
        "packet_grid": int(summary.get("packet_grid") or 8),
        "source_truth": source_truth,
    }


def _load_truth_shape(meta: dict[str, Any]) -> tuple[int, float]:
    path = Path(str(meta["truth_path"]))
    if not path.exists():
        raise SystemExit(f"source truth does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "omega_snapshots" not in data.files:
            raise SystemExit(f"{path} lacks omega_snapshots")
        omega = np.asarray(data["omega_snapshots"])
        meta_json = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
    if omega.ndim != 5 or omega.shape[-1] != 3:
        raise SystemExit(f"{path} omega_snapshots shape is incompatible: {omega.shape}")
    return int(omega.shape[1]), float(meta_json.get("domain_length") or (2.0 * math.pi))


def _build_label_volume(rows: list[dict[str, str]], n: int, L: float, packet_grid: int, state_column: str) -> np.ndarray:
    shell_map = sprint54._build_shell_map(n, L)
    cell_map = sprint56._cell_map(n, packet_grid)
    labels = np.full((n, n, n), np.nan, dtype=np.float32)
    for row in rows:
        packet_id = str(row["child_packet_id"])
        parsed = sprint56._parse_packet_id(packet_id)
        if parsed is None:
            continue
        k, cell = parsed
        state = str(row.get(state_column, ""))
        if state not in STATE_TO_LABEL:
            continue
        mask = (shell_map == k) & (cell_map == cell)
        if bool(np.any(mask)):
            labels[mask] = STATE_TO_LABEL[state]
    return labels


def _raw_action_state(row: dict[str, str], threshold: float) -> str:
    net_key = "A_raw_net" if "A_raw_net" in row else "raw_A_net"
    if net_key not in row and str(row.get("lagrangian_trit_after_integration", "")) in STATE_TO_LABEL:
        return str(row["lagrangian_trit_after_integration"])
    net = float(row.get(net_key) or 0.0)
    if net > threshold:
        return "plus"
    if net < -threshold:
        return "minus"
    return "zero"


def _read_raw_action_rows(path: Path | None, threshold: float, run_filter: str) -> dict[float, list[dict[str, str]]]:
    if path is None:
        raise SystemExit("--trit-source raw_action requires --raw-action-csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"packet_id"}
    missing = sorted(required.difference(rows[0].keys() if rows else []))
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    by_time: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("run", run_filter)) != run_filter:
            continue
        item = dict(row)
        item["child_packet_id"] = str(row.get("packet_id", ""))
        item["raw_action_state"] = _raw_action_state(row, threshold)
        time_key = "t_end" if "t_end" in row else "time"
        if time_key not in row:
            raise SystemExit(f"{path} raw-action rows need either time or t_end")
        by_time.setdefault(float(row[time_key]), []).append(item)
    return by_time


def _sample_points(labels: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords_by_state = []
    labels_by_state = []
    for label in [-1.0, 0.0, 1.0]:
        coords = np.argwhere(labels == label)
        if coords.size:
            coords_by_state.append(coords)
            labels_by_state.append(np.full(len(coords), label, dtype=np.float32))
    if not coords_by_state:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.float32)
    total = sum(len(coords) for coords in coords_by_state)
    sampled_coords = []
    sampled_labels = []
    for coords, lab in zip(coords_by_state, labels_by_state):
        keep = len(coords)
        if total > max_points:
            keep = max(1, int(round(max_points * len(coords) / total)))
        if keep < len(coords):
            idx = rng.choice(len(coords), size=keep, replace=False)
            coords = coords[idx]
            lab = lab[idx]
        sampled_coords.append(coords)
        sampled_labels.append(lab)
    return np.concatenate(sampled_coords), np.concatenate(sampled_labels)


def _rgba_for_labels(label_values: np.ndarray, alpha: float) -> list[tuple[float, float, float, float]]:
    out = []
    for value in label_values:
        if value < 0:
            rgb = STATE_TO_COLOR["minus"]
        elif value > 0:
            rgb = STATE_TO_COLOR["plus"]
        else:
            rgb = STATE_TO_COLOR["zero"]
        out.append((*rgb, alpha))
    return out


def _configure_axes(ax: Any, n: int, background: str) -> None:
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    if background == "black":
        ax.set_facecolor("black")
        ax.figure.patch.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")
    elif background == "transparent":
        ax.set_facecolor((0, 0, 0, 0))
        ax.figure.patch.set_alpha(0)


def _write_3d_scatter(path: Path, labels: np.ndarray, title: str, args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords, state_labels = _sample_points(labels, int(args.max_points), int(args.seed))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    _configure_axes(ax, labels.shape[0], str(args.background))
    ax.view_init(elev=float(args.camera_elev), azim=float(args.camera_azim))
    ax.set_title(title, color="white" if args.background == "black" else "black")
    if len(coords):
        colors = _rgba_for_labels(state_labels, float(args.alpha))
        ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0], c=colors, s=float(args.point_size), depthshade=False)
    fig.tight_layout()
    fig.savefig(path, dpi=int(args.dpi), transparent=args.background == "transparent")
    plt.close(fig)
    return int(len(coords))


def _write_projections(path: Path, labels: np.ndarray, title: str, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords, state_labels = _sample_points(labels, int(args.max_points), int(args.seed))
    bg = str(args.background)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if bg == "black":
        fig.patch.set_facecolor("black")
    elif bg == "transparent":
        fig.patch.set_alpha(0)
    colors = _rgba_for_labels(state_labels, float(args.alpha))
    panels = [
        ("xy", 2, 1),
        ("xz", 2, 0),
        ("yz", 1, 0),
    ]
    for ax, (name, xidx, yidx) in zip(axes, panels):
        if bg == "black":
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            title_color = "white"
        else:
            if bg == "transparent":
                ax.set_facecolor((0, 0, 0, 0))
            title_color = "black"
        ax.set_title(name, color=title_color)
        ax.set_xlim(0, labels.shape[0])
        ax.set_ylim(0, labels.shape[0])
        ax.set_aspect("equal", adjustable="box")
        if len(coords):
            ax.scatter(coords[:, xidx], coords[:, yidx], c=colors, s=float(args.point_size), linewidths=0)
    fig.suptitle(title, color="white" if bg == "black" else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=int(args.dpi), transparent=bg == "transparent")
    plt.close(fig)


def _write_animation_gif(path: Path, frames: list[Path], fps: float) -> None:
    import imageio.v2 as imageio

    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(path, images, duration=1.0 / max(float(fps), 0.01))


def _write_animation_webm(path: Path, frame_dir: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    pattern = str(frame_dir / "frame_%05d.png")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            f"{float(fps):g}",
            "-i",
            pattern,
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-b:v",
            "0",
            "-crf",
            "34",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def _counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "minus": int(np.count_nonzero(labels == -1.0)),
        "zero": int(np.count_nonzero(labels == 0.0)),
        "plus": int(np.count_nonzero(labels == 1.0)),
        "transparent_nil": int(np.count_nonzero(~np.isfinite(labels))),
    }


def _write_snapshot(
    args: argparse.Namespace,
    input_name: str,
    selected_time: float,
    labels: np.ndarray,
    trit_source: str,
    frame_index: int | None = None,
) -> dict[str, Any]:
    if frame_index is None:
        stem = f"{input_name}_t{selected_time:g}_{trit_source}_rgb_threads"
        frame_dir = args.out_dir
        scatter_path = args.out_dir / f"{stem}_3d.png"
    else:
        frame_dir = args.out_dir / f"{input_name}_{trit_source}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{input_name}_t{selected_time:g}_{trit_source}_rgb_threads"
        scatter_path = frame_dir / f"frame_{frame_index:05d}.png"
    volume_path = args.out_dir / f"{stem}.npz"
    projection_path = args.out_dir / f"{stem}_projections.png"
    np.savez_compressed(
        volume_path,
        labels=labels,
        label_semantics=json.dumps({"nan": "transparent_nil", "-1": "minus_blue", "0": "zero_green", "1": "plus_red"}),
        alpha=float(args.alpha),
        time=selected_time,
        source_input=str(args.input),
        trit_source=trit_source,
    )
    title = f"{input_name} t={selected_time:g} {trit_source} R/G/B alpha={float(args.alpha):.2f}"
    plotted = _write_3d_scatter(scatter_path, labels, title, args)
    _write_projections(projection_path, labels, title, args)
    return {
        "time": selected_time,
        "trit_source": trit_source,
        "plotted_points": plotted,
        "counts": _counts(labels),
        "volume_npz": str(volume_path),
        "scatter_png": str(scatter_path),
        "projections_png": str(projection_path),
    }


def main() -> None:
    args = _parse_args()
    rows = _read_table(args.input)
    trit_source = _trit_source(args)
    meta = _truth_meta(args.input, args.truth_root)
    n, L = _load_truth_shape(meta)
    times = _available_times(rows, args)
    raw_by_time = (
        _read_raw_action_rows(args.raw_action_csv, float(args.raw_action_threshold), args.input.name)
        if trit_source == "raw_action"
        else {}
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Path] = []
    frame_summaries: list[dict[str, Any]] = []
    for idx, selected_time in enumerate(times):
        if trit_source == "raw_action":
            selected = raw_by_time.get(selected_time, [])
            labels = _build_label_volume(selected, n, L, int(meta["packet_grid"]), "raw_action_state")
        else:
            selected = [row for row in rows if float(row["time"]) == selected_time]
            labels = _build_label_volume(selected, n, L, int(meta["packet_grid"]), trit_source)
        frame = _write_snapshot(
            args,
            args.input.name,
            selected_time,
            labels,
            trit_source,
            idx if bool(args.all_times) else None,
        )
        frame["selected_table_rows"] = len(selected)
        frame_summaries.append(frame)
        frames.append(Path(str(frame["scatter_png"])))

    gif_path = None
    webm_path = None
    webm_written = False
    fmt = str(args.animation_format)
    if bool(args.all_times) and frames and fmt in {"gif", "both"}:
        gif_path = args.out_dir / f"{args.input.name}_{trit_source}_rgb_threads.gif"
        _write_animation_gif(gif_path, frames, float(args.fps))
    if bool(args.all_times) and frames and fmt in {"webm", "both"}:
        webm_path = args.out_dir / f"{args.input.name}_{trit_source}_rgb_threads.webm"
        webm_written = _write_animation_webm(webm_path, frames[0].parent, float(args.fps))

    total_counts = {"minus": 0, "zero": 0, "plus": 0, "transparent_nil": 0}
    for frame in frame_summaries:
        for key in total_counts:
            total_counts[key] += int(frame["counts"][key])
    summary = {
        "contract": "ns_rgb_thread_volume_visualization",
        "input": str(args.input),
        "source_truth": meta["source_truth"],
        "time": frame_summaries[0]["time"] if frame_summaries else None,
        "times": times,
        "all_times": bool(args.all_times),
        "frame_count": len(frame_summaries),
        "state_column": str(args.state_column),
        "trit_source": trit_source,
        "raw_action_threshold": float(args.raw_action_threshold) if trit_source == "raw_action" else None,
        "alpha": float(args.alpha),
        "fps": float(args.fps),
        "N": n,
        "packet_grid": int(meta["packet_grid"]),
        "selected_table_rows": sum(int(frame["selected_table_rows"]) for frame in frame_summaries),
        "plotted_points": sum(int(frame["plotted_points"]) for frame in frame_summaries),
        "counts": total_counts if bool(args.all_times) else frame_summaries[0]["counts"],
        "frames": frame_summaries,
        "volume_npz": frame_summaries[0]["volume_npz"] if frame_summaries else None,
        "scatter_png": frame_summaries[0]["scatter_png"] if frame_summaries else None,
        "projections_png": frame_summaries[0]["projections_png"] if frame_summaries else None,
        "gif": str(gif_path) if gif_path is not None else None,
        "webm": str(webm_path) if webm_written and webm_path is not None else None,
        "nil_nan_transparent": True,
    }
    if bool(args.all_times):
        summary_path = args.out_dir / f"{args.input.name}_{trit_source}_rgb_threads_animation_summary.json"
    else:
        summary_path = args.out_dir / f"{args.input.name}_t{times[0]:g}_{trit_source}_rgb_threads_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for frame in frame_summaries[:3]:
        print(f"[ns_rgb_thread_volume_visualizer] wrote {frame['scatter_png']}")
    if len(frame_summaries) > 3:
        print(f"[ns_rgb_thread_volume_visualizer] wrote {len(frame_summaries)} frame PNGs")
    if gif_path is not None:
        print(f"[ns_rgb_thread_volume_visualizer] wrote {gif_path}")
    if webm_written and webm_path is not None:
        print(f"[ns_rgb_thread_volume_visualizer] wrote {webm_path}")
    print(f"[ns_rgb_thread_volume_visualizer] wrote {summary_path}")
    print(f"[ns_rgb_thread_volume_visualizer] counts={summary['counts']} plotted_points={summary['plotted_points']}")


if __name__ == "__main__":
    main()
