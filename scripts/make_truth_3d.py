#!/usr/bin/env python3
"""Generate 3D periodic incompressible Navier-Stokes truth artifacts.

Example:
  python3 scripts/make_truth_3d.py \
    --N 32 --steps 200 --save-every 10 --dt 0.002 --nu0 0.001 \
    --seed 0 --out outputs/truth3d/ns3d_N32_seed0.npz
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


Array = np.ndarray


def _run_text(cmd: List[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    text = (result.stdout or result.stderr or "").strip()
    return text if text else f"exit={result.returncode}"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unavailable"


def collect_host_provenance(args: argparse.Namespace, gpu_device: Dict[str, object] | None = None) -> Dict[str, object]:
    uname = platform.uname()
    env_keys = [
        "VK_ICD_FILENAMES",
        "VK_LAYER_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "HSA_OVERRIDE_GFX_VERSION",
        "ROC_ENABLE_PRE_VEGA",
        "HSA_ENABLE_SDMA",
        "HIP_LAUNCH_BLOCKING",
    ]
    icd_paths = [p for p in os.environ.get("VK_ICD_FILENAMES", "").split(":") if p]
    if not icd_paths:
        for parent in (Path("/usr/share/vulkan/icd.d"), Path("/etc/vulkan/icd.d")):
            if parent.is_dir():
                icd_paths.extend(str(p) for p in sorted(parent.glob("*.json")))
    pacman = shutil.which("pacman")
    icd_packages = {}
    if pacman:
        for icd in icd_paths:
            icd_packages[icd] = _run_text([pacman, "-Qo", icd])
    return {
        "captured_at_unix": time.time(),
        "uname": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
            "raw": " ".join(uname),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "packages": {
            "numpy": np.__version__,
            "vulkan": _package_version("vulkan"),
            "pyvkfft": _package_version("pyvkfft"),
        },
        "commands": {
            "glslc": _run_text([shutil.which("glslc") or "glslc", "--version"]) if shutil.which("glslc") else "unavailable",
            "vulkaninfo_summary": _run_text(["vulkaninfo", "--summary"], timeout=10.0) if shutil.which("vulkaninfo") else "unavailable",
        },
        "environment": {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
        "vulkan": {
            "icd_paths": icd_paths,
            "icd_packages": icd_packages,
            "device": gpu_device or {},
        },
        "generator": {
            "backend": args.backend,
            "fft_backend": args.fft_backend if args.backend == "gpu" else "numpy",
            "argv": sys.argv,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu", help="truth backend")
    p.add_argument(
        "--fft-backend",
        choices=["vkfft", "vkfft-vulkan"],
        default="vkfft-vulkan",
        help="GPU FFT backend used when --backend gpu",
    )
    p.add_argument("--N", type=int, default=32, help="grid size per dimension")
    p.add_argument("--steps", type=int, default=200, help="number of solver steps")
    p.add_argument("--save-every", type=int, default=10, help="snapshot stride")
    p.add_argument("--dt", type=float, default=0.002, help="time step")
    p.add_argument("--nu0", type=float, default=0.001, help="kinematic viscosity")
    p.add_argument("--seed", type=int, default=0, help="random initial-condition seed")
    p.add_argument("--L", type=float, default=2.0 * math.pi, help="periodic box length")
    p.add_argument("--target-urms", type=float, default=1.0, help="initial RMS velocity after projection")
    p.add_argument("--k-min", type=float, default=1.0, help="minimum initialized Fourier radius")
    p.add_argument("--k-max", type=float, default=None, help="maximum initialized Fourier radius; default N/3")
    p.add_argument("--k-star", type=int, default=None, help="shell support cutoff; default derived from nu0")
    p.add_argument("--max-cfl", type=float, default=0.8, help="fail if saved CFL exceeds this value")
    p.add_argument("--div-tol", type=float, default=1e-8, help="fail if saved div-u RMS exceeds this")
    p.add_argument("--curl-tol", type=float, default=1e-8, help="fail if stored omega disagrees with curl(u)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="snapshot storage dtype")
    p.add_argument("--omit-velocity", action="store_true", help="omit velocity_snapshots from NPZ")
    p.add_argument("--out", type=Path, default=Path("outputs/truth3d/ns3d_N32_seed0.npz"), help="output NPZ path")
    p.add_argument("--progress-every", type=int, default=0, help="print progress every K steps")
    p.add_argument("--update-manifest", action="store_true", help="update outputs/truth3d_manifest.json")
    return p.parse_args()


def make_grid(n: int, length: float) -> Tuple[float, Array, Array, Array, Array, Array]:
    dx = float(length) / int(n)
    k = np.fft.fftfreq(n, d=dx) * 2.0 * math.pi
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    return dx, kx, ky, kz, k2, np.sqrt(k2)


def dealias_mask(k_radius: Array, n: int) -> Array:
    # With L=2*pi, integer Fourier modes have |k_i|=integer.  The 2/3 rule
    # keeps component modes with |k_i| <= N/3; the radius mask is stricter but
    # stable and simple for this artifact generator.
    return k_radius <= (float(n) / 3.0)


def component_dealias_mask(kx: Array, ky: Array, kz: Array, n: int, length: float) -> Array:
    cutoff = (float(n) / 3.0) * (2.0 * math.pi / float(length))
    return (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff) & (np.abs(kz) <= cutoff)


def project_leray(u_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array) -> Array:
    out = np.array(u_hat, dtype=np.complex128, copy=True)
    dot = kx * out[..., 0] + ky * out[..., 1] + kz * out[..., 2]
    inv_k2 = np.zeros_like(k2, dtype=np.float64)
    np.divide(1.0, k2, out=inv_k2, where=k2 > 0.0)
    out[..., 0] -= kx * dot * inv_k2
    out[..., 1] -= ky * dot * inv_k2
    out[..., 2] -= kz * dot * inv_k2
    out[0, 0, 0, :] = 0.0
    return out


def apply_dealias(u_hat: Array, mask: Array) -> Array:
    out = np.array(u_hat, dtype=np.complex128, copy=True)
    out[~mask, :] = 0.0
    out[0, 0, 0, :] = 0.0
    return out


def ifft_vec(u_hat: Array) -> Array:
    return np.fft.ifftn(u_hat, axes=(0, 1, 2)).real


def fft_vec(u: Array) -> Array:
    return np.fft.fftn(u, axes=(0, 1, 2))


def curl_hat(u_hat: Array, kx: Array, ky: Array, kz: Array) -> Array:
    out = np.empty_like(u_hat, dtype=np.complex128)
    out[..., 0] = 1j * (ky * u_hat[..., 2] - kz * u_hat[..., 1])
    out[..., 1] = 1j * (kz * u_hat[..., 0] - kx * u_hat[..., 2])
    out[..., 2] = 1j * (kx * u_hat[..., 1] - ky * u_hat[..., 0])
    return out


def divergence(u_hat: Array, kx: Array, ky: Array, kz: Array) -> Array:
    div_hat = 1j * (kx * u_hat[..., 0] + ky * u_hat[..., 1] + kz * u_hat[..., 2])
    return np.fft.ifftn(div_hat).real


def spectral_derivative(u_hat: Array, k_comp: Array, component: int) -> Array:
    return np.fft.ifftn(1j * k_comp * u_hat[..., component]).real


def nonlinear_rhs(u_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array, mask: Array, nu0: float) -> Array:
    u = ifft_vec(u_hat)
    grad = np.empty(u.shape + (3,), dtype=np.float64)
    for comp in range(3):
        grad[..., comp, 0] = spectral_derivative(u_hat, kx, comp)
        grad[..., comp, 1] = spectral_derivative(u_hat, ky, comp)
        grad[..., comp, 2] = spectral_derivative(u_hat, kz, comp)
    adv = np.empty_like(u)
    for comp in range(3):
        adv[..., comp] = (
            u[..., 0] * grad[..., comp, 0]
            + u[..., 1] * grad[..., comp, 1]
            + u[..., 2] * grad[..., comp, 2]
        )
    adv_hat = apply_dealias(fft_vec(adv), mask)
    rhs = -project_leray(adv_hat, kx, ky, kz, k2) - float(nu0) * k2[..., None] * u_hat
    rhs[0, 0, 0, :] = 0.0
    return apply_dealias(rhs, mask)


def rk2_step(u_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array, mask: Array, dt: float, nu0: float) -> Array:
    k1 = nonlinear_rhs(u_hat, kx, ky, kz, k2, mask, nu0)
    mid = apply_dealias(project_leray(u_hat + float(dt) * k1, kx, ky, kz, k2), mask)
    k2_rhs = nonlinear_rhs(mid, kx, ky, kz, k2, mask, nu0)
    next_hat = u_hat + 0.5 * float(dt) * (k1 + k2_rhs)
    return apply_dealias(project_leray(next_hat, kx, ky, kz, k2), mask)


def init_velocity_hat(args: argparse.Namespace, kx: Array, ky: Array, kz: Array, k2: Array, mask: Array) -> Array:
    rng = np.random.default_rng(args.seed)
    n = int(args.N)
    k_radius = np.sqrt(k2)
    component_cutoff = (float(n) / 3.0) * (2.0 * math.pi / float(args.L))
    k_max_default = math.sqrt(3.0) * component_cutoff
    k_max = float(args.k_max if args.k_max is not None else max(2.0, k_max_default))
    shell = (k_radius >= float(args.k_min)) & (k_radius <= k_max) & mask
    u = rng.standard_normal((n, n, n, 3))
    u_hat = fft_vec(u)
    envelope = np.zeros((n, n, n), dtype=np.float64)
    envelope[shell] = np.exp(-0.5 * (k_radius[shell] / max(k_max, 1e-12)) ** 2)
    u_hat *= envelope[..., None]
    u_hat = apply_dealias(project_leray(u_hat, kx, ky, kz, k2), mask)
    u_real = ifft_vec(u_hat)
    urms = math.sqrt(float(np.mean(np.sum(u_real * u_real, axis=-1))))
    if not math.isfinite(urms) or urms <= 0.0:
        raise RuntimeError("initial velocity has zero or non-finite RMS")
    u_hat *= float(args.target_urms) / urms
    return apply_dealias(project_leray(u_hat, kx, ky, kz, k2), mask)


def shell_energy_omega(omega_hat: Array, k_radius: Array) -> Array:
    shells = np.floor(k_radius + 1e-12).astype(np.int64)
    max_shell = int(shells.max())
    density = np.sum(np.abs(omega_hat) ** 2, axis=-1) / float(k_radius.size * k_radius.size)
    out = np.zeros(max_shell + 1, dtype=np.float64)
    for j in range(max_shell + 1):
        out[j] = float(density[shells == j].sum())
    return out


def derive_k_star(nu0: float, n: int) -> int:
    max_dealiased_radius = math.sqrt(3.0) * (float(n) / 3.0)
    feasible_tail_start = max(0, int(math.floor(max_dealiased_radius)) - 5)
    if nu0 > 0.0 and math.isfinite(nu0):
        raw = int(math.floor(0.75 * math.log2(1.0 / nu0)))
        return max(0, min(raw, feasible_tail_start))
    return max(0, min(n // 4, feasible_tail_start))


def snapshot_diagnostics(
    u_hat: Array,
    omega: Array,
    omega_hat: Array,
    *,
    dx: float,
    dt: float,
    kx: Array,
    ky: Array,
    kz: Array,
    k_radius: Array,
    k_star: int,
) -> Dict[str, object]:
    u = ifft_vec(u_hat)
    div = divergence(u_hat, kx, ky, kz)
    omega_from_u = ifft_vec(curl_hat(u_hat, kx, ky, kz))
    curl_rel = float(np.linalg.norm(omega_from_u - omega) / (np.linalg.norm(omega) + 1e-30))
    shell_e = shell_energy_omega(omega_hat, k_radius)
    tail = shell_e[int(k_star) :] if int(k_star) < shell_e.size else np.asarray([], dtype=np.float64)
    speed = np.linalg.norm(u, axis=-1)
    return {
        "u": u,
        "energy": 0.5 * float(np.mean(np.sum(u * u, axis=-1))),
        "enstrophy": 0.5 * float(np.mean(np.sum(omega * omega, axis=-1))),
        "max_abs_omega": float(np.max(np.linalg.norm(omega, axis=-1))),
        "cfl": float(np.max(speed) * float(dt) / float(dx)),
        "div_rms": float(math.sqrt(np.mean(div * div))),
        "div_linf": float(np.max(np.abs(div))),
        "curl_rel_l2": curl_rel,
        "shell_energy": shell_e,
        "shell_nonzero_count_at_or_above_k_star": int(np.count_nonzero(tail > 1e-20)),
    }


def validate_snapshot(diag: Dict[str, object], args: argparse.Namespace) -> None:
    scalar_keys = ["energy", "enstrophy", "max_abs_omega", "cfl", "div_rms", "div_linf", "curl_rel_l2"]
    for key in scalar_keys:
        value = float(diag[key])
        if not math.isfinite(value):
            raise RuntimeError(f"validation failed: {key} is non-finite")
    if float(diag["cfl"]) > float(args.max_cfl):
        raise RuntimeError(f"validation failed: CFL {diag['cfl']:.6g} exceeds --max-cfl {args.max_cfl}")
    if float(diag["div_rms"]) > float(args.div_tol):
        raise RuntimeError(f"validation failed: div_rms {diag['div_rms']:.6g} exceeds --div-tol {args.div_tol}")
    if float(diag["curl_rel_l2"]) > float(args.curl_tol):
        raise RuntimeError(f"validation failed: curl_rel_l2 {diag['curl_rel_l2']:.6g} exceeds --curl-tol {args.curl_tol}")
    if int(diag["shell_nonzero_count_at_or_above_k_star"]) < 5:
        raise RuntimeError(
            "validation failed: fewer than 5 nonzero omega shells at or above K_star "
            f"({diag['shell_nonzero_count_at_or_above_k_star']})"
        )


def main() -> None:
    args = parse_args()
    if args.N < 16:
        raise SystemExit("--N must be at least 16")
    if args.steps < 0:
        raise SystemExit("--steps must be nonnegative")
    if args.save_every <= 0:
        raise SystemExit("--save-every must be positive")
    if args.dt <= 0.0 or args.nu0 < 0.0:
        raise SystemExit("--dt must be positive and --nu0 must be nonnegative")
    if args.backend == "gpu":
        if args.div_tol < 1e-6:
            print(f"[truth3d] backend=gpu raising div_tol from {args.div_tol:g} to 1e-6 for fp32 validation")
            args.div_tol = 1e-6
        if args.curl_tol < 1e-5:
            print(f"[truth3d] backend=gpu raising curl_tol from {args.curl_tol:g} to 1e-5 for fp32 validation")
            args.curl_tol = 1e-5

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dx, kx, ky, kz, k2, k_radius = make_grid(args.N, args.L)
    mask = component_dealias_mask(kx, ky, kz, args.N, args.L)
    k_star = int(args.k_star if args.k_star is not None else derive_k_star(args.nu0, args.N))
    storage_dtype = np.float32 if args.dtype == "float32" else np.float64

    snapshot_steps = list(range(0, args.steps + 1, args.save_every))
    if snapshot_steps[-1] != args.steps:
        snapshot_steps.append(args.steps)
    snapshot_set = set(snapshot_steps)

    u_hat = init_velocity_hat(args, kx, ky, kz, k2, mask)
    gpu_backend = None
    gpu_meta: Dict[str, object] = {}
    gpu_device: Dict[str, object] = {}
    if args.backend == "gpu":
        from vulkan_truth3d_backend import VulkanTruth3DBackend

        print(f"[truth3d] backend=gpu requested fft_backend={args.fft_backend}")
        print(f"[truth3d] VK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES', '<unset>')}")
        try:
            gpu_backend = VulkanTruth3DBackend(
                args.N,
                dt=args.dt,
                nu0=args.nu0,
                length=args.L,
                fft_backend=args.fft_backend,
            )
        except Exception as exc:
            raise SystemExit(
                "[truth3d] backend=gpu unavailable during Vulkan/vkFFT setup: "
                f"{type(exc).__name__}: {exc}. "
                "Set VK_ICD_FILENAMES to the RADV ICD, ensure the Vulkan device is visible, "
                "and run inside the gfx803/Nix environment if that is where the driver stack is exposed."
            ) from exc
        gpu_backend.set_initial_u_hat(u_hat)
        runtime_info = gpu_backend.runtime_info()
        gpu_device = gpu_backend.device_info()
        gpu_meta = {
            "fft_backend": args.fft_backend,
            "runtime": runtime_info,
            "device": gpu_device,
            "spv_shaders": [
                "complex_copy_3d",
                "real_to_complex_3d",
                "complex_to_real_3d",
                "derivative_hat_3d",
                "leray_project_3d",
                "dealias_3d",
                "curl_3d",
                "advect_vector_3d",
                "rhs_projected_ns_3d",
                "combine_vector_hat_3d",
            ],
        }
        print(
            "[truth3d] backend=gpu active "
            f"fft_plan_backend={runtime_info.get('ifft_plan_backend')} "
            f"device={gpu_device.get('device_name', 'unknown')}"
        )
    else:
        print("[truth3d] backend=cpu active fft_backend=numpy")
    omega_snapshots: List[Array] = []
    velocity_snapshots: List[Array] = []
    energies: List[float] = []
    enstrophies: List[float] = []
    max_abs_omega: List[float] = []
    cfl_values: List[float] = []
    div_rms_values: List[float] = []
    div_linf_values: List[float] = []
    curl_rel_values: List[float] = []
    shell_counts: List[int] = []
    shell_rows: List[Array] = []

    try:
        started = time.perf_counter()
        for step in range(args.steps + 1):
            if step in snapshot_set:
                if gpu_backend is not None:
                    u_hat_for_snapshot = np.asarray(gpu_backend.read_u_hat(), dtype=np.complex128)
                    omega_hat = np.asarray(gpu_backend.read_omega_hat(), dtype=np.complex128)
                else:
                    u_hat_for_snapshot = u_hat
                    omega_hat = curl_hat(u_hat_for_snapshot, kx, ky, kz)
                omega = ifft_vec(omega_hat)
                diag = snapshot_diagnostics(
                    u_hat_for_snapshot,
                    omega,
                    omega_hat,
                    dx=dx,
                    dt=args.dt,
                    kx=kx,
                    ky=ky,
                    kz=kz,
                    k_radius=k_radius,
                    k_star=k_star,
                )
                validate_snapshot(diag, args)
                omega_snapshots.append(omega.astype(storage_dtype, copy=True))
                if not args.omit_velocity:
                    velocity_snapshots.append(np.asarray(diag["u"]).astype(storage_dtype, copy=True))
                energies.append(float(diag["energy"]))
                enstrophies.append(float(diag["enstrophy"]))
                max_abs_omega.append(float(diag["max_abs_omega"]))
                cfl_values.append(float(diag["cfl"]))
                div_rms_values.append(float(diag["div_rms"]))
                div_linf_values.append(float(diag["div_linf"]))
                curl_rel_values.append(float(diag["curl_rel_l2"]))
                shell_counts.append(int(diag["shell_nonzero_count_at_or_above_k_star"]))
                shell_rows.append(np.asarray(diag["shell_energy"], dtype=np.float64))
            if args.progress_every and step % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = step / max(elapsed, 1e-12) if step > 0 else 0.0
                remaining = max(0, int(args.steps) - int(step))
                eta = remaining / max(rate, 1e-12) if step > 0 else float("inf")
                eta_text = "unknown" if not math.isfinite(eta) else f"{eta:.2f}s"
                print(f"[truth3d] step={step}/{args.steps} elapsed={elapsed:.2f}s steps/s={rate:.2f} eta={eta_text}")
            if step == args.steps:
                break
            if gpu_backend is not None:
                gpu_backend.step()
            else:
                u_hat = rk2_step(u_hat, kx, ky, kz, k2, mask, args.dt, args.nu0)
    finally:
        if gpu_backend is not None:
            gpu_backend.close()

    max_shell_len = max(row.size for row in shell_rows)
    shell_energy = np.zeros((len(shell_rows), max_shell_len), dtype=np.float64)
    for i, row in enumerate(shell_rows):
        shell_energy[i, : row.size] = row

    host_provenance = collect_host_provenance(args, gpu_device if args.backend == "gpu" else None)
    meta: Dict[str, object] = {
        "dimension": 3,
        "field": "omega",
        "has_velocity_snapshots": not args.omit_velocity,
        "N": int(args.N),
        "dt": float(args.dt),
        "nu0": float(args.nu0),
        "periodic": True,
        "projection": "leray",
        "dealiasing": "2/3",
        "seed": int(args.seed),
        "backend": args.backend,
        "fft_backend": args.fft_backend if args.backend == "gpu" else "numpy",
        "gpu": gpu_meta,
        "host_provenance": host_provenance,
        "forcing": "none",
        "domain_length": float(args.L),
        "axis_order": "z,y,x,component",
        "component_order": "x,y,z",
        "steps": int(args.steps),
        "save_every": int(args.save_every),
        "snapshots": len(omega_snapshots),
        "dtype": args.dtype,
        "target_urms": float(args.target_urms),
        "k_min": float(args.k_min),
        "k_max": float(args.k_max if args.k_max is not None else max(2.0, math.sqrt(3.0) * (args.N / 3.0))),
        "k_star": int(k_star),
        "max_cfl_allowed": float(args.max_cfl),
        "div_tol": float(args.div_tol),
        "curl_tol": float(args.curl_tol),
        "validation": {
            "max_cfl": float(np.max(cfl_values)),
            "max_div_rms": float(np.max(div_rms_values)),
            "max_div_linf": float(np.max(div_linf_values)),
            "max_curl_rel_l2": float(np.max(curl_rel_values)),
            "min_nonzero_shells_at_or_above_k_star": int(np.min(shell_counts)),
        },
    }

    payload: Dict[str, object] = {
        "omega_snapshots": np.stack(omega_snapshots, axis=0),
        "steps": np.asarray(snapshot_steps, dtype=np.int64),
        "energy": np.asarray(energies, dtype=np.float64),
        "enstrophy": np.asarray(enstrophies, dtype=np.float64),
        "max_abs_omega": np.asarray(max_abs_omega, dtype=np.float64),
        "cfl": np.asarray(cfl_values, dtype=np.float64),
        "div_rms": np.asarray(div_rms_values, dtype=np.float64),
        "div_linf": np.asarray(div_linf_values, dtype=np.float64),
        "curl_rel_l2": np.asarray(curl_rel_values, dtype=np.float64),
        "shell_nonzero_count_at_or_above_k_star": np.asarray(shell_counts, dtype=np.int64),
        "shell_energy": shell_energy,
        "meta_json": json.dumps(meta),
    }
    if not args.omit_velocity:
        payload["velocity_snapshots"] = np.stack(velocity_snapshots, axis=0)

    np.savez_compressed(out_path, **payload)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[truth3d] wrote {out_path}")
    print(f"[truth3d] wrote {meta_path}")

    if args.update_manifest:
        manifest_path = out_path.parent / "truth3d_manifest.json"
        manifest: Dict[str, object] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        manifest["latest"] = {
            "npz": str(out_path),
            "meta": str(meta_path),
            "backend": args.backend,
            "N": int(args.N),
            "seed": int(args.seed),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[truth3d] updated {manifest_path}")


if __name__ == "__main__":
    main()
