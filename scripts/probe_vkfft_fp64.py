#!/usr/bin/env python3
"""Probe Vulkan/vkFFT complex128 support on the local device."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "dashiCORE"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from gpu_vkfft_adapter import VkFFTExecutor  # type: ignore
from gpu_vulkan_dispatcher import VulkanDispatchConfig, create_vulkan_handles  # type: ignore

try:
    import vulkan as vk  # type: ignore
except Exception as exc:  # pragma: no cover
    vk = None  # type: ignore
    VK_IMPORT_ERROR = exc
else:
    VK_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=16, help="cubic FFT size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--json", type=Path, default=None, help="optional JSON report path")
    return parser.parse_args()


def device_report(handles: Any) -> dict[str, Any]:
    props = vk.vkGetPhysicalDeviceProperties(handles.physical_device)
    feats = vk.vkGetPhysicalDeviceFeatures(handles.physical_device)
    return {
        "device_name": str(getattr(props, "deviceName", "")),
        "vendor_id": int(getattr(props, "vendorID", 0)),
        "device_id": int(getattr(props, "deviceID", 0)),
        "api_version": int(getattr(props, "apiVersion", 0)),
        "driver_version": int(getattr(props, "driverVersion", 0)),
        "shaderFloat64": bool(getattr(feats, "shaderFloat64", False)),
        "shaderInt64": bool(getattr(feats, "shaderInt64", False)),
    }


def main() -> None:
    args = parse_args()
    if vk is None:
        raise SystemExit(f"vulkan python package not available: {VK_IMPORT_ERROR}")
    n = int(args.N)
    rng = np.random.default_rng(int(args.seed))
    x = (rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))).astype(np.complex128)
    handles = create_vulkan_handles(VulkanDispatchConfig(enable_shader_float64=True))
    executor = VkFFTExecutor(handles=handles, fft_backend="vkfft-vulkan", timing_enabled=True)
    report: dict[str, Any] = {
        "N": n,
        "dtype": "complex128",
        "VK_ICD_FILENAMES": os.environ.get("VK_ICD_FILENAMES"),
        "device": device_report(handles),
    }
    try:
        y = executor.ifftn(executor.fftn(x))
        abs_err = np.abs(y - x)
        rel = abs_err / (np.abs(x) + 1e-300)
        report.update(
            {
                "backend_used": executor.get_last_timings(),
                "max_abs_error": float(abs_err.max()),
                "rms_abs_error": float(np.sqrt(np.mean(abs_err * abs_err))),
                "max_relative_error": float(rel.max()),
                "allclose": bool(np.allclose(y, x, rtol=float(args.rtol), atol=float(args.atol))),
                "rtol": float(args.rtol),
                "atol": float(args.atol),
            }
        )
    finally:
        executor.close()
        handles.close()
    text = json.dumps(report, indent=2, allow_nan=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    if not report.get("allclose", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
