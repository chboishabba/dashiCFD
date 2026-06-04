import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_truth_3d import fft_vec, make_grid  # noqa: E402
from vulkan_truth3d_backend import VulkanTruth3DBackend  # noqa: E402


def test_gpu_shell_filter_matches_numpy_for_nonzero_shell() -> None:
    if "VK_ICD_FILENAMES" not in os.environ:
        pytest.skip("explicit Vulkan ICD not configured")
    n = 16
    rng = np.random.default_rng(3)
    u = rng.normal(scale=0.1, size=(n, n, n, 3)).astype(np.float32)
    u_hat = fft_vec(u)
    _dx, _kx, _ky, _kz, _k2, k_radius = make_grid(n, 2.0 * np.pi)
    shell_map = np.floor(k_radius + 1e-12).astype(np.int64)
    shell = 3
    mask = shell_map == shell
    expected = np.stack(
        [np.fft.ifftn(np.where(mask, u_hat[..., i], 0.0)).real for i in range(3)],
        axis=-1,
    )

    backend = VulkanTruth3DBackend(n, dt=0.001, nu0=0.001, length=2.0 * np.pi)
    try:
        backend.set_velocity_real(u)
        backend.set_shell_ids(shell_map)
        actual = backend.read_shell_vector("u", shell, "integer-radius")
    finally:
        backend.close()

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_gpu_fp64_shell_filter_matches_numpy_for_nonzero_shell() -> None:
    if "VK_ICD_FILENAMES" not in os.environ:
        pytest.skip("explicit Vulkan ICD not configured")
    n = 16
    rng = np.random.default_rng(4)
    u = rng.normal(scale=0.1, size=(n, n, n, 3)).astype(np.float64)
    u_hat = fft_vec(u)
    _dx, _kx, _ky, _kz, _k2, k_radius = make_grid(n, 2.0 * np.pi)
    shell_map = np.floor(k_radius + 1e-12).astype(np.int64)
    shell = 3
    mask = shell_map == shell
    expected = np.stack(
        [np.fft.ifftn(np.where(mask, u_hat[..., i], 0.0)).real for i in range(3)],
        axis=-1,
    )

    from vulkan_truth3d_backend import VulkanSpectralDiagnostic3DBackend

    backend = VulkanSpectralDiagnostic3DBackend(n, length=2.0 * np.pi, precision="float64")
    try:
        backend.set_velocity_real(u)
        backend.set_shell_ids(shell_map)
        actual = backend.read_shell_vector("u", shell, "integer-radius")
    finally:
        backend.close()

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
