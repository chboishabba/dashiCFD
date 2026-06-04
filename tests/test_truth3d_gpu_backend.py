from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "dashiCORE"


def test_vulkan_truth3d_backend_smoke() -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(CORE_ROOT) not in sys.path:
        sys.path.insert(0, str(CORE_ROOT))

    try:
        import vulkan  # noqa: F401
        import vkfft_vulkan_py  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Vulkan/vkFFT binding unavailable: {exc}")

    from scripts.make_truth_3d import component_dealias_mask, init_velocity_hat, make_grid
    from vulkan_truth3d_backend import VulkanTruth3DBackend

    class Args:
        N = 16
        L = 2.0 * np.pi
        seed = 0
        k_min = 1.0
        k_max = None
        target_urms = 1.0

    dx, kx, ky, kz, k2, _k_radius = make_grid(Args.N, Args.L)
    del dx
    mask = component_dealias_mask(kx, ky, kz, Args.N, Args.L)
    u_hat0 = init_velocity_hat(Args, kx, ky, kz, k2, mask)

    backend = VulkanTruth3DBackend(Args.N, dt=0.001, nu0=0.001, length=Args.L)
    try:
        backend.set_initial_u_hat(u_hat0)
        u_hat = backend.read_u_hat()
        assert u_hat.shape == (16, 16, 16, 3)
        assert np.isfinite(u_hat.view(np.float32)).all()
        backend.step()
        stepped = backend.read_u_hat()
        assert stepped.shape == (16, 16, 16, 3)
        assert np.isfinite(stepped.view(np.float32)).all()
    finally:
        backend.close()
