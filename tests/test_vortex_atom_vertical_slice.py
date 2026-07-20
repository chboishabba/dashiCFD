import math

import numpy as np

from dashi_vortex_atoms import (
    AtomCodecConfig,
    VortexAtom,
    decode_atoms,
    decode_proxy,
    defect_metrics,
    extract_vortex_atoms,
    proxy_step,
    spectral_velocity,
    transport_atoms,
)


def _periodic_vortex(n: int = 32, y0: float = 10.0, x0: float = 12.0, sign: float = 1.0) -> np.ndarray:
    yy, xx = np.indices((n, n), dtype=np.float64)
    dy = ((yy - y0 + n / 2.0) % n) - n / 2.0
    dx = ((xx - x0 + n / 2.0) % n) - n / 2.0
    field = sign * np.exp(-(dx * dx + dy * dy) / (2.0 * 2.5**2))
    return field - field.mean()


def test_extract_decode_preserves_signed_structure() -> None:
    omega = _periodic_vortex() - 0.7 * _periodic_vortex(y0=22.0, x0=20.0, sign=1.0)
    config = AtomCodecConfig(
        smooth_k=9,
        threshold_sigma=0.25,
        max_atoms=8,
        peak_candidates=16,
        bits_per_atom=0.0,
        q=0.1,
    )
    state = extract_vortex_atoms(omega, config)
    decoded = decode_proxy(state)
    metrics = defect_metrics(omega, decoded)

    assert state.atoms
    assert any(atom.sign > 0 for atom in state.atoms)
    assert any(atom.sign < 0 for atom in state.atoms)
    assert metrics.rel_l2 < 1.0
    assert metrics.correlation > 0.0


def test_atom_decode_is_deterministic() -> None:
    atom = VortexAtom(
        atom_id=1,
        parent_id=None,
        y=3.5,
        x=7.25,
        sign=-1,
        amplitude=-2.0,
        circulation=0.0,
        core_scale=2.0,
        orientation=math.pi / 4.0,
        anisotropy=3.0,
    )
    a = decode_atoms((16, 16), [atom])
    b = decode_atoms((16, 16), [atom])
    np.testing.assert_array_equal(a, b)


def test_transport_wraps_and_increments_lifetime() -> None:
    n = 16
    u = np.full((n, n), 2.0)
    v = np.full((n, n), 1.0)
    atom = VortexAtom(
        atom_id=5,
        parent_id=None,
        y=15.75,
        x=15.5,
        sign=1,
        amplitude=1.0,
        circulation=0.0,
        core_scale=2.0,
        orientation=0.0,
        anisotropy=1.0,
    )
    moved = transport_atoms([atom], u, v, 0.5)[0]
    assert 0.0 <= moved.y < n
    assert 0.0 <= moved.x < n
    assert math.isclose(moved.y, 0.25)
    assert math.isclose(moved.x, 0.5)
    assert moved.lifetime == 1
    assert moved.sign == atom.sign


def test_spectral_velocity_is_incompressible_to_roundoff() -> None:
    omega = _periodic_vortex()
    u, v = spectral_velocity(omega)
    du_dx = 0.5 * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1))
    dv_dy = 0.5 * (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0))
    assert float(np.linalg.norm(du_dx + dv_dy)) < 1e-8


def test_proxy_step_is_truth_free_and_records_ledger() -> None:
    omega = _periodic_vortex()
    config = AtomCodecConfig(
        smooth_k=9,
        threshold_sigma=0.25,
        max_atoms=4,
        peak_candidates=8,
        bits_per_atom=0.0,
        q=0.1,
    )
    state = extract_vortex_atoms(omega, config)
    stepped, ledger = proxy_step(state, 0.01, config, viscosity=1e-4)

    assert stepped.step == state.step + 1
    assert ledger.step == stepped.step
    assert ledger.transported == len(state.atoms)
    assert ledger.atom_count_after == len(stepped.atoms)
    assert ledger.total_mdl_bits >= 0.0
    assert np.isfinite(decode_proxy(stepped)).all()
