import numpy as np

from free_surface_shallow_water import Grid, RunConfig, make_hull_mask, run


def test_hull_mask_is_nonempty_and_bounded():
    grid = Grid(nx=64, ny=32)
    solid = make_hull_mask(grid)
    assert solid.any()
    assert (~solid).any()
    assert not solid[0].any()
    assert not solid[-1].any()


def test_short_free_surface_run_is_finite_and_bounded():
    grid = Grid(nx=64, ny=32)
    cfg = RunConfig(seconds=0.5, cfl=0.25)
    U, solid, receipt = run(grid, cfg)
    assert np.isfinite(U).all()
    assert np.all(U[:, solid] == 0.0)
    assert np.min(U[0, ~solid]) > 0.0
    assert abs(float(receipt["mass_relative_error"])) < 0.01
    assert float(receipt["max_abs_eta_m"]) > 0.0
    assert receipt["fidelity"] == "reduced_free_surface_shallow_water_not_3d_vof_rans_les"
    assert receipt["physically_calibrated_hull"] is False


def test_receipt_is_deterministic():
    grid = Grid(nx=48, ny=24)
    cfg = RunConfig(seconds=0.25)
    _, _, a = run(grid, cfg)
    _, _, b = run(grid, cfg)
    assert a["artifact_sha256"] == b["artifact_sha256"]
