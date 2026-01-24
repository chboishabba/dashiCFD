# Repository Guidelines

Use this guide to modify or extend the CFD + DASHI experiments without breaking reproducibility.

## Project Structure & Module Organization
- Core rollouts: `dashi_cfd_operator_v3.py` (baseline LES proxy) and `dashi_cfd_operator_v4.py` (residual-closure proxy with codec + plots).
- DASHI codecs: `dashi_les_vorticity_codec.py` and `dashi_les_vorticity_codec_v2.py` hold ternary masks, quantization, and rate–distortion helpers.
- Other experiments: `naw.py` / `naw2.py` (gauge-coupling scan), `vortex_tester_mdl.py` (minimal vorticity sandbox).
- Assets: PNG figures and `dashi_signed_branchedflow_codec.npz` (codec weights). Keep new plots in an `outputs/` subfolder or follow the existing `output - YYYY-MM-DDTHHMMSS.png` pattern to avoid root clutter.

## Build, Test, and Development Commands
- Install deps: `python3 -m pip install -U numpy matplotlib`.
- Run residual-closure experiment: `python3 dashi_cfd_operator_v4.py` (prints speed ledger; opens several matplotlib windows).
- Baseline proxy: `python3 dashi_cfd_operator_v3.py`.
- Codec evaluation: `python3 dashi_les_vorticity_codec_v2.py` (rate–distortion plots).
- Headless/CI runs: prefix commands with `MPLBACKEND=Agg` to save plots without a GUI.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, snake_case for functions/variables; constants ALL_CAPS.
- Favor pure/numpy-vectorized helpers; keep spectral grid utilities small and reusable.
- Use type hints and docstrings for new APIs; prefer f-strings for logging.
- Place runnable code under `if __name__ == "__main__":` so modules stay importable.

## Testing Guidelines
- No formal test suite yet; when adding tests use `pytest` with deterministic RNG seeds (`np.random.default_rng(seed)`).
- Golden-metric idea: assert relative L2 error, correlation, and energy/enstrophy drift stay within expected tolerances for short rollouts.
- Save large fixtures as compressed `.npz`; avoid checking in files >10 MB.

## Commit & Pull Request Guidelines
- Repo is currently unversioned; adopt Conventional Commit prefixes (`feat:`, `fix:`, `perf:`, `docs:`) to ease future history.
- For changes, include: brief experiment description, command used, runtime params (N, steps, dt, seeds), and representative plots or metrics.
- If you change numerical defaults, note why and the expected effect on stability/speed.

## Security & Configuration Tips
- Do not commit credentials; only simulation-derived data belongs here.
- Heavy runs can be CPU-intensive; set `OMP_NUM_THREADS` to match your machine and start with smaller grids (`N=64`) for quick checks.
- Save matplotlib figures with explicit DPI to keep file sizes manageable.
