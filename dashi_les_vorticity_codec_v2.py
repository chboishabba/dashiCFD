#!/usr/bin/env python3
"""
dashi_les_vorticity_codec_v2.py

DASHI–LES 2D vorticity simulator + *structural codec* benchmark with visuals and MDL-style storage accounting.

New vs v1:
- Visual outputs (initial/final ω, baseline vs DASHI-gated, reconstruction + error, s* + mask)
- Faster kernel saturation (vectorized 3x3 majority via np.roll; no Python loops)
- Structural update cadence (compute DASHI mask every K steps, reuse in between)
- Better storage accounting:
  * Support-as-rule when mask ~full domain (“FULL” flag)
  * Otherwise RLE estimate for mask and ternary s*
  * Residual bits estimated via Shannon entropy of quantized residuals (proxy for entropy coding)
- Quality slider: sweep quantization step q and report rate–distortion (bits vs error)

Dependencies: numpy, matplotlib. (No SciPy.)
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from time import perf_counter

# -----------------------------
# Spectral utilities (periodic box)
# -----------------------------

def fft2(a): return np.fft.fft2(a)
def ifft2(a): return np.fft.ifft2(a).real

def make_grid(N: int, L: float = 2*np.pi):
    dx = L / N
    kx = np.fft.fftfreq(N, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2*np.pi
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    return dx, KX, KY, K2

def deriv_x(a, KX): return ifft2(1j * KX * fft2(a))
def deriv_y(a, KY): return ifft2(1j * KY * fft2(a))
def laplacian(a, K2): return ifft2(-K2 * fft2(a))

def poisson_solve_minus_lap(omega, K2):
    # ∇²ψ = -ω  => ψ̂ = ω̂ / K², with ψ̂[0,0]=0
    oh = fft2(omega)
    psih = oh / K2
    psih[0, 0] = 0.0
    return ifft2(psih)

def velocity_from_psi(psi, KX, KY):
    u = deriv_y(psi, KY)      # u = ∂ψ/∂y
    v = -deriv_x(psi, KX)     # v = -∂ψ/∂x
    return u, v

# -----------------------------
# Cheap smoothers (box filter)
# -----------------------------

def smooth2d(a, k=9):
    """Separable box filter, same-sized output. Periodic effects ignored at edges (ok for demo)."""
    if k <= 1:
        return a.copy()
    w = np.ones(k, dtype=float) / k
    tmp = np.apply_along_axis(lambda r: np.convolve(r, w, mode="same"), 1, a)
    return np.apply_along_axis(lambda c: np.convolve(c, w, mode="same"), 0, tmp)

# -----------------------------
# DASHI ternary structural layer
# -----------------------------

def ternary_sym(X, tau):
    s = np.zeros_like(X, dtype=np.int8)
    s[X >=  tau] = +1
    s[X <= -tau] = -1
    return s

def majority_kernel_3x3(s: np.ndarray) -> np.ndarray:
    """
    Vectorized 3x3 signed majority.
    Uses periodic wrap (np.roll), matching our periodic flow box.
    """
    acc = (
        s +
        np.roll(s,  1, axis=0) + np.roll(s, -1, axis=0) +
        np.roll(s,  1, axis=1) + np.roll(s, -1, axis=1) +
        np.roll(np.roll(s,  1, axis=0),  1, axis=1) +
        np.roll(np.roll(s,  1, axis=0), -1, axis=1) +
        np.roll(np.roll(s, -1, axis=0),  1, axis=1) +
        np.roll(np.roll(s, -1, axis=0), -1, axis=1)
    ).astype(np.int16)
    out = np.zeros_like(s, dtype=np.int8)
    out[acc > 0] = 1
    out[acc < 0] = -1
    return out

def saturate_ternary(s0: np.ndarray, iters: int = 8) -> np.ndarray:
    s = s0
    for _ in range(iters):
        sn = majority_kernel_3x3(s)
        if np.array_equal(sn, s):
            break
        s = sn
    return s

def dilate_mask(m: np.ndarray, r: int = 1) -> np.ndarray:
    b = m.astype(bool)
    for _ in range(r):
        b = (b |
             np.roll(b, 1, axis=0) | np.roll(b, -1, axis=0) |
             np.roll(b, 1, axis=1) | np.roll(b, -1, axis=1))
    return b.astype(np.uint8)

# -----------------------------
# LES components
# -----------------------------

def strain_mag(u, v, KX, KY):
    du_dx = deriv_x(u, KX)
    du_dy = deriv_y(u, KY)
    dv_dx = deriv_x(v, KX)
    dv_dy = deriv_y(v, KY)
    Sxy = 0.5*(du_dy + dv_dx)
    return np.sqrt(2*(du_dx**2 + dv_dy**2 + 2*Sxy**2) + 1e-30)

def smagorinsky_nu(u, v, KX, KY, Cs, Delta):
    return (Cs * Delta)**2 * strain_mag(u, v, KX, KY)

def rhs_vorticity(omega, nu, KX, KY, K2):
    psi = poisson_solve_minus_lap(omega, K2)
    u, v = velocity_from_psi(psi, KX, KY)
    adv = u * deriv_x(omega, KX) + v * deriv_y(omega, KY)
    diff = nu * laplacian(omega, K2)
    return -adv + diff

def step_rk2(omega, nu, dt, KX, KY, K2):
    k1 = rhs_vorticity(omega, nu, KX, KY, K2)
    k2 = rhs_vorticity(omega + dt*k1, nu, KX, KY, K2)
    return omega + 0.5*dt*(k1 + k2)

# -----------------------------
# Storage accounting helpers (MDL-ish)
# -----------------------------

def shannon_bits(symbols: np.ndarray) -> float:
    """Empirical Shannon bits for integer symbols (entropy coding proxy)."""
    if symbols.size == 0:
        return 0.0
    _, counts = np.unique(symbols, return_counts=True)
    p = counts.astype(float) / symbols.size
    H = -np.sum(p * np.log2(p + 1e-30))
    return float(symbols.size * H)

def rle_bits_binary(mask: np.ndarray) -> float:
    """Row-wise run-length estimate for binary mask. Lengths cost ~log2(runlen)."""
    H, W = mask.shape
    bits = 0.0
    for i in range(H):
        row = mask[i, :].astype(np.uint8)
        cur = int(row[0])
        run = 1
        bits += 1.0  # first value bit
        for j in range(1, W):
            v = int(row[j])
            if v == cur:
                run += 1
            else:
                bits += math.ceil(math.log2(run + 1))
                cur = v
                run = 1
        bits += math.ceil(math.log2(run + 1))
    return bits

def rle_bits_ternary(s: np.ndarray) -> float:
    """Row-wise run-length estimate for ternary in {-1,0,+1}. Values cost 2 bits, lengths ~log2(runlen)."""
    H, W = s.shape
    bits = 0.0
    for i in range(H):
        row = s[i, :].astype(np.int8)
        cur = int(row[0])
        run = 1
        bits += 2.0  # first value (2 bits to cover 3 states)
        for j in range(1, W):
            v = int(row[j])
            if v == cur:
                run += 1
            else:
                bits += math.ceil(math.log2(run + 1))
                bits += 2.0
                cur = v
                run = 1
        bits += math.ceil(math.log2(run + 1))
    return bits

# -----------------------------
# DASHI structural codec (field -> payload bits -> reconstruction)
# -----------------------------

@dataclass
class CodecStats:
    raw_bits: float
    total_bits: float
    compression_ratio: float
    rel_l2: float
    correlation: float
    support_frac: float
    bits_mask: float
    bits_s: float
    bits_residual: float

def dashi_codec(
    omega: np.ndarray,
    tau: float,
    smooth_k: int,
    q: float,
    band_r: int = 1,
    support_full_thresh: float = 0.98,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CodecStats]:
    """
    Encode omega with DASHI structural codec:
      - signed anomaly -> ternary s* -> support mask m
      - residual quantization on a band B(m)
      - storage: mask as rule if ~full, else RLE; s* as RLE; residual via Shannon bits proxy
    Returns: omega_rec, s*, m, stats
    """
    H, W = omega.shape
    raw_bits = float(H * W * 32)  # float32 baseline

    # structural extraction
    base = smooth2d(omega, smooth_k)
    X = omega - base
    X = np.clip(X / (np.max(np.abs(X)) + 1e-12), -1, 1)
    s0 = ternary_sym(X, tau)
    s = saturate_ternary(s0, iters=8)
    m = (s != 0).astype(np.uint8)
    support_frac = float(np.mean(m))

    # structural predictor: keep omega on support and smooth
    omega_keep = np.zeros_like(omega)
    omega_keep[m.astype(bool)] = omega[m.astype(bool)]
    omega_hat = smooth2d(omega_keep, smooth_k)

    # residual on band
    band = dilate_mask(m, r=band_r).astype(bool)
    resid = omega - omega_hat
    resid_q = np.zeros_like(omega, dtype=np.int16)
    resid_q[band] = np.round(resid[band] / q).astype(np.int16)
    omega_rec = omega_hat + resid_q.astype(float) * q

    # storage accounting
    if support_frac >= support_full_thresh:
        bits_mask = 1.0  # "FULL"
    else:
        bits_mask = rle_bits_binary(m)

    bits_s = rle_bits_ternary(s)
    bits_residual = shannon_bits(resid_q[band])

    bits_params = 96.0  # tau, smooth_k, q, band_r, etc (rough constant)
    total_bits = float(bits_mask + bits_s + bits_residual + bits_params)

    # accuracy
    rel_l2 = float(np.linalg.norm(omega - omega_rec) / (np.linalg.norm(omega) + 1e-12))
    corr = float(np.corrcoef(omega.flatten(), omega_rec.flatten())[0, 1])

    stats = CodecStats(
        raw_bits=raw_bits,
        total_bits=total_bits,
        compression_ratio=float(raw_bits / total_bits),
        rel_l2=rel_l2,
        correlation=corr,
        support_frac=support_frac,
        bits_mask=float(bits_mask),
        bits_s=float(bits_s),
        bits_residual=float(bits_residual),
    )
    return omega_rec, s, m, stats

# -----------------------------
# Main experiment
# -----------------------------

def simulate(
    N: int = 64,
    steps: int = 250,
    dt: float = 0.01,
    nu0: float = 1e-4,
    Cs: float = 0.17,
    dashi_tau: float = 0.35,
    dashi_smooth_k: int = 11,
    alpha: float = 0.7,
    dashi_update_every: int = 8,
):
    """Run baseline LES + DASHI-gated LES. Returns snapshots and enstrophy curves."""
    dx, KX, KY, K2 = make_grid(N)

    omega0 = smooth2d(np.random.randn(N, N), 11)
    omega0 = (omega0 - omega0.mean()) / (omega0.std() + 1e-12)

    omega_base = omega0.copy()
    omega_gated = omega0.copy()

    Z_base = []
    Z_gated = []

    # cached DASHI mask for gated run
    m_cached = np.ones((N, N), dtype=np.float64)
    s_cached = np.zeros((N, N), dtype=np.int8)

    t0 = perf_counter()
    for t in range(steps):
        # baseline step
        psi = poisson_solve_minus_lap(omega_base, K2)
        u, v = velocity_from_psi(psi, KX, KY)
        nu_t = smagorinsky_nu(u, v, KX, KY, Cs, dx)
        omega_base = step_rk2(omega_base, nu0 + np.maximum(0.0, nu_t), dt, KX, KY, K2)

        # gated step
        psi_g = poisson_solve_minus_lap(omega_gated, K2)
        ug, vg = velocity_from_psi(psi_g, KX, KY)
        nu_tg = smagorinsky_nu(ug, vg, KX, KY, Cs, dx)

        if (t % dashi_update_every) == 0:
            base_g = smooth2d(omega_gated, dashi_smooth_k)
            X = omega_gated - base_g
            X = np.clip(X / (np.max(np.abs(X)) + 1e-12), -1, 1)
            s0 = ternary_sym(X, dashi_tau)
            s_cached = saturate_ternary(s0, iters=8)
            m_cached = (s_cached != 0).astype(np.float64)

        nu_eff = nu0 + nu_tg * (1.0 - alpha * m_cached)
        omega_gated = step_rk2(omega_gated, nu_eff, dt, KX, KY, K2)

        Z_base.append(0.5*np.mean(omega_base**2))
        Z_gated.append(0.5*np.mean(omega_gated**2))

    sim_seconds = perf_counter() - t0
    return omega0, omega_base, omega_gated, np.array(Z_base), np.array(Z_gated), s_cached, m_cached, sim_seconds

def main():
    np.random.seed(0)

    # ---- simulation ----
    omega0, omega_base, omega_gated, Zb, Zg, s_final, m_final, sim_s = simulate(
        N=64, steps=250, dt=0.01, dashi_update_every=8
    )
    print(f"[sim] seconds={sim_s:.3f}  support_frac_final={float(np.mean(m_final)):.4f}")

    # ---- codec sweeps (quality slider) ----
    qs = [0.02, 0.05, 0.1, 0.2]
    codec_rows = []
    omega_rec_ref = None

    t1 = perf_counter()
    for q in qs:
        omega_rec, s, m, stats = dashi_codec(
            omega_gated, tau=0.35, smooth_k=11, q=q, band_r=1, support_full_thresh=0.98
        )
        codec_rows.append((q, stats))
        if omega_rec_ref is None:
            omega_rec_ref = omega_rec
    codec_s = perf_counter() - t1

    print(f"[codec] seconds={codec_s:.3f}")
    for q, st in codec_rows:
        print(
            f"q={q:<5}  ratio={st.compression_ratio:8.3f}  relL2={st.rel_l2:8.4f}  corr={st.correlation:8.6f}  "
            f"bits(mask/s/res)={st.bits_mask:.1f}/{st.bits_s:.1f}/{st.bits_residual:.1f}  support={st.support_frac:.3f}"
        )

    # ---- plots ----
    plt.figure(figsize=(8,4))
    plt.plot(Zb, label="baseline LES (Smagorinsky)")
    plt.plot(Zg, label="DASHI-gated LES")
    plt.xlabel("timestep")
    plt.ylabel("enstrophy 0.5⟨ω²⟩")
    plt.title("2D decaying vorticity: enstrophy decay")
    plt.legend()
    plt.tight_layout()
    plt.show()

    for title, field, vmin, vmax in [
        ("Initial vorticity ω0", omega0, None, None),
        ("Final ω (baseline LES)", omega_base, None, None),
        ("Final ω (DASHI-gated LES)", omega_gated, None, None),
        ("Final s* (ternary structure on ω)", s_final, -1, 1),
        ("Final support mask m", m_final, 0, 1),
        (f"Reconstructed ω̂ (q={qs[0]})", omega_rec_ref, None, None),
        (f"Error ω - ω̂ (q={qs[0]})", omega_gated - omega_rec_ref, None, None),
    ]:
        plt.figure(figsize=(5,5))
        if vmin is None:
            im = plt.imshow(field, origin="lower")
        else:
            im = plt.imshow(field, origin="lower", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    bits = [st.total_bits for _, st in codec_rows]
    rel = [st.rel_l2 for _, st in codec_rows]
    plt.figure(figsize=(6,4))
    plt.plot(bits, rel, marker="o")
    plt.xlabel("estimated bits (total)")
    plt.ylabel("relative L2 error")
    plt.title("DASHI codec rate–distortion (estimated)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
