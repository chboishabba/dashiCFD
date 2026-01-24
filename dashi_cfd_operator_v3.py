#!/usr/bin/env python3
"""
dashi_cfd_operator_v3.py

Minimal "learned operator on a DASHI proxy" benchmark for 2D incompressible vorticity (periodic box).

Pipeline
1) Generate a baseline LES trajectory (Smagorinsky).
2) Build a *proxy* (compressed state) z_t from the field ω_t:
   - Low-frequency spectral coefficients (circular cutoff |k|<=k_cut)  [main compressive carrier]
   - A few DASHI structural scalars (support fraction from ternary residual structure) [metadata]
   - Residual energy band scalars (mid/high)                            [metadata]
3) Learn a linear update operator A such that z_{t+1} ≈ z_t @ A (ridge regression).
4) Roll out the learned operator in proxy space and decode to ω̂_t via inverse FFT of predicted low-k coefficients.
5) Report:
   - time per LES step vs time per proxy update (plus decode cost)
   - reconstruction error over time
   - energy/enstrophy drift
   - snapshots + curves

Dependencies: numpy, matplotlib (no SciPy).
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
    kx = np.fft.fftfreq(N, d=dx) * 2*np.pi  # for L=2π these are integer wave numbers
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

def energy_from_omega(omega, KX, KY, K2):
    psi = poisson_solve_minus_lap(omega, K2)
    u, v = velocity_from_psi(psi, KX, KY)
    return 0.5 * float(np.mean(u*u + v*v))

def enstrophy(omega):
    return 0.5 * float(np.mean(omega*omega))


# -----------------------------
# Simple smoother (for DASHI residual extraction)
# -----------------------------

def smooth2d(a, k=9):
    if k <= 1:
        return a.copy()
    w = np.ones(k, dtype=float) / k
    tmp = np.apply_along_axis(lambda r: np.convolve(r, w, mode="same"), 1, a)
    return np.apply_along_axis(lambda c: np.convolve(c, w, mode="same"), 0, tmp)


# -----------------------------
# DASHI ternary structure (fast 3x3 majority via rolls)
# -----------------------------

def ternary_sym(X, tau):
    s = np.zeros_like(X, dtype=np.int8)
    s[X >=  tau] = +1
    s[X <= -tau] = -1
    return s

def majority_kernel_3x3(s: np.ndarray) -> np.ndarray:
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


# -----------------------------
# LES (Smagorinsky) — baseline generator
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

def simulate_les_trajectory(
    N: int = 64,
    steps: int = 300,
    dt: float = 0.01,
    nu0: float = 1e-4,
    Cs: float = 0.17,
    seed: int = 0,
):
    """Return ω trajectory [steps+1, N, N] using baseline Smagorinsky LES."""
    np.random.seed(seed)
    dx, KX, KY, K2 = make_grid(N)

    omega0 = smooth2d(np.random.randn(N, N), 11)
    omega0 = (omega0 - omega0.mean()) / (omega0.std() + 1e-12)

    traj = np.zeros((steps+1, N, N), dtype=np.float64)
    traj[0] = omega0

    t0 = perf_counter()
    omega = omega0.copy()
    for t in range(steps):
        psi = poisson_solve_minus_lap(omega, K2)
        u, v = velocity_from_psi(psi, KX, KY)
        nu_t = np.maximum(0.0, smagorinsky_nu(u, v, KX, KY, Cs, dx))
        omega = step_rk2(omega, nu0 + nu_t, dt, KX, KY, K2)
        traj[t+1] = omega
    sim_s = perf_counter() - t0
    return traj, (dx, KX, KY, K2), sim_s


# -----------------------------
# Proxy encoding (DASHI + spectral chart)
# -----------------------------

@dataclass
class ProxyConfig:
    k_cut: float = 8.0          # circular cutoff in |k|
    dashi_tau: float = 0.35
    dashi_smooth_k: int = 11
    resid_mid_cut: float = 12.0 # split residual energy into mid/high bands

def circular_kmask(KX, KY, k_cut: float):
    kmag = np.sqrt(KX*KX + KY*KY)
    return (kmag <= k_cut)

def encode_proxy(omega: np.ndarray, grid, cfg: ProxyConfig):
    """
    Return feature vector z and the low-k mask.
    z layout:
      [lowk_real..., lowk_imag..., support_frac, resid_mid_E, resid_high_E]
    """
    dx, KX, KY, K2 = grid
    N = omega.shape[0]

    # Low-k spectral chart
    mask = circular_kmask(KX, KY, cfg.k_cut)
    oh = fft2(omega)
    lowk = oh[mask]  # complex
    scale = float(N*N)  # normalize
    lowk_r = (lowk.real / scale).astype(np.float64)
    lowk_i = (lowk.imag / scale).astype(np.float64)

    # DASHI structural scalar: support fraction from signed residual
    base = smooth2d(omega, cfg.dashi_smooth_k)
    R = omega - base
    Rn = np.clip(R / (np.max(np.abs(R)) + 1e-12), -1, 1)
    s = saturate_ternary(ternary_sym(Rn, cfg.dashi_tau), iters=8)
    support_frac = float(np.mean(s != 0))

    # Residual energy bands (mid/high) in spectral domain
    kmag = np.sqrt(KX*KX + KY*KY)
    mid = (kmag > cfg.k_cut) & (kmag <= cfg.resid_mid_cut)
    high = (kmag > cfg.resid_mid_cut)
    Rhat = fft2(R)
    resid_mid_E = float(np.mean(np.abs(Rhat[mid])**2) / (scale*scale) if np.any(mid) else 0.0)
    resid_high_E = float(np.mean(np.abs(Rhat[high])**2) / (scale*scale) if np.any(high) else 0.0)

    z = np.concatenate([lowk_r, lowk_i, np.array([support_frac, resid_mid_E, resid_high_E], dtype=np.float64)])
    return z, mask

def decode_lowk(z: np.ndarray, mask: np.ndarray, N: int):
    """Decode ω̂ from low-k coefficients contained in z (ignores DASHI scalars)."""
    scale = float(N*N)
    M = int(np.sum(mask))
    lowk_r = z[:M]
    lowk_i = z[M:2*M]
    lowk = (lowk_r + 1j*lowk_i) * scale

    oh = np.zeros((N, N), dtype=np.complex128)
    oh[mask] = lowk
    return ifft2(oh)


# -----------------------------
# Learned operator: ridge regression for A
# -----------------------------

def learn_linear_operator(Z: np.ndarray, ridge: float = 1e-3):
    """
    Z: [T, D]. Learn A: z_{t+1} ≈ z_t @ A
    Returns A [D, D].
    """
    X = Z[:-1]  # [T-1, D]
    Y = Z[1:]   # [T-1, D]
    D = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    A = np.linalg.solve(XtX + ridge*np.eye(D), XtY)
    return A


# -----------------------------
# Main benchmark
# -----------------------------

def main():
    # ---- Config ----
    N = 64
    steps = 300
    dt = 0.01
    nu0 = 1e-4
    Cs = 0.17
    seed = 0

    # Proxy settings
    cfg = ProxyConfig(k_cut=8.0, resid_mid_cut=12.0, dashi_tau=0.35, dashi_smooth_k=11)
    ridge = 1e-3
    decode_every = 1  # set >1 to simulate "decode only sometimes"
    snap_t = [0, steps//3, 2*steps//3, steps]

    # ---- Generate baseline trajectory ----
    traj, grid, sim_s = simulate_les_trajectory(N=N, steps=steps, dt=dt, nu0=nu0, Cs=Cs, seed=seed)
    dx, KX, KY, K2 = grid
    print(f"[baseline LES] steps={steps}  seconds={sim_s:.3f}  per_step_ms={1000*sim_s/steps:.3f}")

    # ---- Encode proxy sequence ----
    t0 = perf_counter()
    Z = []
    mask0 = None
    for t in range(steps+1):
        z, mask = encode_proxy(traj[t], grid, cfg)
        if mask0 is None:
            mask0 = mask
        Z.append(z)
    Z = np.stack(Z, axis=0)
    enc_s = perf_counter() - t0
    D = Z.shape[1]
    lowk_modes = int(np.sum(mask0))
    print(f"[encode] T={steps+1} D={D} seconds={enc_s:.3f} per_frame_ms={1000*enc_s/(steps+1):.3f}  k_cut={cfg.k_cut} lowk_modes={lowk_modes}")

    # ---- Learn operator ----
    t1 = perf_counter()
    A = learn_linear_operator(Z, ridge=ridge)
    learn_s = perf_counter() - t1
    print(f"[learn] ridge={ridge} seconds={learn_s:.3f}")

    # ---- Rollout operator ----
    Zhat = np.zeros_like(Z)
    Zhat[0] = Z[0]
    t2 = perf_counter()
    for t in range(steps):
        Zhat[t+1] = Zhat[t] @ A
    roll_s = perf_counter() - t2
    print(f"[rollout] seconds={roll_s:.3f} per_step_us={1e6*roll_s/steps:.2f}  (proxy update only)")

    # ---- Decode + evaluate ----
    t3 = perf_counter()
    rel_l2 = np.zeros(steps+1, dtype=np.float64)
    corr = np.zeros(steps+1, dtype=np.float64)
    Eb = np.zeros(steps+1, dtype=np.float64)
    Eh = np.zeros(steps+1, dtype=np.float64)
    Zb = np.zeros(steps+1, dtype=np.float64)
    Zh = np.zeros(steps+1, dtype=np.float64)

    omega_hat_snap = {}

    prev_hat = None
    for t in range(steps+1):
        omega_true = traj[t]
        Eb[t] = energy_from_omega(omega_true, KX, KY, K2)
        Zb[t] = enstrophy(omega_true)

        if (t % decode_every) == 0 or prev_hat is None:
            omega_hat = decode_lowk(Zhat[t], mask0, N=N)
            prev_hat = omega_hat
        else:
            omega_hat = prev_hat

        Eh[t] = energy_from_omega(omega_hat, KX, KY, K2)
        Zh[t] = enstrophy(omega_hat)

        num = np.linalg.norm(omega_true - omega_hat)
        den = np.linalg.norm(omega_true) + 1e-12
        rel_l2[t] = float(num / den)
        corr[t] = float(np.corrcoef(omega_true.flatten(), omega_hat.flatten())[0, 1])

        if t in snap_t:
            omega_hat_snap[t] = omega_hat.copy()

    dec_eval_s = perf_counter() - t3
    print(f"[decode+eval] seconds={dec_eval_s:.3f} per_frame_ms={1000*dec_eval_s/(steps+1):.3f} (includes invariants)")
    print(f"[final] relL2={rel_l2[-1]:.4f}  corr={corr[-1]:.6f}  ΔE={Eh[-1]-Eb[-1]:+.3e}  ΔZ={Zh[-1]-Zb[-1]:+.3e}")

    # ---- Plots ----
    plt.figure(figsize=(8,4))
    plt.plot(rel_l2)
    plt.xlabel("timestep")
    plt.ylabel("relative L2 error")
    plt.title("Learned operator rollout error (decode from low-k chart)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(corr)
    plt.xlabel("timestep")
    plt.ylabel("correlation")
    plt.title("Learned operator rollout correlation")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(Eb, label="baseline E")
    plt.plot(Eh, label="learned/decode E")
    plt.xlabel("timestep")
    plt.ylabel("energy")
    plt.title("Energy drift")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(Zb, label="baseline enstrophy")
    plt.plot(Zh, label="learned/decode enstrophy")
    plt.xlabel("timestep")
    plt.ylabel("0.5⟨ω²⟩")
    plt.title("Enstrophy drift")
    plt.legend()
    plt.tight_layout()
    plt.show()

    for t in snap_t:
        omega_true = traj[t]
        omega_hat = omega_hat_snap[t]
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(omega_true, origin="lower")
        plt.title(f"ω true (t={t})")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(omega_hat, origin="lower")
        plt.title(f"ω̂ decoded (t={t})")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(omega_true - omega_hat, origin="lower")
        plt.title("error ω-ω̂")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    # ---- Speed ledger ----
    baseline_ms = 1000*sim_s/steps
    proxy_us = 1e6*roll_s/steps
    print("\n=== Speed ledger (this run) ===")
    print(f"Baseline LES:      {baseline_ms:.3f} ms/step")
    print(f"Proxy update:      {proxy_us:.2f} µs/step  (z_{t+1}=z_t @ A)")
    print(f"Decode+eval:       {1000*dec_eval_s/(steps+1):.3f} ms/frame  (includes invariants + FFT decode)")
    print("Note: decode is O(N^2 log N) due to FFT; proxy update is O(D^2) here (small D).")
    print("For 'fast mode', set decode_every>1 or decode only on demand.")

if __name__ == "__main__":
    main()
