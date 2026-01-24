#!/usr/bin/env python3
"""
dashi_cfd_operator_v4.py  —  "Residual closure" version

This extends v3 by *using the predicted residual scalars* in z_t to synthesize
a mid/high-k residual field, injected back into ω̂ during decode.

Key idea:
- Proxy z_t contains: [low-k coeffs, support_frac, resid_mid_E, resid_high_E]
- We learn A on the full z, then roll out z purely in proxy space.
- During decode, we reconstruct:
      ω̂_t = ω_lowk_t  +  r̂_mid_t + r̂_high_t
  where r̂_mid/high are synthesized to match target band energies and
  are gated by a DASHI-derived support mask m(ω_lowk).

Dependencies: numpy, matplotlib.
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
    oh = fft2(omega)
    psih = oh / K2
    psih[0, 0] = 0.0
    return ifft2(psih)

def velocity_from_psi(psi, KX, KY):
    u = deriv_y(psi, KY)
    v = -deriv_x(psi, KX)
    return u, v

def energy_from_omega(omega, KX, KY, K2):
    psi = poisson_solve_minus_lap(omega, K2)
    u, v = velocity_from_psi(psi, KX, KY)
    return 0.5 * float(np.mean(u*u + v*v))

def enstrophy(omega):
    return 0.5 * float(np.mean(omega*omega))


# -----------------------------
# Simple smoother
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
# Proxy encoding
# -----------------------------

@dataclass
class ProxyConfig:
    k_cut: float = 8.0
    dashi_tau: float = 0.35
    dashi_smooth_k: int = 11
    resid_mid_cut: float = 12.0
    topk_mid: int = 128  # number of mid-band complex coeffs to preserve (phase-carrying)

def circular_kmask(KX, KY, k_cut: float):
    kmag = np.sqrt(KX*KX + KY*KY)
    return (kmag <= k_cut)

def encode_proxy(omega: np.ndarray, grid, cfg: ProxyConfig, anchor_idx=None):
    dx, KX, KY, K2 = grid
    N = omega.shape[0]

    mask_low = circular_kmask(KX, KY, cfg.k_cut)
    oh = fft2(omega)
    lowk = oh[mask_low]
    scale = float(N*N)
    lowk_r = (lowk.real / scale).astype(np.float64)
    lowk_i = (lowk.imag / scale).astype(np.float64)

    base = smooth2d(omega, cfg.dashi_smooth_k)
    R = omega - base
    Rn = np.clip(R / (np.max(np.abs(R)) + 1e-12), -1, 1)
    s = saturate_ternary(ternary_sym(Rn, cfg.dashi_tau), iters=8)
    support_frac = float(np.mean(s != 0))

    kmag = np.sqrt(KX*KX + KY*KY)
    mid = (kmag > cfg.k_cut) & (kmag <= cfg.resid_mid_cut)
    high = (kmag > cfg.resid_mid_cut)
    Rhat = fft2(R)

    # --- top-K mid-band preservation ---
    mid_flat_idx = np.flatnonzero(mid)
    if anchor_idx is None:
        topk = min(cfg.topk_mid, int(np.count_nonzero(mid)))
        if topk > 0:
            mid_vals = Rhat.flat[mid_flat_idx]
            mag = np.abs(mid_vals)
            topk_idx_local = np.argpartition(mag, -topk)[-topk:]
            anchor_idx = mid_flat_idx[topk_idx_local]
        else:
            anchor_idx = np.array([], dtype=np.int64)

    kept_vals = Rhat.flat[anchor_idx] if anchor_idx is not None and len(anchor_idx) > 0 else np.array([], dtype=np.complex128)
    topk = len(kept_vals)

    # Energies (mean per coefficient) with kept energy removed from mid band
    n_mid = int(np.count_nonzero(mid))
    n_high = int(np.count_nonzero(high))
    sum_mid = float(np.sum(np.abs(Rhat[mid])**2) / (scale*scale) if n_mid else 0.0)
    sum_mid_kept = float(np.sum(np.abs(kept_vals)**2) / (scale*scale) if topk else 0.0)
    rem_mid = n_mid - topk
    resid_mid_E = float((sum_mid - sum_mid_kept) / rem_mid) if rem_mid > 0 else 0.0
    resid_high_E = float(np.mean(np.abs(Rhat[high])**2) / (scale*scale) if n_high else 0.0)

    # Pack kept mid-band coeffs (indices as float to stay in one vector)
    kept_r = (kept_vals.real / scale).astype(np.float64)
    kept_i = (kept_vals.imag / scale).astype(np.float64)

    header = np.array([support_frac, resid_mid_E, resid_high_E, float(topk)], dtype=np.float64)
    z = np.concatenate([lowk_r, lowk_i, header, kept_r, kept_i])
    return z, mask_low, anchor_idx


def decode_with_residual(
    z: np.ndarray,
    grid,
    cfg: ProxyConfig,
    mask_low: np.ndarray,
    anchor_idx,
    rng: np.random.Generator,
):
    dx, KX, KY, K2 = grid
    N = KX.shape[0]
    scale = float(N*N)

    M = int(np.sum(mask_low))
    lowk_r = z[:M]
    lowk_i = z[M:2*M]
    target_mid_E = float(z[2*M + 1])
    target_high_E = float(z[2*M + 2])
    k_keep = int(round(z[2*M + 3]))
    k_keep = max(0, k_keep)

    # --- low-k reconstruction ---
    lowk = (lowk_r + 1j*lowk_i) * scale
    oh = np.zeros((N, N), dtype=np.complex128)
    oh[mask_low] = lowk
    omega_lp = ifft2(oh)

    # --- DASHI mask from ω_lowk ---
    base = smooth2d(omega_lp, cfg.dashi_smooth_k)
    Rlp = omega_lp - base
    Rn = np.clip(Rlp / (np.max(np.abs(Rlp)) + 1e-12), -1, 1)
    s = saturate_ternary(ternary_sym(Rn, cfg.dashi_tau), iters=8)
    m = (s != 0).astype(np.float64)

    kmag = np.sqrt(KX*KX + KY*KY)
    mid = (kmag > cfg.k_cut) & (kmag <= cfg.resid_mid_cut)
    high = (kmag > cfg.resid_mid_cut)
    n_mid = int(np.count_nonzero(mid))
    n_high = int(np.count_nonzero(high))

    # --- unpack kept mid-band coeffs ---
    offset = 2*M + 4
    k_keep = min(k_keep, n_mid, len(anchor_idx))
    kept_r = z[offset: offset + k_keep]
    kept_i = z[offset + k_keep: offset + 2*k_keep]

    if k_keep > 0:
        oh_flat = oh.ravel()
        oh_flat[anchor_idx[:k_keep]] = (kept_r + 1j*kept_i) * scale
        oh = oh_flat.reshape(N, N)

    def synth_band(band_mask, target_E, n_band):
        if n_band == 0 or target_E <= 0.0:
            return np.zeros((N, N), dtype=np.float64)

        # target_E = mean(|X|^2)/(N^4) on the band. Choose constant magnitude:
        mag = math.sqrt(target_E) * scale

        phases = rng.uniform(0.0, 2*np.pi, size=(N, N))
        X = np.zeros((N, N), dtype=np.complex128)
        X[band_mask] = mag * (np.cos(phases[band_mask]) + 1j*np.sin(phases[band_mask]))

        r = ifft2(X) * m  # gate in physical space

        # Convert target_E to target mean(r^2) using Parseval:
        # mean(r^2) = (1/N^4) * sum_band |X|^2 = (n_band * mag^2) / N^4 = n_band * target_E
        target_var = n_band * target_E
        cur_var = float(np.mean(r*r))
        if cur_var > 1e-30 and target_var > 0.0:
            r *= math.sqrt(target_var / cur_var)
        return r

    # Do not overwrite preserved mid coeffs when synthesizing
    if k_keep > 0 and n_mid > 0:
        mid_mask_flat = mid.ravel()
        mid_mask_flat[anchor_idx[:k_keep]] = False
        mid = mid_mask_flat.reshape(N, N)
        n_mid = int(np.count_nonzero(mid))

    r_mid = synth_band(mid, target_mid_E, n_mid)
    r_high = synth_band(high, target_high_E, n_high)
    omega_hat = omega_lp + r_mid + r_high
    return omega_hat, omega_lp, m, s


# -----------------------------
# Learned operator
# -----------------------------

def learn_linear_operator(Z: np.ndarray, ridge: float = 1e-3):
    X = Z[:-1]
    Y = Z[1:]
    D = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    return np.linalg.solve(XtX + ridge*np.eye(D), XtY)


# -----------------------------
# Main
# -----------------------------

def main():
    N = 64
    steps = 300
    dt = 0.01
    nu0 = 1e-4
    Cs = 0.17
    seed = 0

    cfg = ProxyConfig(k_cut=8.0, resid_mid_cut=12.0, dashi_tau=0.35, dashi_smooth_k=11)
    ridge = 1e-3
    decode_every = 1
    residual_seed = 12345

    snap_t = [0, steps//3, 2*steps//3, steps]

    traj, grid, sim_s = simulate_les_trajectory(N=N, steps=steps, dt=dt, nu0=nu0, Cs=Cs, seed=seed)
    dx, KX, KY, K2 = grid
    print(f"[baseline LES] steps={steps}  seconds={sim_s:.3f}  per_step_ms={1000*sim_s/steps:.3f}")

    t0 = perf_counter()
    Z = []
    mask_low0 = None
    anchor_idx = None
    for t in range(steps+1):
        z, mask_low, anchor_idx = encode_proxy(traj[t], grid, cfg, anchor_idx=anchor_idx)
        if mask_low0 is None:
            mask_low0 = mask_low
        Z.append(z)
    Z = np.stack(Z, axis=0)
    enc_s = perf_counter() - t0
    D = Z.shape[1]
    lowk_modes = int(np.sum(mask_low0))
    print(f"[encode] T={steps+1} D={D} seconds={enc_s:.3f} per_frame_ms={1000*enc_s/(steps+1):.3f}  lowk_modes={lowk_modes}")

    t1 = perf_counter()
    A = learn_linear_operator(Z, ridge=ridge)
    learn_s = perf_counter() - t1
    print(f"[learn] ridge={ridge} seconds={learn_s:.3f}")

    Zhat = np.zeros_like(Z)
    Zhat[0] = Z[0]
    t2 = perf_counter()
    for t in range(steps):
        Zhat[t+1] = Zhat[t] @ A
    roll_s = perf_counter() - t2
    print(f"[rollout] seconds={roll_s:.3f} per_step_us={1e6*roll_s/steps:.2f}  (proxy update only)")

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
            rng = np.random.default_rng(residual_seed + 1000003*t)
            omega_hat, omega_lp, m, s = decode_with_residual(Zhat[t], grid, cfg, mask_low0, anchor_idx, rng)
            prev_hat = omega_hat
        else:
            omega_hat = prev_hat

        Eh[t] = energy_from_omega(omega_hat, KX, KY, K2)
        Zh[t] = enstrophy(omega_hat)

        rel_l2[t] = float(np.linalg.norm(omega_true - omega_hat) / (np.linalg.norm(omega_true) + 1e-12))
        corr[t] = float(np.corrcoef(omega_true.flatten(), omega_hat.flatten())[0, 1])

        if t in snap_t:
            omega_hat_snap[t] = omega_hat.copy()

    dec_eval_s = perf_counter() - t3
    print(f"[decode+eval] seconds={dec_eval_s:.3f} per_frame_ms={1000*dec_eval_s/(steps+1):.3f}")
    print(f"[final] relL2={rel_l2[-1]:.4f}  corr={corr[-1]:.6f}  ΔE={Eh[-1]-Eb[-1]:+.3e}  ΔZ={Zh[-1]-Zb[-1]:+.3e}")

    # Plots
    plt.figure(figsize=(8,4))
    plt.plot(rel_l2)
    plt.xlabel("timestep")
    plt.ylabel("relative L2 error")
    plt.title("v4 residual closure: rollout error")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(corr)
    plt.xlabel("timestep")
    plt.ylabel("correlation")
    plt.title("v4 residual closure: rollout correlation")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(Eb, label="baseline E")
    plt.plot(Eh, label="learned+residual E")
    plt.xlabel("timestep")
    plt.ylabel("energy")
    plt.title("Energy drift (v4)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(Zb, label="baseline enstrophy")
    plt.plot(Zh, label="learned+residual enstrophy")
    plt.xlabel("timestep")
    plt.ylabel("0.5⟨ω²⟩")
    plt.title("Enstrophy drift (v4)")
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
        plt.title(f"ω̂ decoded+residual (t={t})")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(omega_true - omega_hat, origin="lower")
        plt.title("error ω-ω̂")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    baseline_ms = 1000*sim_s/steps
    proxy_us = 1e6*roll_s/steps
    print("\n=== Speed ledger (v4) ===")
    print(f"Baseline LES:      {baseline_ms:.3f} ms/step")
    print(f"Proxy update:      {proxy_us:.2f} µs/step")
    print(f"Decode+eval:       {1000*dec_eval_s/(steps+1):.3f} ms/frame")
    print("For 'fast mode', set decode_every>1 or decode only on demand.")

if __name__ == "__main__":
    main()
