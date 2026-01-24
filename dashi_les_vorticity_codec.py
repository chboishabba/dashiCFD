#!/usr/bin/env python3
"""
dashi_les_vorticity_codec.py

Self-contained DASHI–LES vorticity simulator + structural compression tester.
"""

import numpy as np
import matplotlib.pyplot as plt

def fft2(a): return np.fft.fft2(a)
def ifft2(a): return np.fft.ifft2(a).real

def make_grid(N, L=2*np.pi):
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

def smooth2d(a, k=9):
    w = np.ones(k) / k
    tmp = np.apply_along_axis(lambda r: np.convolve(r, w, mode="same"), 1, a)
    return np.apply_along_axis(lambda c: np.convolve(c, w, mode="same"), 0, tmp)

def ternary_sym(X, tau):
    s = np.zeros_like(X, dtype=np.int8)
    s[X >=  tau] = +1
    s[X <= -tau] = -1
    return s

def majority_kernel_2d(s, radius=1):
    H, W = s.shape
    out = np.zeros_like(s)
    for i in range(H):
        i0 = max(0, i-radius)
        i1 = min(H, i+radius+1)
        for j in range(W):
            j0 = max(0, j-radius)
            j1 = min(W, j+radius+1)
            acc = int(np.sum(s[i0:i1, j0:j1]))
            out[i, j] = 1 if acc > 0 else -1 if acc < 0 else 0
    return out

def saturate_2d(s, iters=10, radius=1):
    for _ in range(iters):
        sn = majority_kernel_2d(s, radius)
        if np.array_equal(sn, s):
            break
        s = sn
    return s

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

def dashi_compress_reconstruct(omega, tau=0.35, smooth_k=11, q=0.05):
    H, W = omega.shape
    raw_bits = H * W * 32

    base = smooth2d(omega, smooth_k)
    X = np.clip((omega - base) / (np.max(np.abs(omega - base)) + 1e-12), -1, 1)

    s = saturate_2d(ternary_sym(X, tau))
    m = (s != 0)

    amps_q = np.round(omega[m] / q).astype(np.int16)

    omega_hat = np.zeros_like(omega)
    omega_hat[m] = amps_q.astype(float) * q
    omega_hat = smooth2d(omega_hat, smooth_k)

    resid_q = np.zeros_like(omega, dtype=np.int16)
    resid_q[m] = np.round((omega - omega_hat)[m] / q)

    omega_rec = omega_hat + resid_q * q

    bits_idx = np.sum(m) * (np.ceil(np.log2(H)) + np.ceil(np.log2(W)))
    bits_sign = np.sum(m)
    bits_amp = np.sum(m) * 16
    bits_res = np.sum(m) * 16
    bits_params = 64

    total_bits = bits_idx + bits_sign + bits_amp + bits_res + bits_params

    rel_l2 = np.linalg.norm(omega - omega_rec) / np.linalg.norm(omega)
    corr = np.corrcoef(omega.flatten(), omega_rec.flatten())[0,1]

    return omega_rec, dict(
        raw_bits=raw_bits,
        total_bits=total_bits,
        compression_ratio=raw_bits/total_bits,
        rel_l2=rel_l2,
        correlation=corr,
        support_cells=int(np.sum(m)),
    )

def main():
    np.random.seed(0)

    N = 64
    dx, KX, KY, K2 = make_grid(N)
    nu0 = 1e-4
    dt = 0.01
    steps = 200
    Cs = 0.17

    # --- initial condition ---
    omega0 = smooth2d(np.random.randn(N, N), 11)
    omega0 = (omega0 - omega0.mean()) / omega0.std()

    omega = omega0.copy()

    # --- time integration ---
    for _ in range(steps):
        psi = poisson_solve_minus_lap(omega, K2)
        u, v = velocity_from_psi(psi, KX, KY)
        nu_base = smagorinsky_nu(u, v, KX, KY, Cs, dx)

        base = smooth2d(omega, 11)
        X = np.clip((omega - base)/(np.max(np.abs(omega-base))+1e-12), -1, 1)
        s = saturate_2d(ternary_sym(X, 0.35))
        m = (s != 0).astype(float)

        nu_eff = nu0 + nu_base * (1 - 0.7*m)
        omega = step_rk2(omega, nu_eff, dt, KX, KY, K2)

    # --- compress + reconstruct ---
    q = 0.05
    omega_rec, stats = dashi_compress_reconstruct(omega, q=q)

    print("DASHI codec stats:", stats)

    # --- recompute structure for plotting ---
    base = smooth2d(omega, 11)
    X = np.clip((omega - base)/(np.max(np.abs(omega-base))+1e-12), -1, 1)
    s = saturate_2d(ternary_sym(X, 0.35))
    m = (s != 0)

    # --- plots ---
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0,0].imshow(omega0, origin="lower")
    axs[0,0].set_title("Initial ω")

    axs[0,1].imshow(omega, origin="lower")
    axs[0,1].set_title("Final ω (DASHI–LES)")

    axs[0,2].imshow(omega_rec, origin="lower")
    axs[0,2].set_title("Reconstructed ω̂")

    axs[1,0].imshow(omega - omega_rec, origin="lower")
    axs[1,0].set_title("Error ω − ω̂")

    axs[1,1].imshow(s, origin="lower", vmin=-1, vmax=1)
    axs[1,1].set_title("Ternary structure s*")

    axs[1,2].imshow(m, origin="lower")
    axs[1,2].set_title("Support mask m")

    for ax in axs.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
