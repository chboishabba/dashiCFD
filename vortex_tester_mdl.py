import numpy as np

import matplotlib.pyplot as plt



# -----------------------------

# Minimal 2D vorticity flow + (toy) LES + DASHI gating

# Periodic box, pseudo-spectral derivatives (FFT), RK2 time stepping

# -----------------------------



np.random.seed(0)



N = 64

L = 2*np.pi

dx = L / N

x = np.linspace(0, L, N, endpoint=False)

y = np.linspace(0, L, N, endpoint=False)

yy, xx = np.meshgrid(y, x, indexing="ij")



# Wavenumbers

kx = np.fft.fftfreq(N, d=dx) * 2*np.pi

ky = np.fft.fftfreq(N, d=dx) * 2*np.pi

KY, KX = np.meshgrid(ky, kx, indexing="ij")

K2 = KX**2 + KY**2

K2[0, 0] = 1.0  # avoid divide by zero for inversion; we'll force psi_hat[0,0]=0 anyway



def fft2(a): return np.fft.fft2(a)

def ifft2(a): return np.fft.ifft2(a).real



def deriv_x(a):

    return ifft2(1j * KX * fft2(a))



def deriv_y(a):

    return ifft2(1j * KY * fft2(a))



def laplacian(a):

    return ifft2(-K2 * fft2(a))



def poisson_solve_minus_lap(omega):

    # Solve ∇^2 psi = -omega  => psi_hat = omega_hat / K2 with psi_hat[0,0]=0

    oh = fft2(omega)

    psih = oh / K2

    psih[0, 0] = 0.0

    return ifft2(psih)



def velocity_from_psi(psi):

    u = deriv_y(psi)      # u = dψ/dy

    v = -deriv_x(psi)     # v = -dψ/dx

    return u, v



def smooth2d(a, k=9):

    w = np.ones(k) / k

    tmp = np.apply_along_axis(lambda r: np.convolve(r, w, mode="same"), 1, a)

    out = np.apply_along_axis(lambda c: np.convolve(c, w, mode="same"), 0, tmp)

    return out



def ternary_sym(X, tau):

    s = np.zeros_like(X, dtype=np.int8)

    s[X >=  tau] = +1

    s[X <= -tau] = -1

    return s



def K_majority_2d(s, radius=1):

    H, W = s.shape

    out = np.zeros_like(s)

    for i in range(H):

        i0 = max(0, i-radius)

        i1 = min(H, i+radius+1)

        for j in range(W):

            j0 = max(0, j-radius)

            j1 = min(W, j+radius+1)

            acc = int(np.sum(s[i0:i1, j0:j1]))

            if acc > 0: out[i, j] = +1

            elif acc < 0: out[i, j] = -1

            else: out[i, j] = 0

    return out



def saturate_2d(s, max_iter=15, radius=1):

    for _ in range(max_iter):

        s_next = K_majority_2d(s, radius=radius)

        if np.array_equal(s_next, s):

            break

        s = s_next

    return s



def strain_mag(u, v):

    du_dx = deriv_x(u)

    du_dy = deriv_y(u)

    dv_dx = deriv_x(v)

    dv_dy = deriv_y(v)

    Sxx = du_dx

    Syy = dv_dy

    Sxy = 0.5*(du_dy + dv_dx)

    # |S| = sqrt(2 S_ij S_ij) where S_ij S_ij = Sxx^2 + Syy^2 + 2*Sxy^2 in 2D

    return np.sqrt(2.0*(Sxx**2 + Syy**2 + 2.0*(Sxy**2)) + 1e-30)



def smagorinsky_nu(u, v, Cs=0.17, Delta=dx):

    Smag = strain_mag(u, v)

    return (Cs * Delta)**2 * Smag



def rhs_vorticity(omega, nu_eff):

    # ω_t + u·∇ω = ν_eff ∇²ω   (toy: spatially varying ν treated as ν∇²ω)

    psi = poisson_solve_minus_lap(omega)

    u, v = velocity_from_psi(psi)

    adv = u * deriv_x(omega) + v * deriv_y(omega)

    diff = nu_eff * laplacian(omega)

    return -adv + diff



def step_rk2(omega, nu_eff, dt):

    k1 = rhs_vorticity(omega, nu_eff)

    omega1 = omega + dt*k1

    k2 = rhs_vorticity(omega1, nu_eff)

    return omega + 0.5*dt*(k1 + k2)



# Initial vorticity: random smooth field (decaying turbulence-like)

omega0 = smooth2d(np.random.randn(N, N), k=11)

omega0 = omega0 - np.mean(omega0)

omega0 = omega0 / (np.std(omega0) + 1e-12)



# Parameters

nu0 = 1e-4

dt = 0.01

steps = 250

Cs = 0.17



# Two runs: baseline LES vs DASHI-gated LES

def run_sim(gated=False, alpha=0.7, tau=0.35):

    omega = omega0.copy()

    enstrophy = []

    for _ in range(steps):

        psi = poisson_solve_minus_lap(omega)

        u, v = velocity_from_psi(psi)

        nu_t_base = smagorinsky_nu(u, v, Cs=Cs, Delta=dx)

        

        if gated:

            # DASHI structural layer on vorticity anomaly

            omega_base = smooth2d(omega, k=11)

            X = omega - omega_base

            X = X / (np.max(np.abs(X)) + 1e-12)

            X = np.clip(X, -1, 1)

            s0 = ternary_sym(X, tau)

            s_star = saturate_2d(s0, max_iter=10, radius=1)

            m = (s_star != 0).astype(np.float64)

            g = (1.0 - alpha * m)  # suppress SGS on coherent signed structure

            nu_t = nu_t_base * g

        else:

            nu_t = nu_t_base

        

        nu_eff = nu0 + np.maximum(0.0, nu_t)

        omega = step_rk2(omega, nu_eff, dt)

        enstrophy.append(0.5*np.mean(omega**2))

    return omega, np.array(enstrophy)



omega_baseline, Z_baseline = run_sim(gated=False)

omega_gated, Z_gated = run_sim(gated=True)



# Plot: enstrophy decay comparison

plt.figure(figsize=(8,4))

plt.plot(Z_baseline, label="baseline LES (Smagorinsky)")

plt.plot(Z_gated, label="DASHI-gated LES")

plt.xlabel("timestep")

plt.ylabel("enstrophy (0.5⟨ω²⟩)")

plt.title("2D decaying vorticity: enstrophy decay")

plt.legend()

plt.tight_layout()

plt.show()



# Plot: vorticity snapshots

plt.figure(figsize=(5,5))

plt.imshow(omega0, origin="lower")

plt.title("Initial vorticity ω")

plt.colorbar()

plt.tight_layout()

plt.show()



plt.figure(figsize=(5,5))

plt.imshow(omega_baseline, origin="lower")

plt.title("Final ω (baseline LES)")

plt.colorbar()

plt.tight_layout()

plt.show()



plt.figure(figsize=(5,5))

plt.imshow(omega_gated, origin="lower")

plt.title("Final ω (DASHI-gated LES)")

plt.colorbar()

plt.tight_layout()

plt.show()



# Plot: show the structural mask on final gated run

# Recompute s* on final omega_gated for visualization

omega_base_f = smooth2d(omega_gated, k=11)

Xf = omega_gated - omega_base_f

Xf = Xf / (np.max(np.abs(Xf)) + 1e-12)

Xf = np.clip(Xf, -1, 1)

s0f = ternary_sym(Xf, 0.35)

s_star_f = saturate_2d(s0f, max_iter=10, radius=1)

m_f = (s_star_f != 0).astype(np.uint8)



plt.figure(figsize=(5,5))

plt.imshow(s_star_f, origin="lower", vmin=-1, vmax=1)

plt.title("Final s* (ternary structure on ω)")

plt.colorbar()

plt.tight_layout()

plt.show()



plt.figure(figsize=(5,5))

plt.imshow(m_f, origin="lower", vmin=0, vmax=1)

plt.title("Final support mask m (structure locations)")

plt.colorbar()

plt.tight_layout()

plt.show()
