"""
Usage:
    python hermite_single.py <data_path> <species> <phase_filename> [drift]

    data_path      : directory containing metadata.h5 and phase/ subfolder
    species        : particle type string  (e.g. "ion", "electron")
    phase_filename : specific phase file   (e.g. "phase_5000.h5")
    drift          : optional float, bulk drift velocity to subtract (default 0)

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import h5py
import sys
import os
from os.path import join as pjoin


def compute_hermite_basis(v, M):
    Nv = len(v)
    psi = np.zeros((M, Nv))
    psi[0] = np.pi**(-0.25) * np.exp(-v**2 / 2)
    if M > 1:
        psi[1] = np.sqrt(2) * v * psi[0]
    for m in range(1, M - 1):
        psi[m + 1] = (np.sqrt(2 / (m + 1)) * v * psi[m]
                      - np.sqrt(m / (m + 1)) * psi[m - 1])
    return psi


#------------------- Parse Arguments -------------------
if len(sys.argv) < 4:
    print("Usage: python hermite_single.py <data_path> <species> <phase_filename> [drift]")
    sys.exit(1)


data_path      = os.path.abspath(sys.argv[1].strip())
species        = sys.argv[2].strip()
ts_step        = sys.argv[3].strip()
ts_step        = int(ts_step)
phase_filename = f"phase_{ts_step}.h5"
drift          = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

print("\nData path      :", data_path)
print("Species        :", species)
print("Phase file     :", phase_filename)
print("Drift velocity :", drift)

#------------------- Check Paths -------------------
meta_file  = pjoin(data_path, "metadata.h5")
phase_path = pjoin(data_path, "phase", phase_filename)

if not os.path.exists(meta_file):
    print("metadata.h5 not found")
    sys.exit(1)

if not os.path.exists(phase_path):
    print(f"Phase file not found: {phase_path}")
    sys.exit(1)

#------------------- Read Metadata -------------------
f_meta = h5py.File(meta_file, 'r')
meta   = f_meta['/metadata']

NC       = int(meta.attrs['NC'])
NUM_TS   = int(meta.attrs['NUM_TS'])
DT_coeff = float(meta.attrs['DT_coeff'])
wpe      = float(meta.attrs['wpe'])
wpi      = float(meta.attrs['wpi'])

f_meta.close()

mfactor = wpi / wpe

#------------------- Load Phase Data -------------------
with h5py.File(phase_path, 'r') as pf:
    data = pf[f"{species}/phase"][:]

x_raw = data[:, 0]
v_raw = data[:, 1] - drift

# Infer timestep number from filename  (e.g. phase_5000.h5 -> ts=5000)
ts_num   = int(os.path.splitext(phase_filename)[0].split("_")[1])
time_val = ts_num * DT_coeff * mfactor

print(f"\nLoaded {len(x_raw)} particles  |  t = {time_val:.3f} wpi^-1")
print(f"v range (after drift): [{v_raw.min():.3f}, {v_raw.max():.3f}]")

#------------------- Grid Parameters -------------------
Nx    = NC
Nv    = 1024
M     = 200    # number of Hermite modes
n_max = 200    # number of Fourier modes (k)

#------------------- Build Velocity Grid -------------------
vmin  = v_raw.min()
vmax  = v_raw.max()
v_buf = 0.1 * (vmax - vmin)
vmin -= v_buf
vmax += v_buf

v_grid    = np.linspace(vmin, vmax, Nv)
v_centers = 0.5 * (v_grid[:-1] + v_grid[1:])
dv        = v_centers[1] - v_centers[0]

#------------------- Build Spatial Grid & Bin Particles -------------------
x_grid = np.linspace(x_raw.min(), x_raw.max(), Nx)
dx     = x_grid[1] - x_grid[0]

ix    = np.searchsorted(x_grid, x_raw) - 1
valid = (ix >= 0) & (ix < Nx)
ix    = ix[valid]
v     = v_raw[valid]

f_xv = np.zeros((Nx, Nv - 1))
for i in range(Nx):
    vi = v[ix == i]
    if vi.size:
        f_xv[i], _ = np.histogram(vi, bins=v_grid, density=True)
f_xv /= Nx

integral_f = np.sum(f_xv) * dx * dv
print(f"Normalization check: integral of (f dx dv) = {integral_f:.4f}  (should be close to 1)")

#------------------- Hermite Decomposition -------------------
psi          = compute_hermite_basis(v_centers, M)   # (M, Nv-1)
psi_weighted = psi * dv                              # multiply by dv

f_m = f_xv @ psi_weighted.T    # (Nx, M)  Hermite coefficients at each x

#------------------- Fourier Transform along x -------------------
# old line f_nm  = np.fft.fft(f_m, axis=0)[:n_max]   # (n_max, M)
f_nm  = np.fft.rfft(f_m, axis=0)            # (Nx//2+1, M)
n_max = min(200, f_nm.shape[0])             # never exceed what rfft gives
f_nm  = f_nm[:n_max]

#------------------- Power Spectra -------------------
E_m   = np.mean(np.abs(f_m)**2, axis=0)   # (M,)  x-averaged Hermite power
power = np.abs(f_nm)**2                    # (n_max, M) Fourier-Hermite power

#------------------- Output Directory -------------------
outdir = pjoin(data_path, "paper_plots")
os.makedirs(outdir, exist_ok=True)

mp.rc('font', family='sans-serif', size=10)
mp.rc('axes', titlesize=10, labelsize=10)


fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

ax.plot(np.arange(M), E_m, lw=1.5, label=rf"$\omega_{{pi}} t = {time_val:.1f}$")

m_ref = np.arange(40, M)
ax.plot(m_ref, E_m[40] * (m_ref / 40)**(-1/2), "k:", lw=2, label=r"$m^{-5/2}$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m$")
ax.set_ylabel(r"$\langle |f_m|^2 \rangle_x$")
ax.set_title(rf"Hermite spectrum — {phase_filename}  ($\omega_{{pi}} t={time_val:.1f}$)")
ax.legend()

out1 = pjoin(outdir, f"hermite_spectrum_{ts_num}.pdf")
fig.savefig(out1, dpi=300)
plt.show()
print(f"Saved: {out1}")


fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

im = ax.imshow(np.log10(power.T),origin="lower",aspect="auto",interpolation="bilinear",cmap="RdYlBu_r",extent=[0, n_max, 0, M])#, vmin=-7, vmax=-5)

ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$m$")
ax.set_title(rf"Fourier-Hermite power  $\omega_{{pi}} t = {time_val:.1f}$")
fig.colorbar(im, ax=ax, label=r"$\log_{10}|f_{nm}|^2$")

out2 = pjoin(outdir, f"fourier_hermite_{ts_num}.png")
fig.savefig(out2, dpi=300)
plt.show()
print(f"Saved: {out2}")

print("Done.")