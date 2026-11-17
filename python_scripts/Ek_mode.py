import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import re

# Usage
if len(sys.argv) != 4:
    print("\nUsage:")
    print(" python growth_analysis.py <output_folder> <fit_start_time> <fit_end_time>")
    print(" Example: python growth_analysis.py ./data 50.0 150.0")
    print(" (time values are in simulation units)\n")
    sys.exit(1)

path = os.path.abspath(sys.argv[1])
t_start = float(sys.argv[2])
t_end = float(sys.argv[3])

field_dir = os.path.join(path, "field")
meta_file = os.path.join(path, "metadata.h5")

print("\nPath:", path)
print("R fit window: t =", t_start, "to", t_end)

with h5py.File(meta_file, "r") as f:
    meta = f["/metadata"].attrs
    NC = meta["NC"]
    DT_coeff = meta["DT_coeff"]
    write_interval = meta["write_int"]
    wpe = meta["wpe"]
    wpi = meta["wpi"]
    normscheme = meta["norm_scheme"]
    density = meta["density"]
    LDe = meta["LDe"]
    LDi = meta["LDi"]

# Normalization factor
mfactor = wpi / wpe
if normscheme in [1, 2]:
    mfactor = 1

L = LDe
if normscheme in [2, 4]:
    L = LDi


print(f"LDe: {LDe}, LDi: {LDi}")
efield_norm = (density * 1.602176565e-19 * L) / 8.85418782E-12

#empty list to store electric field data
files = []

for fname in os.listdir(field_dir):
    if fname.startswith("data_") and fname.endswith(".h5"):
        ts = int(fname[5:-3])
        files.append((ts, fname))

files.sort()

EF_list = []
times = []

print("Loading electric field data...")

for ts, fname in files:
    with h5py.File(os.path.join(field_dir, fname), "r") as f:

        #if "fielddata" not in f:
        #    continue

        #if "efield" not in f["fielddata"]:
        #    continue

        data = f["fielddata/efield/efield"][:]

        x = data[:, 0]
        E = data[:, 1]

        EF_list.append(E)
        times.append(ts)

#if not EF_list:
#    print("No electric field data found.")
#    sys.exit(1)

EF = np.array(EF_list)      # shape (Nt, Nx)
Nt, Nx = EF.shape

print("Loaded electric field data with shape:", EF.shape)

dx = x[1] - x[0]

print("dx =", dx," Nt =", Nt, "Nx =", Nx)

time_full = np.array(times) * DT_coeff * mfactor

#print(f"Time range: {time_full[0]:.3f} to {time_full[-1]:.3f}")

# Find indices corresponding to requested time window
t1 = np.argmin(np.abs(time_full - t_start))
t2 = np.argmin(np.abs(time_full - t_end))

# Ensure t2 > t1
if t2 <= t1:
    print(f"Error: End time must be after start time.")
    #print(f"Got indices: t1={t1} (t={time_full[t1]:.3f}), t2={t2} (t={time_full[t2]:.3f})")
    sys.exit(1)

print(f"fit window  in wpet = {time_full[t1]:.3f} to {time_full[t2]:.3f}")

F_k = np.fft.fft(EF, axis=1, norm='ortho') * efield_norm * (2 / np.sqrt(NC))
kfft = 2 * np.pi * np.fft.fftfreq(Nx, dx)

power_spectrum = np.mean(np.abs(F_k)**2, axis=0)

# Only positive k
k_max_idx = np.argmax(power_spectrum[:Nx//2])
k_max = kfft[k_max_idx]

print("Dominant k =", k_max)

time_fit = time_full[t1:t2]

E_k_max_full = np.abs(F_k[:, k_max_idx])
E_k_max_fit = E_k_max_full[t1:t2]


valid = E_k_max_fit > 0
time_valid = time_fit[valid]
log_E = np.log(E_k_max_fit[valid])

if len(time_valid) < 2:
    print("Not enough valid points for fitting.")
    sys.exit(1)

coeffs = np.polyfit(time_valid, log_E, 1)
gamma = coeffs[0]

print("Growth rate gamma =", gamma)

# Get the reference value at the start of fit window
E0_fit = E_k_max_fit[0]
t0_fit = time_fit[0]

# Create extended time array for plotting the fit
time_extended = time_full

# Compute fitted curve: E(t) = E0 * exp(gamma * (t - t0))
fitted_curve = E0_fit * np.exp(gamma * (time_extended - t0_fit))

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# theortical landau dam rate
gamma_theory = -np.sqrt(np.pi / 8) / (k_max **3) * np.exp(-1.0 / (2.0 * k_max**2) - 1.5)
print(f"Theoretical Landau gamma = {gamma_theory:.6f}")
theory_curve = E0_fit * np.exp(gamma_theory * (time_extended - t0_fit))

epsilon = 8.85418782E-12
me = 9.10938356e-31
# spwt
therm_energy = 0.5 * me * 600634234.381768 * (wpe * LDe)**2

#Power Spectrum
ax1.plot(kfft[:Nx//2], (0.5*epsilon*power_spectrum[:Nx//2])/therm_energy)
ax1.axvline(k_max, linestyle='--',label=f'$k_{{max}}$ = {k_max:.3f}')
ax1.set_xlabel(r'$k \lambda_{de}$')
ax1.set_ylabel(r'$|E(k)|^2$')
#ax1.set_title('Spatial Power Spectrum')
ax1.grid(True)
ax1.legend()

#ax2.semilogy(time_full, E_k_max_full, 'o-', label='Simulation', alpha=0.7, markersize=4)
ax2.semilogy(time_full, E_k_max_full)
ax2.semilogy(time_extended, fitted_curve,'--', linewidth=2, color='red',label=f'PIC: $\\gamma$ = {gamma:.4e}')
ax2.semilogy(time_extended, theory_curve, ':', linewidth=2, color='blue',label=f'Theory: $\gamma$ = {gamma_theory:.4e}')
ax2.set_xlabel(r'$\omega_{pe} t$')
ax2.set_ylabel(r'$|E(k_{max})|$')
#ax2.set_title('Growth / Damping of Dominant Mode')
ax2.grid(True)
ax2.legend()

plt.tight_layout()

out_path = os.path.join(path, "growth_analysis.pdf")
plt.savefig(out_path, dpi=300)
plt.show()
