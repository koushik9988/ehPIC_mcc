"""
@kaushik kalita
@Date : 16/02/2025
"""

import numpy as np
import h5py
import sys
import os
from os.path import join as pjoin
from scipy.constants import value as constants
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mp


# ------------------- CONSTNATS --------------------
eps0 = constants('electric constant')
kb   = constants('Boltzmann constant')
me   = constants('electron mass')
AMU  = constants('atomic mass constant')
e    = constants('elementary charge')
# ------------------------------------------------

# ---------------- ARGUMENT CHECK ----------------
if len(sys.argv) != 2:
    print("Usage: python kubo_local.py <path_to_data> <particle_type>")
    sys.exit(1)

base_path = os.path.abspath(sys.argv[1].strip())


#output_dir = 'analysis'

phase_dir = os.path.join(base_path, "phase")
field_dir = os.path.join(base_path, "field")
meta_file = os.path.join(base_path, "metadata.h5")

path_fig = os.path.join(base_path,"analysis")
os.makedirs(path_fig, exist_ok=True)

print("\nBase path :", base_path)



# ---------------- CHECK FILES ----------------
if not os.path.exists(meta_file):
    print("metadata.h5 not found")
    sys.exit(1)

if not os.path.isdir(phase_dir):
    print("phase directory not found")
    sys.exit(1)

if not os.path.isdir(field_dir):
    print("field directory not found")
    sys.exit(1)


#---------------------READ METADATA ----------------

with h5py.File(meta_file, 'r') as f_meta:

    metadata_group = f_meta['/metadata']

    NC = metadata_group.attrs['NC']
    NUM_TS = metadata_group.attrs['NUM_TS']

    write_interval = metadata_group.attrs['write_int']
    write_interval_phase = metadata_group.attrs['write_int_phase']

    DT_coeff = metadata_group.attrs['DT_coeff']

    wpe = metadata_group.attrs['wpe']
    wpi = metadata_group.attrs['wpi']

    LDe = metadata_group.attrs['LDe']
    LDi = metadata_group.attrs['LDi']

    normscheme = metadata_group.attrs['norm_scheme']
    n0 = metadata_group.attrs['density']

# ---------------- NORMALIZATION ----------------

mFactor = wpi / wpe

if normscheme in [1, 2]:
    mFactor = 1

L = LDe
W = wpe

if normscheme in [2, 4]:
    L = LDi
    W = wpi


L = 0.001
W = 13.56e6

DT = DT_coeff * (1.0 / W)

print("normscheme:", normscheme)
print("mFactor:", mFactor)


#------------------- LOAD FIELD --------------------

field_files = []

efield_norm = (n0 * e * L) / eps0

for fname in os.listdir(field_dir):
    if fname.startswith("data_") and fname.endswith(".h5"):
        ts = int(fname[5:-3])
        field_files.append((ts, fname))

field_files.sort()

electric_field_data = []

for ts, fname in field_files:

    with h5py.File(os.path.join(field_dir, fname), 'r') as f:
        # Access: fielddata/efield/efield
        data = f["fielddata"]["efield"]["efield"][:]
        electric_field_data.append(data[:, 1])  # Column 1 is the E-field value


EF = np.vstack(electric_field_data)  # Shape: (Nt, Nx)
Nt, Nx = EF.shape
x = np.linspace(0, NC, Nx, endpoint=False)
dx = x[1] - x[0]

print(f"Data shape: EF({Nt}, {Nx})")

# Spatial FFT
F_k = np.fft.fft(EF, axis=1, norm='ortho') * efield_norm * (2 / np.sqrt(NC))
kfft = 2 * np.pi * np.fft.fftfreq(Nx, dx)

# Temporal FFT
F_k_t = np.fft.fft(F_k, axis=0, norm='ortho')
omega_fft = 2 * np.pi * np.fft.fftfreq(Nt, DT_coeff * write_interval * mFactor)

# Shift FFT outputs to center zero frequency/wavenumber
F_k_t_shifted = np.fft.fftshift(F_k_t, axes=(0, 1))
k_shifted = np.fft.fftshift(kfft)
omega_shifted = np.fft.fftshift(omega_fft)

# Compute power spectrum
dispersion_power = np.abs(F_k_t_shifted)**2
dispersion_power /= np.max(dispersion_power)  # normalize


power_log = np.log10(dispersion_power)

power_log = power_log[:, ::-1]
k_shifted = -k_shifted[::-1]



################## fidn domiant w-k pair ######################################
psd_vs_k = np.mean(dispersion_power, axis=0)  # Shape (Nx,)

k_peak_indices, _ = find_peaks(psd_vs_k, height=np.max(psd_vs_k) * 0.01, distance=2)
dominant_k_values = -k_shifted[k_peak_indices]

dominant_omega_values = []
for k_idx in k_peak_indices:
    psd_vs_omega_at_k = dispersion_power[:, k_idx]  # Shape (Nt,)
    
    omega_peak_indices, _ = find_peaks(psd_vs_omega_at_k, height=np.max(psd_vs_omega_at_k) * 0.01, distance=1)
    
    if len(omega_peak_indices) > 0:
        strongest_omega_idx_in_peaks = np.argmax(psd_vs_omega_at_k[omega_peak_indices])
        strongest_omega_idx = omega_peak_indices[strongest_omega_idx_in_peaks]
        dominant_omega_values.append(omega_shifted[strongest_omega_idx])
    else:
        dominant_omega_values.append(np.nan)

dominant_omega_values = np.array(dominant_omega_values)

valid_indices = ~np.isnan(dominant_omega_values)
dominant_k_plot = dominant_k_values[valid_indices]
dominant_omega_plot = dominant_omega_values[valid_indices]
print(f"Found {len(dominant_k_plot)} valid (k, ω) pairs to plot.")

#for i in range(len(dominant_k_plot)):
#    print(dominant_k_plot[i],":",dominant_omega_plot[i])
#################
raw_analytic_IAW = True

EV_TO_K = e / kb            # convert eV → Kelvin  (≈ 11604 K/eV)
Te      = 1.0               # electron temperature [eV]
Ti      = 0.026               # ion temperature [eV]  — adjust as needed
mi      = 1.6726219e-27     # proton mass [kg]
LD      = LDi               # Debye length used in simulation normalisation

# IAW dispersion relation:
#   ω²(k) = (kb*Te/mi) * k²/(1 + k²λD²)  +  (kb*Ti/mi) * k²
# positive-k branch only; negative branch is symmetric
k_analytic = k_shifted[k_shifted > 0]/LDi

omega_IAW = np.sqrt((Te * EV_TO_K * kb / mi) * (k_analytic**2 / (1.0 + k_analytic**2 * LD**2))+ (Ti * EV_TO_K * kb / mi) * k_analytic**2)
s = 1000
alpha1 = s/(1+s)
omega_IAW = np.sqrt(((Ti * EV_TO_K * kb/ mi) * (1 + alpha1 + 3 * (k_analytic**2 * LDi**2) + 3 * (1 - alpha1) * (Ti / Te))) /((k_analytic* LDi)**2 + (Ti / Te) * (1 - alpha1))) * k_analytic
alpha0 = 0
omega_IAW0 = np.sqrt(((Ti * EV_TO_K * kb/ mi) * (1 + alpha0 + 3 * (k_analytic**2 * LDi**2) + 3 * (1 - alpha0) * (Ti / Te))) /((k_analytic* LDi)**2 + (Ti / Te) * (1 - alpha0))) * k_analytic

omega_IAW1 = np.sqrt(2*wpi**2 +  3*(wpi*LDi)**2*k_analytic**2)


figsize = np.array([150,150/1.618])  # Figure size in mm
dpi = 300
ppi = np.sqrt(1920**2 + 1200**2) / 24

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=12, serif='Computer Modern Roman')
mp.rc('axes', titlesize=12)
mp.rc('axes', labelsize=12)
mp.rc('xtick', labelsize=12)
mp.rc('ytick', labelsize=12)
mp.rc('legend', fontsize=20)

fig, ax = plt.subplots(figsize=figsize/25.4, dpi=ppi)
extent = [k_shifted[0], k_shifted[-1], omega_shifted[0], omega_shifted[-1]]


import scienceplots
plt.style.use(['science','no-latex'])

im = ax.imshow(power_log,extent=extent, origin='lower', aspect='auto', interpolation="bilinear", cmap='coolwarm', vmin=-4, vmax=-1)#RdBu_r
#im = ax.imshow(power_log, extent=extent, origin='lower', aspect='auto',
#               interpolation="bilinear", cmap='turbo', vmin=-3.5, vmax=-2)


if raw_analytic_IAW:
    # Four branches (±k, ±ω) for the full symmetric dispersion
    #ax.plot( k_analytic*LDi,  omega_IAW/wpi, color="#090DDB", linestyle='--', lw=2, label=r'FAW')
    #ax.plot( k_analytic*LDi,  omega_IAW0/wpi, color="#D608F1", linestyle='--', lw=2, label=r'IAW')
    #ax.plot( k_analytic*LDi,  omega_IAW1/wpi, color="#EAEE0B", linestyle='--', lw=2, label=r'ILW')
    #ax.plot(-k_analytic*LDi, -omega_IAW/wpi, color='black', linestyle='--', lw=1.2)
    #ax.plot( k_analytic*LDi, -omega_IAW/wpi, color='cyan', linestyle='--', lw=1.2)
    #ax.plot(-k_analytic*LDi,  omega_IAW/wpi, color='cyan', linestyle='--', lw=1.2)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.5)

#ax.scatter(dominant_k_plot, dominant_omega_plot, color='cyan', s=25, marker='x', label='Dominant Modes')
#ax.scatter(dominant_k, dominant_omega, color='cyan', s=50, label='Dominant (k,ω)')

ax.set_xlabel(r"$k\ \lambda_{Di}$")
ax.set_ylabel(r"$\omega \ / \omega_{pi}$")
#ax.set_xlim(0,.2)
#ax.set_ylim(0,2.0)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\log|E(k,\omega)|^2$")
plt.tight_layout()
print(pjoin(path_fig, "dispersion_psd.png"))
plt.savefig(pjoin(path_fig, "dispersion_alpha_1000.png"), dpi=300)
plt.show()
