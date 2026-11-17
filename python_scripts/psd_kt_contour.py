"""
@kaushik kalita
@Date : 16/02/2025
This script computes the power spectral density (PSD) of the electric field E(x,t) in k-space and time, and visualizes it as a contour plot. 
The PSD is calculated using the Fast Fourier Transform (FFT) of the electric field snapshots at each time step. The resulting PSD(k,t)
 map helps to identify dominant modes and their evolution over time in plasma simulations.
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

mfactor = wpi / wpe

if normscheme in [1, 2]:
    mfactor = 1

L = LDe
W = wpe

if normscheme in [2, 4]:
    L = LDi
    W = wpi


DT = DT_coeff * (1.0 / W)

print("normscheme:", normscheme)
print("mFactor:", mfactor)


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
        #data = f["fielddata"]["den_ion"]["ion"][:]
        #data = f["fielddata"]["rho"]["rho"][:]
        electric_field_data.append(data[:, 1])  # Column 1 is the E-field value


EF = np.vstack(electric_field_data)  # Shape: (Nt, Nx)

Nt, Nx = EF.shape

# ----------------------------- Grid -----------------------------
x = np.linspace(0, NC, Nx, endpoint=False)
dx = x[1] - x[0]

# Time array (ω_pi t)
time_full = np.arange(Nt) * write_interval * DT_coeff * mfactor
print(time_full[-1], "omega_pi t at last time step")

# ----------------------------- FFT Setup -----------------------------
kfft = 2 * np.pi * np.fft.fftfreq(Nx, dx)
k_modes = kfft[:Nx // 2]

# ----------------------------- Compute PSD(k,t) -----------------------------
PSD_all = []

window = np.hanning(Nx)   # Apply once (same for all time)

figsize = np.array([150,150/1.618])#Figure size in mm (FOR SINGLE FIGURE)
dpi = 300                        #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

for time_idx in tqdm(range(Nt), desc="Computing PSD"):
    E_snapshot = EF[time_idx, :]

    # Apply window
    E_windowed = E_snapshot# * window

    # FFT
    F_k = np.fft.fft(E_windowed, norm='ortho')
    F_k_pos = F_k[:Nx // 2]

    # PSD
    PSD = np.abs(F_k_pos) ** 2

    PSD_all.append(PSD)

PSD_all = np.array(PSD_all)   # Shape: (Nt, Nk)

dk = k_modes[1] - k_modes[0]

path_fig = pjoin(base_path, "tempplots")
os.makedirs(path_fig, exist_ok=True)

plt.figure(figsize= figsize / 25.4, constrained_layout=True, dpi=dpi)

plt.imshow(np.log10(PSD_all),aspect='auto',origin='lower',interpolation='bilinear',
           extent=[k_modes[0]-dk/2, k_modes[-1]+dk/2,time_full[0], time_full[-1]],vmax=np.log10(np.max(PSD_all)),vmin=np.log10(np.max(PSD_all)*1e-5))

plt.xlim(0, 0.5)

# vertical lines
k_refs = [0.135122, 0.153548, 0.171974]
#for kval in k_refs:
    #plt.axvline(kval, color='white', linestyle='--', linewidth=1.0, alpha=0.9)

plt.xlabel('k')
plt.ylabel(r'$\omega_{pi} t$')
#plt.title('PSD(k, t)')

plt.colorbar(label=r'$\log|E(k, t)|^2$')

#plt.tight_layout()
plt.savefig(pjoin(path_fig, f"psd_kt_map_alpha.png"), dpi=dpi)
plt.show()