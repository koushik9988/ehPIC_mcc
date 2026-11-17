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

path_out = os.path.join(base_path,"analysis")
os.makedirs(path_out, exist_ok=True)

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
    mFactor = 1

L = LDe
W = wpe

if normscheme in [2, 4]:
    L = LDi
    W = wpi


DT = DT_coeff * (1.0 / W)




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


EF = np.vstack(electric_field_data)

Nt_field, Nx = EF.shape

EF = np.vstack(electric_field_data)        # (Nt , Nx)
Nt, Nx = EF.shape

x  = np.linspace(0, NC, Nx, endpoint=False)
dx = x[1] - x[0]
time_full = np.arange(Nt) * DT_coeff * write_interval * mfactor


#====================================================
#  Spatial FFT → E(k,t)
#====================================================
Fk = np.fft.fft(EF, axis=1, norm='ortho')
kfft = 2*np.pi*np.fft.fftfreq(NC, dx)

# Only positive k modes
k_modes = kfft[:NC//2]
Fk_pos = Fk[:, :NC//2] * efield_norm * (2/np.sqrt(NC))
energy_spectrum = np.abs(Fk_pos)**2


avg_power = np.mean(energy_spectrum, axis=0)

peaks, _ = find_peaks(avg_power, height=np.max(avg_power)*0.001, distance=2)

print("Detected modes at k:", k_modes[peaks])

dt_snapshot = DT_coeff * write_interval * mfactor
phase_labels = []

for idx in peaks:
    k_dim = k_modes[idx]              
   
    Ek_t = Fk_pos[:, idx]                

    wspec = np.fft.fft(Ek_t)
    freqs = 2*np.pi*np.fft.fftfreq(Nt, dt_snapshot)

    # keep positive ω only
    pos = slice(0, Nt//2)
    peak_ω = freqs[pos][ np.argmax( np.abs(wspec[pos]) ) ]

    vph = peak_ω/k_dim                  

    #phase_labels.append(f"v_ph ≈ {vph:.3e}")
    phase_labels.append(f"$v_\phi$ = {vph:.3f}")



plt.figure()

for j, idx in enumerate(peaks):
    Ek = energy_spectrum[:, idx]
    plt.plot(time_full, np.log10(Ek), label=phase_labels[j])

plt.xlabel(r"$\omega_{pi}t$")
#plt.ylim([np.max(np.log10(energy_spectrum))*0.6, np.max(np.log10(energy_spectrum))*1.1])
plt.ylabel(r"$|E_k|^2$")
#plt.title("Evolution of Spectral Modes vs Time\n(labelled by Phase Velocity)")
plt.grid(alpha=0.4)
plt.legend(fontsize=9)
plt.tight_layout()

plt.savefig(pjoin(path_out,"Ek_vs_time_phase_velocity.pdf"),dpi=300)
plt.show()
