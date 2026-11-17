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



# ---------------- CLI ARGUMENT CHECK ----------------
if len(sys.argv) != 3:
    print("Usage: python kubo_local.py <path_to_data> <particle_type>")
    sys.exit(1)

base_path = os.path.abspath(sys.argv[1].strip())
particle_type = sys.argv[2].strip()

output_dir = 'kubo_analysis'

phase_dir = os.path.join(base_path, "phase")
field_dir = os.path.join(base_path, "field")
meta_file = os.path.join(base_path, "metadata.h5")

path_out = os.path.join(base_path,"analysis", output_dir)
os.makedirs(path_out, exist_ok=True)

print("\nBase path :", base_path)
print("Species   :", particle_type)


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

    write_interval_field = metadata_group.attrs['write_int']
    write_interval_phase = metadata_group.attrs['write_int_phase']

    DT_coeff = metadata_group.attrs['DT_coeff']

    wpe = metadata_group.attrs['wpe']
    wpi = metadata_group.attrs['wpi']

    LDe = metadata_group.attrs['LDe']
    LDi = metadata_group.attrs['LDi']

    normscheme = metadata_group.attrs['norm_scheme']
    n0 = metadata_group.attrs['density']


    species_metadata_path = f'/metadata_species/{particle_type}'
    metadata_species = f_meta[species_metadata_path]

    species_mass = metadata_species.attrs['mass']
    species_charge = metadata_species.attrs['charge']
    species_temp = metadata_species.attrs['temperature']
    species_spwt = metadata_species.attrs['spwt']
    species_num = metadata_species.attrs['num_particles']


# ------------------- CONSTNATS --------------------
eps0 = constants('electric constant')
kb   = constants('Boltzmann constant')
me   = constants('electron mass')
AMU  = constants('atomic mass constant')
e    = constants('elementary charge')
# ------------------------------------------------

# ---------------- NORMALIZATION ----------------

mfactor = wpi / wpe

if normscheme in [1, 2]:
    mfactor = 1

L = LDe
W = wpe

if normscheme == 2:
    L = LDi
    W = wpi

if normscheme == 4:
    L = LDi
    #W = wpi

DT = DT_coeff * (1.0 / W)


mb = species_mass

dt_phase = write_interval_phase * DT_coeff * mfactor
dt_field = write_interval_field * DT_coeff * mfactor

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

#-----------------LOAD PARTICLE-----------------------
phase_files = []

for fname in os.listdir(phase_dir):
    if fname.startswith("phase_") and fname.endswith(".h5"):
        ts = int(fname[6:-3])
        phase_files.append((ts, fname))

phase_files.sort()
particle_data = []

for ts, fname in phase_files:
    with h5py.File(os.path.join(phase_dir, fname), 'r') as f:
        # Correct structure: particle_type/phase
        data_phase = f[f'{particle_type}/phase'][:]
        particle_data.append(data_phase)

phase_all = np.stack(particle_data)
Nt_particle, Np, _ = phase_all.shape
print(f"Loaded particle shape:", phase_all.shape)

# let say NUM_TS = 10000, write_int_field = 10 , write_int_phase =100
# filed data indices are 0,10,20,30,... = total = 1000 whoch is Nt_field
# phase indices are 0,100,200,300.... = total is 100 which is Nt_particle
# np.linspace(0, Nt_field - 1, Nt_particle) step = 1000/100 = 10
# retrun  = [0, 10, 20, 30, 40, ... 990]
# filed_indices(0) = 0 ,field_indices(1) = 10 
# meaning is 0 phase index correspond to 0 field index and 1 phase index correspod to 10 th field index
# field_indices(2)=>2 phase index correspond to 20 field idex
field_indices = np.round(np.linspace(0, Nt_field - 1, Nt_particle)).astype(int)

#NUMBER OF PARTICLE TO ANALYZE
num_particles_to_scan = min(Np, 500000)


# --------------DEFINE ANALYSIS WINDOW INDICES----------------
# Define time slice for analysis
t_start_index = 2000      # in wpe*t/wpi*t unit
t_end_index = 5000     # in wpe/wpi*t unit

# Convert to particle phase indices
t0_idx = int(round(t_start_index / dt_phase))
t_end_idx = int(round(t_end_index / dt_phase))

# Ensure indices are within bounds
t0_idx = max(0, min(t0_idx, Nt_particle - 1))
t_end_idx = min(t_end_idx, Nt_particle)
n_ti = t_end_idx - t0_idx

# Slice electric field for the analysis window
# Convert particle indices to field indices
f0_idx = field_indices[t0_idx]
f_end_idx = field_indices[t_end_idx - 1] + 1

#-------------------------------------------------------


###
# Sample velocities at several times across the analysis window
#///sample_times = np.linspace(t0_idx, t_end_idx - 1, min(10, n_ti)).astype(int)
#///v_sample = phase_all[np.ix_(sample_times, np.arange(num_particles_to_scan))][..., 1].ravel()

#vmin, vmax = np.percentile(v_sample, [1, 99])
#v_edges = np.linspace(vmin, vmax, nv_bins + 1)
#v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
##

#----------------bin center velocity determination-------------
#----------------bin center velocity determination-------------
nv_bins = 50

v_flat_list = []
for idx in range(t0_idx, t_end_idx, 10):  # every 10th snapshot
    v_flat_list.append(phase_all[idx, :num_particles_to_scan, 1])

v_flat = np.concatenate(v_flat_list)
vmin, vmax = np.percentile(v_flat, [1, 99])

v_edges = np.linspace(vmin, vmax, nv_bins + 1)
v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
#--------------------------------------------------------------


#nv_bins = 50
#v_sample = phase_all[Nt_particle, :num_particles_to_scan, 1]
#vmin, vmax = np.percentile(v_sample, [1, 99])

#v_edges = np.linspace(vmin, vmax, nv_bins + 1)
#v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
#--------------------------------------------------------------

#-----------------------BOUNCE FREQUNCY CACULATION STARTS--------------------
EF_slice = EF[f0_idx:f_end_idx, :] 

x = np.linspace(0, NC, Nx)
dx = x[1] - x[0]

Fk = np.fft.fft(EF_slice, axis=1, norm='ortho')

kfft = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)

k_pos = kfft[:Nx//2]
Fk_pos = Fk[:, :Nx//2] * efield_norm * (2.0 / np.sqrt(Nx))

Ek2 = np.abs(Fk_pos)**2


power_spectrum = np.mean(Ek2, axis=0)

threshold = np.max(power_spectrum) * 0.01

peaks, _ = find_peaks(power_spectrum, height=threshold)

if len(peaks) == 0:
    print("No peaks found in sliced field")
    sys.exit(1)

top_idx = peaks[np.argsort(power_spectrum[peaks])[-50:]]

k_modes = k_pos[top_idx] / L

E2_mean = np.mean(Ek2[:, top_idx], axis=0)

k_eff = np.sum(k_modes * E2_mean) / np.sum(E2_mean)
k_dom = k_pos[top_idx] / L
k_avg = np.average(k_dom)
k_max = k_pos[top_idx[np.argmax(power_spectrum[top_idx])]] / L

E_eff = np.sqrt(np.mean(np.sum(Ek2[:, top_idx], axis=1)))

omega_b = np.sqrt((e * k_avg * E_eff) / species_mass)

omega_b_norm = omega_b / W

Tb_eff = 2*np.pi / omega_b_norm



#-----------------------BOUNCE FREQUNCY CACULATION ENDS--------------------

# Update EF to use sliced field for main analysis
EF = EF_slice

# Update field_indices for the sliced particle time range
# Map particle indices [t0_idx, t_end_idx) to field indices [0, len(EF))
field_indices_updated = np.round(np.linspace(0, len(EF) - 1, n_ti)).astype(int)

#----------------- TIME WINDOWS - NON-OVERLAPPING 

# Window size in physical time units (scaled by bounce time)
window_phys = 30#1.2 * Tb_eff

# Convert to phase file index steps
window_steps = int(window_phys / dt_phase)

decorr_window = window_steps

# Maximum valid starting index (within analysis window)
ti_max_start = t_end_idx - window_steps

# Create NON-OVERLAPPING time windows within the analysis window
# like [t0 - (t0 + decorr),(t0 + decorr) - (t0 + 2*decorr ),(t0 + 2*decorr) - (t0 + 3*decorr )]
time_windows = np.arange(t0_idx, ti_max_start, decorr_window)


#--------------------DISPLAY IMPORTANT PARAMTERS-----------------------------------
if len(time_windows) == 0:
    print(f"\nERROR: No valid time windows!")
    print(f"  Need at least {window_steps} phase snapshots in analysis window")
    print(f"  Current window has {n_ti} snapshots")
    print(f"  Try: reducing Tb_eff multiplier or extending t_end_index")
    sys.exit(1)


print(f"\nLoaded field shape: ({Nt_field}, {Nx})")
print(f"Loaded particle shape: ({Nt_particle}, {Np}, 2)")

print("ANALYSIS TIME WINDOW")

print(f"Start Index (phase) : {t0_idx:6d}   (t = {t0_idx * dt_phase:8.2f})")
print(f"End Index (phase)   : {t_end_idx:6d}   (t = {t_end_idx * dt_phase:8.2f})")
print(f"Window size         : {n_ti:6d} phase snapshots")
print(f"Field slice         : [{f0_idx}:{f_end_idx}]  ({EF_slice.shape[0]} field snapshots)")


print("DOMINANT MODE PARAMETERS (from sliced field)")

print(f"k_eff               : {k_eff:.6f}")
print(f"k_avg               : {k_avg:.6f}")
print(f"k_max               : {k_max:.6f}")
print(f"E_eff               : {E_eff:.6f} (with 2/sqrt(N) normalization)")
print(f"omega_b / wpi       : {omega_b_norm:.6f}")
print(f"Tb (bounce period)  : {Tb_eff:.6f}")


print("WINDOW CONFIGURATION")

print(f"Window size         : {window_steps:6d} steps  ({window_phys:.2f} time units)")
print(f"Decorr. window      : {decorr_window:6d} steps  (= window_steps)")
print(f"Num. windows        : {len(time_windows):6d} independent windows")
print(f"First window at     : t = {time_windows[0] * dt_phase:.2f}")
print(f"Last window at      : t = {time_windows[-1] * dt_phase:.2f}")

#----------------------------------------------------------------------------------------

#------------------DATA STORAGE ------------------------------------------------------
bin_acf_sum = np.zeros((nv_bins, window_steps))
bin_acf_count = np.zeros(nv_bins)


n_windows        = len(time_windows)
acf_per_window   = np.zeros((n_windows, nv_bins, window_steps))
count_per_window = np.zeros((n_windows, nv_bins))
#---------------------------------------------------------------------------------

#------------------MAIN LOOP STARTS------------------------------------
print("\nProcessing time windows...")
for iw, t0 in enumerate(tqdm(time_windows)):

    t1 = t0 + window_steps

    vel0 = phase_all[t0, :, 1]

    win_acf_sum   = np.zeros((nv_bins, window_steps))
    win_acf_count = np.zeros(nv_bins)

    for ib in range(nv_bins):

        mask   = (vel0 >= v_edges[ib]) & (vel0 < v_edges[ib+1])
        p_idxs = np.where(mask)[0]

        for p in p_idxs:

            xp     = phase_all[t0:t1, p, 0]
            E_traj = np.zeros(window_steps)

            for j in range(window_steps):
                f_idx = field_indices[t0 + j] - f0_idx
                x     = xp[j]
                ix    = int(x) % Nx
                ixn   = (ix + 1) % Nx
                dxp   = x - ix
                E_traj[j] = EF[f_idx, ix] * (1 - dxp) + EF[f_idx, ixn] * dxp

            E_traj -= np.mean(E_traj)

            ac = np.correlate(E_traj, E_traj, mode='full')[window_steps - 1:]

            if ac[0] == 0:
                continue

            ac /= ac[0]

            win_acf_sum[ib]   += ac
            win_acf_count[ib] += 1

            bin_acf_sum[ib]   += ac
            bin_acf_count[ib] += 1

    # fill per-window arrays OUTSIDE the ib loop
    for ib in range(nv_bins):
        if win_acf_count[ib] > 0:
            acf_per_window[iw, ib] = win_acf_sum[ib] / win_acf_count[ib]

    count_per_window[iw] = win_acf_count


# global time-averaged ACF
acf = np.zeros_like(bin_acf_sum)

for ib in range(nv_bins):
    if bin_acf_count[ib] > 0:
        acf[ib] = bin_acf_sum[ib] / bin_acf_count[ib]

tau = np.arange(window_steps) * dt_phase


def exp_decay(t, tc):
    return np.exp(-t / tc)


# global Ku (time-averaged)
tau_c = np.zeros(nv_bins)
Ku    = np.zeros(nv_bins)

for ib in range(nv_bins):
    if bin_acf_count[ib] < 10:
        continue

    try:
        fit_end = int(0.3 * window_steps)
        popt, _ = curve_fit(exp_decay, tau[:fit_end], acf[ib, :fit_end],p0=[window_phys * 0.3])
        tau_c[ib] = popt[0]
        Ku[ib]    = popt[0] / Tb_eff
    except:
        pass

# per-window Ku
tau_c_per_window = np.zeros((n_windows, nv_bins))
Ku_per_window    = np.zeros((n_windows, nv_bins))

for iw in range(n_windows):
    for ib in range(nv_bins):
        if count_per_window[iw, ib] < 5:
            continue
        try:
            fit_end = int(0.5 * window_steps)
            popt, _ = curve_fit(exp_decay, tau[:fit_end],acf_per_window[iw, ib, :fit_end], p0=[window_phys * 0.3])
            tau_c_per_window[iw, ib] = popt[0]
            Ku_per_window[iw, ib]    = popt[0] / Tb_eff
        except:
            pass

# -------------------- SAVE --------------------
savefile = pjoin(path_out, f"kubo_ensemble_{particle_type}.h5")

with h5py.File(savefile, "w") as f:

    metadata = f.create_group("metadata")
    metadata.attrs["window_steps"]      = window_steps
    metadata.attrs["window_phys"]       = window_phys
    metadata.attrs["decorr_window"]     = decorr_window
    metadata.attrs["n_windows"]         = n_windows
    metadata.attrs["dt_phase"]          = dt_phase
    metadata.attrs["Tb_eff"]            = Tb_eff
    metadata.attrs["omega_b_norm"]      = omega_b_norm
    metadata.attrs["k_eff"]             = k_eff
    metadata.attrs["E_eff"]             = E_eff
    metadata.attrs["t_start_index"]     = t_start_index
    metadata.attrs["t_end_index"]       = t_end_index
    metadata.attrs["t0_idx"]            = t0_idx
    metadata.attrs["t_end_idx"]         = t_end_idx
    metadata.attrs["analysis_window_size"] = n_ti

    # global results
    f.create_dataset("v_centers",  data=v_centers)
    f.create_dataset("acf",        data=acf)
    f.create_dataset("tau",        data=tau)
    f.create_dataset("tau_c",      data=tau_c)
    f.create_dataset("Ku",         data=Ku)
    f.create_dataset("counts",     data=bin_acf_count)

    # per-window results
    f.create_dataset("acf_per_window",   data=acf_per_window)    # (n_windows, nv_bins, window_steps)
    f.create_dataset("count_per_window", data=count_per_window)  # (n_windows, nv_bins)
    f.create_dataset("tau_c_per_window", data=tau_c_per_window)  # (n_windows, nv_bins)
    f.create_dataset("Ku_per_window",    data=Ku_per_window)     # (n_windows, nv_bins)
    f.create_dataset("window_times",     data=time_windows * dt_phase)

print(f"\nSaved results to: {savefile}")

#*********************************************END***************************************************************

# EXTRA NOT NEEDED JUST TO QUICK VISYULIZATION
# -------------------- PLOTS --------------------

# Global Ku vs velocity
plt.figure(figsize=(7, 5))
plt.plot(v_centers, Ku, 'o-')
plt.xlabel("v")
plt.ylabel("Ku")
plt.title("Kubo Number vs Velocity")
plt.grid()
plt.savefig(pjoin(path_out, "Ku_vs_v.png"), dpi=200)

window_times = time_windows * dt_phase

plt.figure(figsize=(9, 5))
plt.pcolormesh(window_times,v_centers,Ku_per_window.T,cmap='hot',shading='auto')
plt.colorbar(label='Ku')
plt.xlabel("t")
plt.ylabel("v")
plt.title("Ku(v, t) — uniform = steady, varying = non-stationary")
plt.savefig(pjoin(path_out, "Ku_v_t.png"), dpi=200)

# Ku time evolution for a few velocity bins
plt.figure(figsize=(8, 4))
for ib in [10, 25, 40]:
    plt.plot(window_times, Ku_per_window[:, ib], 'o-',label=f'v = {v_centers[ib]:.2f}')
plt.xlabel("t")
plt.ylabel("Ku(t)")
plt.title("Ku temporal evolution per velocity class")
plt.legend()
plt.grid()
plt.savefig(pjoin(path_out, "Ku_time_evolution.png"), dpi=200)

plt.show()
