import numpy as np
import h5py
import sys
import os
import glob
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# --- Settings ---
window = 100        # Steps for ACF window
stride = 100        # Steps between windows
nvbins = 50         # Number of velocity bins
mass_key = "mass"   # Metadata key for mass

if len(sys.argv) < 3:
    print("Usage: python kubo.py <data_path> <species>")
    sys.exit()

basepath = sys.argv[1]
species = sys.argv[2]
phasedir = os.path.join(basepath, "phase")
fielddir = os.path.join(basepath, "field")
outdir = os.path.join(basepath, "kubo_out")
os.makedirs(outdir, exist_ok=True)

# --- Helper: Sort files numerically ---
def get_files(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
    return files

phasefiles = get_files(phasedir, "phase_*.h5")
fieldfiles = get_files(fielddir, "data_*.h5")
print(f"Found {len(phasefiles)} phase files, {len(fieldfiles)} field files.")

# --- 1. Metadata Setup ---
with h5py.File(os.path.join(basepath, "metadata.h5"), 'r') as f:
    meta = f['/metadata'].attrs
    wpi = meta['wpi']
    wpe = meta['wpe']
    dx = meta.get('dx', 1.0) # Default to 1 if not found
    
    # Calculate time step (dt)
    dt_sim = meta['DT_coeff']
    nstep = meta['write_int_phase']
    mass = f[f'/metadata_species/{species}'].attrs['mass']
    
    # Normalization factor
    mfactor = wpi / wpe if 'wpe' in meta else 1.0
    dt = dt_sim * nstep * mfactor

# Map Phase file index -> Field file index
def get_ts(fname): return int(re.search(r'(\d+)', fname).group(1))
field_ts = np.array([get_ts(f) for f in fieldfiles])
phase_ts = np.array([get_ts(f) for f in phasefiles])

# Create a lookup array: map[phase_idx] = field_idx
filemap = np.zeros(len(phasefiles), dtype=int)
for i, pts in enumerate(phase_ts):
    filemap[i] = np.argmin(np.abs(field_ts - pts))

# --- 2. Bounce Frequency (Omega_b) ---
print("Estimating bounce frequency...")
sample_idx = np.linspace(0, len(fieldfiles)-1, 10, dtype=int)
E_stack = []
for i in sample_idx:
    with h5py.File(fieldfiles[i], 'r') as f:
        E_stack.append(f["fielddata/efield/efield"][:, 1])

E_stack = np.array(E_stack)
Nx = E_stack.shape[1]
# FFT to get dominant k
Ek = np.abs(np.fft.fft(E_stack, axis=1))[:, :Nx//2]
k_axis = 2 * np.pi * np.fft.fftfreq(Nx)[:Nx//2]
power = np.mean(Ek**2, axis=0)

peak = np.argmax(power[1:]) + 1 # Ignore DC
keff = k_axis[peak]
Eeff = np.sqrt(np.mean(E_stack**2))
wb = np.sqrt(keff * Eeff / mass)
wb = wb/wpe
Tb = 2 * np.pi / wb
print(f"k_eff: {keff:.3f}, wb: {wb:.3f}, Tb: {Tb:.3f}")

# --- 3. Velocity Binning ---
print("Binning velocities...")
with h5py.File(phasefiles[0], 'r') as f:
    v_sample = f[f"{species}/phase"][:, 1]

v_min, v_max = np.percentile(v_sample, [5, 95])
v_edges = np.linspace(v_min, v_max, nvbins + 1)
v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

# Storage for ACF [bins, time_lag]
acf_sum = np.zeros((nvbins, window))
acf_count = np.zeros(nvbins, dtype=int)

# --- 4. Main ACF Loop ---
# Loop over time windows
for t0 in range(0, len(phasefiles) - window, stride):
    print(f"Processing window starting at index {t0}...", end='\r')
    
    # Cache Field Data for this window (avoid re-reading)
    # We need field files corresponding to phase indices t0 -> t0+window
    needed_fields = sorted(list(set(filemap[t0 : t0+window])))
    field_cache = {}
    for fi in needed_fields:
        with h5py.File(fieldfiles[fi], 'r') as f:
            field_cache[fi] = f["fielddata/efield/efield"][:, 1]

    # Load Particle Data (Positions and Velocities)
    # 1. Get initial velocity to determine bin
    with h5py.File(phasefiles[t0], 'r') as f:
        data = f[f"{species}/phase"][:]
    pos0 = data[:, 0]
    vel0 = data[:, 1]
    
    # 2. Load trajectory positions for the whole window
    # Shape: (window, nparticles)
    pos_traj = np.zeros((window, len(vel0)))
    for j in range(window):
        with h5py.File(phasefiles[t0 + j], 'r') as f:
            pos_traj[j] = f[f"{species}/phase"][:, 0]

    # Process by Velocity Bin
    for b in range(nvbins):
        # Find particles in this bin
        pidx = np.where((vel0 >= v_edges[b]) & (vel0 < v_edges[b+1]))[0]
        if len(pidx) == 0: continue
        
        n_sel = len(pidx)
        xp = pos_traj[:, pidx] # (Time, Particles)

        # Interpolate Electric Field onto Particle Positions
        E_traj = np.zeros_like(xp)
        
        for j in range(window):
            # Get cached field
            fidx = filemap[t0 + j]
            Ex = field_cache[fidx]
            
            # Linear Interpolation
            # Assumes xp is in grid units. If xp is physical, divide by dx.
            xg = xp[j]
            ix = np.floor(xg).astype(int) % Nx
            dx_loc = xg - np.floor(xg)
            ix_next = (ix + 1) % Nx
            
            E_traj[j] = Ex[ix] * (1.0 - dx_loc) + Ex[ix_next] * dx_loc

        # Remove mean (DC component) per particle
        E_fluct = E_traj - np.mean(E_traj, axis=0)

        # Compute ACF using FFT (fast correlation)
        # Pad to avoid circular correlation
        nfft = 2 * window
        fft_E = np.fft.rfft(E_fluct, n=nfft, axis=0)
        corr = np.fft.irfft(fft_E * np.conj(fft_E), n=nfft, axis=0)
        
        # Normalize and store
        raw_acf = corr[:window] # Keep causal part
        norm_factor = raw_acf[0]
        
        # Avoid div by zero
        valid = norm_factor != 0
        if np.any(valid):
            # Sum normalized ACFs
            acf_sum[b] += np.sum(raw_acf[:, valid] / norm_factor[valid], axis=1)
            acf_count[b] += np.sum(valid)

print("\nAnalysis complete. Fitting...")

# --- 5. Fitting and Plotting ---
tau_axis = np.arange(window) * dt
kubo_nums = []
taus = []

def decay(t, tc): return np.exp(-t/tc)

for b in range(nvbins):
    if acf_count[b] < 10:
        taus.append(np.nan)
        kubo_nums.append(np.nan)
        continue

    # Average ACF for this bin
    C = acf_sum[b] / acf_count[b]
    
    # Fit exponential
    try:
        popt, _ = curve_fit(decay, tau_axis[:window//3], C[:window//3], p0=[10*dt])
        tc = np.abs(popt[0])
        taus.append(tc)
        kubo_nums.append(tc / Tb) # Kubo = tau_c / T_bounce
    except:
        taus.append(np.nan)
        kubo_nums.append(np.nan)

# Save
with h5py.File(os.path.join(outdir, "kubo_results.h5"), 'w') as f:
    f.create_dataset("v", data=v_centers)
    f.create_dataset("kubo", data=kubo_nums)
    f.create_dataset("tau_c", data=taus)
    f.create_dataset("acf", data=acf_sum/np.maximum(acf_count[:,None], 1))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(v_centers, kubo_nums, 'o-', color='tab:blue')
ax1.set_xlabel("Velocity")
ax1.set_ylabel("Kubo Number (tau_c / Tb)")
ax1.grid(True, alpha=0.3)

# Plot a few ACFs
step = max(1, nvbins//5)
for i in range(0, nvbins, step):
    if acf_count[i] > 0:
        C = acf_sum[i] / acf_count[i]
        ax2.plot(tau_axis, C, label=f"v={v_centers[i]:.1f}")

ax2.set_xlabel("Time")
ax2.set_ylabel("ACF")
ax2.legend(fontsize='small')

plt.tight_layout()
plt.savefig(os.path.join(outdir, "kubo_plot.png"))
print(f"Done. Saved to {outdir}")