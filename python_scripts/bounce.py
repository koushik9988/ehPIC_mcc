"""
Simple single-particle bounce frequency detector
Track one particle, compute FFT, find omega

Author: Kaushik Kalita
"""

import numpy as np
import h5py
import os
import sys
import matplotlib.pyplot as plt
from os.path import join as pjoin


# ---------------- INPUT ----------------

if len(sys.argv) != 4:
    print("Usage: python single_particle_bounce.py <path> <species>")
    sys.exit(1)

base_path = os.path.abspath(sys.argv[1])
species = sys.argv[2]
id = sys.argv[3]

phase_dir = os.path.join(base_path, "phase")
field_dir = os.path.join(base_path, "field")
meta_file = os.path.join(base_path, "metadata.h5")

output_dir = os.path.join(base_path, "analysis", "bounce_analysis")
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# READ METADATA
# ============================================================

with h5py.File(meta_file, 'r') as f:
    meta = f["/metadata"]
    
    DT_coeff = meta.attrs["DT_coeff"]
    write_int_phase = meta.attrs["write_int_phase"]
    
    wpe = meta.attrs["wpe"]
    wpi = meta.attrs["wpi"]
    normscheme = meta.attrs["norm_scheme"]

W = wpe if normscheme not in [2,4] else wpi
dt_phase = DT_coeff * write_int_phase


# ============================================================
# LOAD VELOCITY DATA
# ============================================================

phase_files = sorted([
    (int(f[6:-3]), f)
    for f in os.listdir(phase_dir)
    if f.startswith("phase_")
])

vel = []

for _, fname in phase_files:
    with h5py.File(os.path.join(phase_dir, fname), 'r') as f:
        vel.append(f[f"{species}/phase"][:,1])

vel = np.array(vel)

Nt, Np = vel.shape

print(f"Loaded velocity data: {Nt} timesteps, {Np} particles")


# ============================================================
# SELECT PARTICLE AND TIME SLICE
# ============================================================

# Time slice
t0 = int(0 / dt_phase)
t1 = int(1000 / dt_phase)

# Select a single particle (e.g., particle index 100)
particle_idx = int(id)

# Extract velocity for this particle
v_particle = vel[t0:t1, particle_idx]

# Remove mean (detrend)
v_particle = v_particle - np.mean(v_particle)

print(f"\nTracking particle {particle_idx}")
print(f"Time slice: t={t0} to t={t1} (N={len(v_particle)} points)")
print(f"Velocity range: [{v_particle.min():.3f}, {v_particle.max():.3f}]")


# ============================================================
# FFT ANALYSIS
# ============================================================

# Apply window to reduce spectral leakage
window = np.hanning(len(v_particle))
v_windowed = v_particle * window

# Compute FFT
fft = np.fft.fft(v_windowed)
freq = np.fft.fftfreq(len(v_windowed), dt_phase)

# Take positive frequencies only
maskf = freq > 0
freq_pos = freq[maskf]
power = np.abs(fft[maskf])**2

# Exclude DC component and very low frequencies
MIN_FREQ = 0.05
mask_valid = freq_pos > MIN_FREQ

freq_valid = freq_pos[mask_valid]
power_valid = power[mask_valid]

# Find peak
peak_idx = np.argmax(power_valid)
omega_bounce = 2 * np.pi * freq_valid[peak_idx]

print(f"\nBounce frequency detected:")
print(f"f = {freq_valid[peak_idx]:.4f} (1/time units)")
print(f"ω/W = {omega_bounce:.4f} (normalized)")


# ============================================================
# PLOT RESULTS
# ============================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Velocity time series
time = np.arange(t0, t1) * dt_phase
ax1.plot(time, v_particle, 'b-', linewidth=1.5)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Velocity (v - mean)', fontsize=12)
ax1.set_title(f'Particle {particle_idx} velocity oscillation', fontsize=14)
ax1.grid(alpha=0.3)

# Plot 2: Power spectrum
omega_array = 2 * np.pi * freq_valid
ax2.plot(omega_array , power_valid / np.max(power_valid), 'b-', linewidth=2, label='FFT Power')
ax2.axvline(omega_bounce , color='r', linestyle='--', linewidth=2, 
            label=f'Peak: ω/W = {omega_bounce:.4f}')
ax2.set_xlabel('ω / W', fontsize=12)
ax2.set_ylabel('Normalized Power', fontsize=12)
ax2.set_title('Bounce frequency spectrum', fontsize=14)
ax2.set_xlim(left=0)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(pjoin(output_dir, "single_particle_bounce.png"), dpi=200, bbox_inches='tight')
plt.show()

print(f"\nPlot saved to: {pjoin(output_dir, 'single_particle_bounce.png')}")