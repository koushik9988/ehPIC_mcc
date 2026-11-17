import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

# Usage:
# python eedf.py <output_folder>

if len(sys.argv) != 2:
    print("Usage: python eedf.py <output_folder>")
    sys.exit()

base_path = os.path.abspath(sys.argv[1])

phase_dir = os.path.join(base_path, "phase")
meta_file = os.path.join(base_path, "metadata.h5")

# -------------------------------
# Read metadata
# -------------------------------
with h5py.File(meta_file, "r") as f:
    meta = f["metadata"]
    NC = meta.attrs["NC"]
    wpe = 13.56e6#meta.attrs["wpe"]

# -------------------------------
# Collect phase files
# -------------------------------
phase_files = []

for fname in os.listdir(phase_dir):
    if fname.startswith("phase_") and fname.endswith(".h5"):
        ts = int(fname[6:-3])
        phase_files.append((ts, fname))

phase_files.sort()

print("Found", len(phase_files), "phase files")

# -------------------------------
# Read one phase file (last file)
# -------------------------------
phase_file = os.path.join(phase_dir, phase_files[-1][1])

with h5py.File(phase_file, "r") as f:
    phase = f["ion/phase"][:]     

datax = phase[:,0]
datavx = phase[:,1]

# -------------------------------
# Probe location
# -------------------------------
x_probe = 60.0
dx = 1

mask = np.abs(datax - x_probe) < dx

# -------------------------------
# Convert normalized velocity
# -------------------------------
Ld = 0.001
v = datavx[mask] * wpe * Ld

# -------------------------------
# Energy (eV)
# -------------------------------
me = 9.10938356e-31
qe = 1.602176634e-19

energy = 0.5 * me * v**2 / qe

# -------------------------------
# Plot EEDF
# -------------------------------
plt.figure(figsize=(6,4))
plt.hist(energy, bins=100)#, density=True)
plt.xlabel("Energy(eV)")
plt.ylabel("EEDF")
plt.title(f"EEDF at x = {x_probe}")
plt.grid(True)
plt.show()

