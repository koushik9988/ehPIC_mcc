import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os

# ---- CLI arguments ----
if len(sys.argv) != 4:
    print("Usage: python3 distribution_fixedtime.py <base_path> <particle_type> <time_step>")
    print("Example: python3 distribution_fixedtime.py <../datafiles/data>  <electron> 10")
    sys.exit(1)

base_path = os.path.abspath(sys.argv[1].strip())
particle_type = sys.argv[2].strip()
ts = int(sys.argv[3])

# ---- Paths ----
phase_file = os.path.join(base_path, "phase", f"phase_{ts}.h5")
meta_file  = os.path.join(base_path, "metadata.h5")

print("\nBase path :", base_path)
print("Species   :", particle_type)
print("TS        :", ts)

# ---- Check files ----
if not os.path.exists(meta_file):
    print("metadata.h5 not found")
    sys.exit(1)

if not os.path.exists(phase_file):
    print("Phase file not found:", phase_file)
    sys.exit(1)

# ---- Read metadata ----
with h5py.File(meta_file, 'r') as f_meta:

    metadata = f_meta["/metadata"]

    DT_coeff = metadata.attrs["DT_coeff"]
    wpe = metadata.attrs["wpe"]
    wpi = metadata.attrs["wpi"]

mFactor = wpi / wpe

time = ts * DT_coeff * mFactor

# ---- Read phase file ----
with h5py.File(phase_file, 'r') as f:

    data_phase = f[f"{particle_type}/phase"][:]

vx = data_phase[:,1]

# ---- Compute f(v) ----
NBINS = 100

f_v, bin_edges = np.histogram(vx, bins=NBINS, density=True)

v = 0.5*(bin_edges[:-1] + bin_edges[1:])

# ---- Plot ----
plt.figure(figsize=(7,5))

plt.plot(v, f_v)

plt.xlabel("v")
plt.ylabel("f(v)")

plt.title(f"Velocity Distribution\nTS={ts}, time={time:.4f}")

plt.grid(True)

plt.show()