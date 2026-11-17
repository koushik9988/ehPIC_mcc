import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py

# ---- INPUT PATHS ----
path1 = sys.argv[1]
path2 = sys.argv[2]

def load_data(path):
    file = os.path.join(path, "logs", "collision_freq_log.csv")
    data = np.loadtxt(file, delimiter="\t", comments="#")
    
    time_step = data[:, 1]
    colle = data[:, 2]
    colli = data[:, 3]
    
    return time_step, colle, colli


meta_file = os.path.join(path1, "metadata.h5")
with h5py.File(meta_file, 'r') as f_meta:
    metadata_group = f_meta['/metadata']
    wpi = metadata_group.attrs['wpi']

# ---- LOAD BOTH DATASETS ----
t1, colle1, colli1 = load_data(path1)
t2, colle2, colli2 = load_data(path2)

# ---- PLOT ----
plt.figure(figsize=(8,5))

# Path 1
plt.plot(t1, colle1/wpi, color="blue", linestyle='-', label=r"$\nu_e$ no dust")
plt.plot(t1, colli1/wpi, color="red", linestyle='-', label=r"$\nu_i$ no dust")

# Path 2
plt.plot(t2, colle2/wpi, color="blue", linestyle='--', label=r"$\nu_e$ dust")
plt.plot(t2, colli2/wpi, color="red", linestyle='--', label=r"$\nu_i$ dust")

# ---- LABELS ----
plt.xlabel(r"$\omega_{pi} t$")
plt.ylabel(r"$\nu\ \ /\omega_{pi}$")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()