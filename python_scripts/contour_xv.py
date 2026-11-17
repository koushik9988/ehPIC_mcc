import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import re


# ==========================================================
# Usage
# ==========================================================

if len(sys.argv) != 4:
    print("\nUsage:")
    print(" python contour_xv.py <output_folder> <species> <plot_type>")
    print(" plot_type: x-t | vx-t | vy-t | vz-t\n")
    sys.exit(1)


# ==========================================================
# Inputs
# ==========================================================

base_path = os.path.abspath(sys.argv[1].strip())
species = sys.argv[2].strip()
plot_type = sys.argv[3].strip()

phase_dir = os.path.join(base_path, "phase")
meta_file = os.path.join(base_path, "metadata.h5")
plot_dir = os.path.join(base_path, "plots")

os.makedirs(plot_dir, exist_ok=True)


print("\nBase path :", base_path)
print("Species   :", species)
print("Plot type :", plot_type)


# ==========================================================
# Checks
# ==========================================================

if not os.path.exists(meta_file):
    print("metadata.h5 not found")
    sys.exit(1)

if not os.path.isdir(phase_dir):
    print("phase directory not found")
    sys.exit(1)


# ==========================================================
# Read metadata
# ==========================================================

with h5py.File(meta_file, "r") as f:
    meta = f["/metadata"].attrs
    NUM_TS = meta["NUM_TS"]
    DT = meta["DT_coeff"]

    species_available = list(f["/metadata_species"].keys())

print("Available species:", species_available)

if species not in species_available:
    print("Invalid species name")
    sys.exit(1)


# ==========================================================
# Column selection
# ==========================================================

col_map = {
    "x-t": 0,
    "vx-t": 1,
    "vy-t": 2,
    "vz-t": 3
}

if plot_type not in col_map:
    print("invalid plot_type")
    sys.exit(1)

col_index = col_map[plot_type]


# ==========================================================
# Collect phase files
# ==========================================================

files = []

for fname in os.listdir(phase_dir):
    m = re.match(r"phase_(\d+)\.h5", fname)
    if m:
        ts = int(m.group(1))
        files.append((ts, fname))

files.sort()

print("Found", len(files), "phase files")


# ==========================================================
# PASS 1: get min/max
# ==========================================================

all_vals = []

print("Scanning data...")

for ts, fname in files:
    with h5py.File(os.path.join(phase_dir, fname), "r") as f:

        if species not in f:
            # empty file or no particles → skip
            continue

        data = f[f"{species}/phase"][:]

        if data.size == 0:
            continue

        all_vals.append(data[:, col_index])


if not all_vals:
    print("No particle data found for this species")
    sys.exit(1)


all_vals = np.concatenate(all_vals)

vmin = np.min(all_vals)
vmax = np.max(all_vals)

print("Value range:", vmin, vmax)


# ==========================================================
# PASS 2: build histogram
# ==========================================================

nbins = 200
ybins = np.linspace(vmin, vmax, nbins + 1)

hist_list = []
valid_steps = []

for ts, fname in files:

    with h5py.File(os.path.join(phase_dir, fname), "r") as f:

        if species not in f:
            continue

        data = f[f"{species}/phase"][:]

        if data.size == 0:
            continue

        hist, _ = np.histogram(data[:, col_index], bins=ybins)

        hist_list.append(hist)
        valid_steps.append(ts)


hist2d = np.array(hist_list).T


# ==========================================================
# Plot
# ==========================================================

labels = {
    "x-t": r"$x$",
    "vx-t": r"$v_x$",
    "vy-t": r"$v_y$",
    "vz-t": r"$v_z$"
}

fig, ax = plt.subplots(figsize=(8, 6))

extent = [
    valid_steps[0] * DT,
    valid_steps[-1] * DT,
    ybins[0],
    ybins[-1]
]

img = ax.imshow(
    hist2d,
    extent=extent,
    origin="lower",
    aspect="auto",
    interpolation="bilinear",
    cmap="plasma"
)

ax.set_xlabel(r"Time [$\omega_{pe}^{-1}$]")
ax.set_ylabel(labels[plot_type])

cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Particle count")

plt.tight_layout()

out_path = os.path.join(plot_dir, f"{plot_type}_{species}.png")
plt.savefig(out_path, dpi=300)



plt.show()
