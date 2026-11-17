import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import matplotlib as mp

# Usage:
# python eepf.py <output_folder> [ts_start] [ts_end]
#
# Builds the time-averaged Electron Energy Probability Function (EEPF)
# in the central 10% of the spatial domain, normalized so that
# integral f(eps)*sqrt(eps) d(eps) = 1  (same convention as eduPIC's
# eepf.dat / Fig. 12).
#
# ts_start/ts_end (optional): restrict averaging to a window of phase
# snapshots (use the converged part of the run, ideally spanning many
# RF cycles for good statistics).

if len(sys.argv) < 2:
    print("Usage: python eepf.py <output_folder> [ts_start] [ts_end]")
    sys.exit()

path = os.path.abspath(sys.argv[1])
ts_start = int(sys.argv[2]) if len(sys.argv) > 2 else None
ts_end = int(sys.argv[3]) if len(sys.argv) > 3 else None

phase_dir = os.path.join(path, "phase")
meta_file = os.path.join(path, "metadata.h5")
plot_dir = os.path.join(path, "plots")
os.makedirs(plot_dir, exist_ok=True)

# -------------------------------
# Read metadata
# -------------------------------
with h5py.File(meta_file, "r") as f:
    meta = f["/metadata"].attrs
    NC = meta["NC"]
    wpe = 13.56E6 #meta["wpe"] if "wpe" in meta else 13.56e6
    LDe = 0.001#meta["LDe"] if "LDe" in meta else 1e-3

Ld = LDe  # length normalization used for velocity conversion

# -------------------------------
# Collect phase files
# -------------------------------
files = []
for fname in os.listdir(phase_dir):
    if fname.startswith("phase_") and fname.endswith(".h5"):
        ts = int(fname[6:-3])
        if ts_start is not None and ts < ts_start:
            continue
        if ts_end is not None and ts > ts_end:
            continue
        files.append((ts, fname))
files.sort()

if not files:
    print("No phase files found in the requested range.")
    sys.exit()

print(f"Averaging EEPF over {len(files)} phase snapshots "
      f"(ts {files[0][0]} .. {files[-1][0]})")

# -------------------------------
# Central 10% spatial window (in grid units, x in [0, NC])
# -------------------------------
x_lo = 0.0 * NC
x_hi = 0.1 * NC

# -------------------------------
# Physical constants
# -------------------------------
me = 9.10938356e-31
me = 6.6E-26
qe = 1.602176634e-19

energies_all = []

for ts, fname in files:
    with h5py.File(os.path.join(phase_dir, fname), "r") as f:
        if "electron/phase" not in f:
            continue
        #phase = f["electron/phase"][:]
        phase = f["ion/phase"][:]

    x = phase[:, 0]
    mask = (x >= x_lo) & (x <= x_hi)
    if not np.any(mask):
        continue

    # velocity components: at minimum vx (col 1); use vy, vz too if present
    ncols = phase.shape[1]
    vx = phase[mask, 1] * wpe * Ld
    v_sqr = vx**2
    if ncols >= 4:
        vy = phase[mask, 2] * wpe * Ld
        vz = phase[mask, 3] * wpe * Ld
        v_sqr = v_sqr + vy**2 + vz**2

    energy = 0.5 * me * v_sqr / qe  # eV
    energies_all.append(energy)

energies_all = np.concatenate(energies_all)

# -------------------------------
# Build EEPF: f(eps) normalized so that
# integral f(eps) sqrt(eps) d(eps) = 1
# -------------------------------
n_bins = 200
e_max = np.percentile(energies_all, 99.9999)  # avoid extreme outlier bins
bins = np.linspace(0, e_max, n_bins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_width = bins[1] - bins[0]

counts, _ = np.histogram(energies_all, bins=bins)

# raw probability density in energy
pdf = counts / (counts.sum() * bin_width)

# EEPF convention: f(eps) = pdf(eps) / sqrt(eps)
with np.errstate(divide="ignore", invalid="ignore"):
    eepf = np.where(bin_centers > 0, pdf / np.sqrt(bin_centers), 0.0)

# renormalize so that integral f(eps) sqrt(eps) d(eps) = 1
norm = np.sum(eepf * np.sqrt(bin_centers) * bin_width)
if norm > 0:
    eepf = eepf / norm

# -------------------------------
# Plot styling
# -------------------------------
figsize = np.array([150, 150 / 1.618])  # mm
dpi = 300

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

fig, ax = plt.subplots(figsize=figsize / 25.4, constrained_layout=True, dpi=dpi)

mask_pos = eepf > 0
ax.semilogy(bin_centers[mask_pos], eepf[mask_pos], color="tab:blue")

ax.set_xlabel(r"$\varepsilon\ [\mathrm{eV}]$")
ax.set_ylabel(r"$f(\varepsilon)\ [\mathrm{eV}^{-3/2}]$")
ax.set_xlim(0, e_max)
ax.grid(True, which="both", alpha=0.3)

out_path = os.path.join(plot_dir, "eepf.pdf")
plt.savefig(out_path, dpi=dpi)
print("Saved:", out_path)
plt.show()