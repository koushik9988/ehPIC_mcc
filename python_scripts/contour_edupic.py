import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import matplotlib as mp

# Usage:
# python xt_overview.py <output_folder> [ts_start] [ts_end]
#
# Builds a multi-panel spatio-temporal ("xt") overview figure, similar
# to Fig. 11 of the eduPIC paper: electron density, ion density,
# potential, electric field, electron current, ion current, electron
# power density, ion power density (8 panels, 4 rows x 2 cols).
#
# ts_start/ts_end (optional): restrict to a single RF-cycle-scale window
# so the panels show one period clearly, like the paper's figure
# (t axis spans one RF cycle there).

if len(sys.argv) < 2:
    print("Usage: python xt_overview.py <output_folder> [ts_start] [ts_end]")
    sys.exit()

path = os.path.abspath(sys.argv[1])
ts_start = int(sys.argv[2]) if len(sys.argv) > 2 else None
ts_end = int(sys.argv[3]) if len(sys.argv) > 3 else None

field_dir = os.path.join(path, "field")
meta_file = os.path.join(path, "metadata.h5")
plot_dir = os.path.join(path, "plots")
os.makedirs(plot_dir, exist_ok=True)

# -------------------------------
# Read metadata
# -------------------------------
with h5py.File(meta_file, "r") as f:
    meta = f["/metadata"].attrs
    DT_coeff = meta["DT_coeff"]
    NC = meta["NC"]

# -------------------------------
# Collect field files
# -------------------------------
files = []
for fname in os.listdir(field_dir):
    if fname.startswith("data_") and fname.endswith(".h5"):
        ts = int(fname[5:-3])
        if ts_start is not None and ts < ts_start:
            continue
        if ts_end is not None and ts > ts_end:
            continue
        files.append((ts, fname))
files.sort()

if not files:
    print("No field files found in the requested range.")
    sys.exit()

print(f"Building xt overview from {len(files)} files "
      f"(ts {files[0][0]} .. {files[-1][0]})")

# -------------------------------
# Panels to build: (group, dataset, title, cbar label)
# Adjust group/dataset names here if your file layout differs.
# -------------------------------
panels = [
    ("den_electron", "electron", r"$n_e$", r"$n_e\ [\mathrm{m}^{-3}]$"),
    ("den_ion",      "ion",      r"$n_i$", r"$n_i\ [\mathrm{m}^{-3}]$"),
    ("pot",          "potential", r"$\Phi$", r"$\Phi\ [\mathrm{V}]$"),
    ("efield",       "efield",    r"$E$", r"$E\ [\mathrm{V/m}]$"),
    ("current",      "electron", r"$j_e$", r"$j_e\ [\mathrm{A/m^2}]$"),
    ("current",      "ion",      r"$j_i$", r"$j_i\ [\mathrm{A/m^2}]$"),
    ("heating",      "electron", r"$P_e$", r"$P_e\ [\mathrm{W/m^3}]$"),
    ("heating",      "ion",      r"$P_i$", r"$P_i\ [\mathrm{W/m^3}]$"),
]

x_positions = None
times = [ts for ts, _ in files]

panel_data = {i: [] for i in range(len(panels))}

for ts, fname in files:
    with h5py.File(os.path.join(field_dir, fname), "r") as f:
        fd = f["fielddata"]

        for i, (group, dataset, _, _) in enumerate(panels):
            if group not in fd or dataset not in fd[group]:
                panel_data[i].append(None)
                continue

            data = fd[group][dataset][:]
            x = data[:, 0]
            value = data[:, 1]

            if x_positions is None:
                x_positions = x

            panel_data[i].append(value)

# -------------------------------
# Plot styling
# -------------------------------
figsize = np.array([180, 240])  # mm, tall figure for 4x2 grid
dpi = 300

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=9, serif='Computer Modern Roman')
mp.rc('axes', titlesize=9)
mp.rc('axes', labelsize=9)
mp.rc('xtick', labelsize=8)
mp.rc('ytick', labelsize=8)

fig, axes = plt.subplots(4, 2, figsize=figsize / 25.4,
                          constrained_layout=True, dpi=dpi)
axes = axes.flatten()

extent = [times[0] * DT_coeff, times[-1] * DT_coeff,
          x_positions[0], x_positions[-1]] if x_positions is not None else None

for i, (group, dataset, title, cbar_label) in enumerate(panels):
    ax = axes[i]
    values = panel_data[i]

    if any(v is None for v in values) or x_positions is None:
        ax.set_title(title + " (missing data)")
        ax.axis("off")
        continue

    values = np.array(values).T  # shape: (n_space, n_time)

    img = ax.imshow(
        values,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="plasma"
    )

    ax.set_title(title)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(cbar_label)

out_path = os.path.join(plot_dir, "xt_overview.pdf")
plt.savefig(out_path, dpi=dpi)
print("Saved:", out_path)
plt.show()