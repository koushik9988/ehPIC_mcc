import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import os
import sys
import matplotlib as mp



if len(sys.argv) < 4:
    print("Usage: python plot_expansion_selfsimilar.py <output_folder> <species> <ts1> <ts2> ...")
    sys.exit(1)

path    = os.path.abspath(sys.argv[1])
species = sys.argv[2]
ts_list = [int(t) for t in sys.argv[3:]]

meta_file = os.path.join(path, "metadata.h5")
plot_dir  = os.path.join(path, "plots")
os.makedirs(plot_dir, exist_ok=True)

# ── Metadata ──────────────────────────────────────────────────────────────────
with h5py.File(meta_file, "r") as f:
    DT_coeff = f["/metadata"].attrs["DT_coeff"]
    wpe      = f["/metadata"].attrs["wpe"]
    wpi      = f["/metadata"].attrs["wpi"]
    LDe      = f["/metadata"].attrs["LDe"]

mass_ratio = 1.0 / 1836.0        # me/mi (hydrogen)
vcs_code   = np.sqrt(mass_ratio)  # cs / v_the  (Boltzmann electrons, isothermal)

# Vacuum interface: ions placed 0 → 64 LDe
x_0_code = 256#128.0

# ── Color map: one color per timestep ────────────────────────────────────────
cmap    = cm.get_cmap("plasma", len(ts_list))
colors  = [cmap(i) for i in range(len(ts_list))]

# ── Mora analytical reference (drawn once) ───────────────────────────────────
xi_mora = np.linspace(-1.0, 6.0, 600)
n_mora  = np.exp(-(1.0 + xi_mora))   # Mora density  (normalised)
v_mora  = 1.0 + xi_mora              # Mora velocity (in units of cs)


figsize = np.array([150,150/1.618])#Figure size in mm (FOR SINGLE FIGURE)
dpi = 300                        #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

fig, (ax_n, ax_v) = plt.subplots(2, 1, figsize = figsize / 25.4, constrained_layout=True, dpi=dpi, sharex=True)
fig.subplots_adjust(hspace=0.08)


wpi_times_plotted = []

for idx, ts in enumerate(ts_list):
    field_file = os.path.join(path, "field", f"data_{ts}.h5")

    if not os.path.exists(field_file):
        print(f"[WARN] Missing file: {field_file} — skipping ts={ts}")
        continue

    with h5py.File(field_file, "r") as f:
        data_vel = f["fielddata"][f"vel_{species}"][species][:]
        data_den = f["fielddata"][f"den_{species}"][species][:]

    x_code   = data_vel[:, 0]
    vel_code = data_vel[:, 1]   # v / v_the
    den_code = data_den[:, 1]   # n / n_ref

    bulk_end = max(len(den_code) // 3, 1)
    n_0      = np.mean(den_code[:bulk_end])

    # Normalised quantities
    n_norm = den_code / n_0
    v_norm = vel_code / vcs_code   # v / cs

    # Self-similar coordinate
    time_code = ts * DT_coeff      # t * wpe
    wpit      = time_code * wpi / wpe
    wpi_times_plotted.append(wpit)

    xi = (x_code - x_0_code) / (vcs_code * time_code)

    label = rf"$\omega_{{pi}}t$ = {wpit:.1f}"
    col   = colors[idx]

    ax_n.plot(xi, n_norm, color=col, lw=1.8, alpha=0.85, label=label)
    ax_v.plot(xi, v_norm, color=col, lw=1.8, alpha=0.85, label=label)

ax_n.plot(xi_mora, n_mora, 'k--', lw=2.0, label=r"$n_i/n_0 = e^{-(1+\xi)}$")
ax_v.plot(xi_mora, v_mora, 'k--', lw=2.0, label=r"$v_i/c_s = 1 + \xi$")

#density
ax_n.set_ylabel(r"$n_i / n_0$")
ax_n.set_ylim(-0.05, 1.40)
ax_n.set_xlim(-3.0, 6.0)
#ax_n.axvline(-1.0, color='gray', lw=0.8, ls=':', alpha=0.7, label=r"$\xi = -1$ (sonic pt.)")
ax_n.legend(loc='upper right', fontsize=8.5, ncol=2, framealpha=0.9)
ax_n.grid(True, linestyle=':', alpha=0.5)

#vel
ax_v.set_xlabel(r"$\xi = (x - x_0)\,/\,(c_s\,t)$")
ax_v.set_ylabel(r"$v_i / c_s$")
ax_v.set_ylim(-0.5, 8.0)
ax_v.axvline(-1.0, color='gray', lw=0.8, ls=':', alpha=0.7)
ax_v.legend(loc='upper left', fontsize=8.5, ncol=2, framealpha=0.9)
ax_v.grid(True, linestyle=':', alpha=0.5)

ts_tag   = "_".join(str(t) for t in ts_list)
out_path = os.path.join(plot_dir, f"selfsimilar_{species}_ts{ts_tag}.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()