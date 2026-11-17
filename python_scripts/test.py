import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

if len(sys.argv) < 4:
    print("Usage: python plot_expansion_benchmark.py <output_folder> <ts> <species>")
    sys.exit(1)

path = os.path.abspath(sys.argv[1])
ts   = int(sys.argv[2])
species = sys.argv[3]

field_file = os.path.join(path, "field", f"data_{ts}.h5")
meta_file  = os.path.join(path, "metadata.h5")
plot_dir   = os.path.join(path, "plots")
os.makedirs(plot_dir, exist_ok=True)

# ── Metadata ──────────────────────────────────────────────────────────────────
with h5py.File(meta_file, "r") as f:
    DT_coeff = f["/metadata"].attrs["DT_coeff"]
    wpe      = f["/metadata"].attrs["wpe"]
    wpi      = f["/metadata"].attrs["wpi"]
    LDe      = f["/metadata"].attrs["LDe"]

# ── Normalized units ──────────────────────────────────────────────────────────
mass_ratio = 1.0 / 1836.0          # me/mi  (hydrogen)
vcs_code   = np.sqrt(mass_ratio)   # cs / v_the  (isothermal electrons, gamma=1)
time_code  = ts * DT_coeff         # t * wpe

# ── Field data ────────────────────────────────────────────────────────────────
with h5py.File(field_file, "r") as f:
    data_vel = f["fielddata"][f"vel_{species}"][species][:]
    data_den = f["fielddata"][f"den_{species}"][species][:]

x_code   = data_vel[:, 0]
vel_code = data_vel[:, 1]   # v / v_the  (code units)
den_code = data_den[:, 1]   # n / n_ref  (code units)

# ── Benchmark config ──────────────────────────────────────────────────────────
# Ions placed 0 → 64 LDe  →  vacuum interface at right edge
x_0_code = 256#128#64.0

# Robust bulk density: mean over left 10 % of domain
bulk_end  = max(len(den_code) // 10, 1)
n_0_code  = np.mean(den_code[:bulk_end])
print(f"n_0_code (bulk average) = {n_0_code:.4f}")

# Self-similar coordinate
if time_code > 0:
    xi = (x_code - x_0_code) / (vcs_code * time_code)
else:
    xi = x_code - x_0_code

# Normalize for plotting
n_normalized = den_code / n_0_code
v_normalized = vel_code / vcs_code   # now in units of cs

# ── Mora analytical solution ──────────────────────────────────────────────────
# Valid for xi >= -1  (rarefaction region)
xi_mora   = np.linspace(-1.0, max(xi.max(), 6.0), 600)
n_mora    = np.exp(-(1.0 + xi_mora))   # = exp(-1 - xi)
v_mora    = 1.0 + xi_mora              # in units of cs

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

# Density (left axis)
c_blue = 'tab:blue'
ax1.set_xlabel(r"Self-Similar Variable $\xi = (x - x_0)\,/\,(c_s\,t)$", fontsize=11)
ax1.set_ylabel(r"Normalized Density $n_i / n_0$", color=c_blue, fontsize=11)
l1, = ax1.plot(xi, n_normalized, color=c_blue,  lw=2,   label="Sim Density")
l2, = ax1.plot(xi_mora, n_mora,  color='navy',  lw=1.5, ls='--',
               label="Mora Density (Theory)", alpha=0.85)
ax1.tick_params(axis='y', labelcolor=c_blue)
ax1.set_ylim(-0.05, 1.15)
ax1.set_xlim(-3.0, 6.0)

# Velocity (right axis)
ax2 = ax1.twinx()
c_org = 'tab:orange'
ax2.set_ylabel(r"Normalized Velocity $v_i / c_s$", color=c_org, fontsize=11)
l3, = ax2.plot(xi, v_normalized, color=c_org,       lw=2,   label="Sim Velocity")
l4, = ax2.plot(xi_mora, v_mora,  color='darkorange', lw=1.5, ls='--',
               label="Mora Velocity (Theory)", alpha=0.85)
ax2.tick_params(axis='y', labelcolor=c_org)

# Legend
ax1.legend([l1, l3, l2, l4],
           [l.get_label() for l in [l1, l3, l2, l4]],
           loc='upper right', fontsize=9)

t_wpi = time_code * wpi / wpe
plt.title(
    f"Expansion Benchmark ({species}) at "
    rf"$t\cdot\omega_{{pe}}$ = {time_code:.2f}  "
    rf"($t\cdot\omega_{{pi}}$ = {t_wpi:.2f})",
    fontsize=11
)
ax1.grid(True, linestyle=':', alpha=0.6)

out_path = os.path.join(plot_dir, f"expansion_benchmark_{species}_ts_{ts}.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()