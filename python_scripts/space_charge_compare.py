import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

# Usage:
# python compare_field.py <path1> <path2> <ts>

path1 = os.path.abspath(sys.argv[1])
path2 = os.path.abspath(sys.argv[2])
ts = int(sys.argv[3])

plot_dir = os.path.join(path1, "plots_compare")
os.makedirs(plot_dir, exist_ok=True)

# ----------- Load metadata (ONLY from path1) -----------
meta_file1 = os.path.join(path1, "metadata.h5")

with h5py.File(meta_file1, 'r') as f_meta:
    metadata_group = f_meta['/metadata']

    DT_coeff = metadata_group.attrs['DT_coeff']

time = ts * DT_coeff

# ----------- Function to read field data -----------
def read_field(path, ts):
    field_file = os.path.join(path, "field", f"data_{ts}.h5")

    with h5py.File(field_file, "r") as f:
        den_e = f["fielddata"]["den_electron"]["electron"][:]
        den_i = f["fielddata"]["den_ion"]["ion"][:]
        rho_data = f["fielddata"]["rho"]["rho"][:]
        pot_data = f["fielddata"]["pot"]["potential"][:]

    x = rho_data[:, 0]
    den_e = den_e[:, 1]
    den_i = den_i[:, 1]
    rho = rho_data[:, 1]
    pot = pot_data[:, 1]

    return x, den_e, den_i, rho, pot


# ----------- Read both datasets -----------
x1, ne1, ni1, rho1, pot1 = read_field(path1, ts)
x2, ne2, ni2, rho2, pot2 = read_field(path2, ts)

# ----------- Plot comparison -----------

plt.figure(figsize=(10, 6))

# Charge density
plt.plot(x1, rho1, label="rho (run1)", linestyle='-')
plt.plot(x2, rho2, label="rho (run2)", linestyle='--')

# Electron density
#plt.plot(x1, ne1, label="n_e (run1)")
#plt.plot(x2, ne2, label="n_e (run2)", linestyle='--')

# Ion density
#plt.plot(x1, ni1, label="n_i (run1)")
#plt.plot(x2, ni2, label="n_i (run2)", linestyle='--')

# Optional: potential
# plt.plot(x1, pot1, label="phi (run1)")
# plt.plot(x2, pot2, label="phi (run2)", linestyle='--')

plt.xlabel(r"$x$")
plt.ylabel("Density / Charge")
plt.title(f"Comparison at time = {time:.3f}")

plt.legend()
plt.grid()

# Save
out_path = os.path.join(plot_dir, f"compare_{ts}.png")
plt.savefig(out_path, dpi=300)
plt.show()