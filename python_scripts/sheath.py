import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

# Usage:
# python plot_field.py <output_folder> <ts> <species>

path = os.path.abspath(sys.argv[1])
ts = int(sys.argv[2])
species = sys.argv[3]

field_file = os.path.join(path, "field", f"data_{ts}.h5")
meta_file = os.path.join(path, "metadata.h5")
plot_dir = os.path.join(path, "plots")

os.makedirs(plot_dir, exist_ok=True)

# Read metadata (for time label)
with h5py.File(meta_file, "r") as f:
    DT_coeff = f["/metadata"].attrs["DT_coeff"]
    wpe = f["/metadata"].attrs["wpe"]
    LDe = f["/metadata"].attrs["LDe"]
    wpi = f["/metadata"].attrs["wpi"]
    LDi = f["/metadata"].attrs["LDi"]


vthe = LDe*wpe
mass_ratio = 1.0 / 1836.0
vcs = vthe * np.sqrt(mass_ratio)
print(vcs)

time = ts * DT_coeff

# Read field data
with h5py.File(field_file, "r") as f:
    # velocity
    data_vel = f["fielddata"][f"vel_{species}"][species][:]
    # density
    data_den = f["fielddata"][f"den_{species}"][species][:]

    rho = f["fielddata"][f"rho"]["rho"][:]
    pot = f["fielddata"][f"pot"]["potential"][:]

#print(rho)
# Extract columns
x = data_vel[:, 0]
vel = data_vel[:, 1]*(vthe/vcs)
den = data_den[:, 1]
rho = rho[:,1]
pot = pot[:,1]

# Plot both together
plt.figure(figsize=(8, 5))

#plt.plot(x, vel, label=r"$Ion velocity$")
plt.plot(x,pot)
#plt.plot(x, den, label="Density")

plt.xlabel("x")
plt.ylabel(r"$\phi$")
#plt.ylabel(r"$v_{cs}$")
plt.title(f"{species}: Velocity and Density at time = {time:.3f}")
plt.legend()
plt.grid()

# Save
out_path = os.path.join(plot_dir, f"vel_den_{species}_ts_{ts}.png")
plt.savefig(out_path, dpi=300)

plt.show()