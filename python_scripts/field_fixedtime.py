import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

# Usage:
# python plot_field.py <output_folder> <field_type> <ts> [species]

path = os.path.abspath(sys.argv[1])
field_type = sys.argv[2]
ts = int(sys.argv[3])
species = sys.argv[4] if len(sys.argv) > 4 else None

field_file = os.path.join(path, "field", f"data_{ts}.h5")
meta_file = os.path.join(path, "metadata.h5")
plot_dir = os.path.join(path, "plots")

os.makedirs(plot_dir, exist_ok=True)


# Map field_type → group, dataset
def get_group_dataset(ftype, species):

    if ftype == "efield":
        return "efield", "efield"

    if ftype == "pot":
        return "pot", "pot"

    if ftype == "rho":
        return "rho", "rho"

    if ftype == "den":
        return f"den_{species}", species

    if ftype == "vel":
        return f"vel_{species}", species

    if ftype == "heating":
        return "heating", species

    if ftype == "current":
        return "current", species

    if ftype == "collrate":
        return f"collrate_{species}", species

    raise ValueError("Unknown field type")


group_name, dataset_name = get_group_dataset(field_type, species)


# Read metadata (optional, only for labeling time)
with h5py.File(meta_file, "r") as f:
    DT_coeff = f["/metadata"].attrs["DT_coeff"]

time = ts * DT_coeff


# Read field file
with h5py.File(field_file, "r") as f:

    data = f["fielddata"][group_name][dataset_name][:]

    x = data[:, 0]
    value = data[:, 1]


# Plot
plt.figure(figsize=(8, 5))

plt.plot(x, value)

plt.xlabel("x")
plt.ylabel(field_type)
plt.title(f"{field_type} at time = {time:.3f}")
plt.grid()


# Save
name = field_type
if species:
    name += f"_{species}"

out_path = os.path.join(plot_dir, f"{name}_ts_{ts}.png")

plt.savefig(out_path, dpi=300)
plt.show()