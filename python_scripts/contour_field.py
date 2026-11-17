import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import matplotlib as mp

# Usage
# python contour_field.py <output_folder> <field_type> [species]

path = os.path.abspath(sys.argv[1])
field_type = sys.argv[2]
species = sys.argv[3] if len(sys.argv) > 3 else None

field_dir = os.path.join(path, "field")
meta_file = os.path.join(path, "metadata.h5")
plot_dir = os.path.join(path, "plots")

os.makedirs(plot_dir, exist_ok=True)


# Map field_type → group, dataset
def get_group_dataset(ftype, species):

    if ftype == "efield":
        return "efield", "efield"

    if ftype == "pot":
        return "pot", "potential"

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


group_name, dataset_name = get_group_dataset(field_type, species)


# Read metadata
with h5py.File(meta_file, "r") as f:
    meta = f["/metadata"].attrs
    DT_coeff = meta["DT_coeff"]
    NUM_TS = meta["NUM_TS"]
    NC = meta['NC']
    wpe = f["/metadata"].attrs["wpe"]
    LDe = f["/metadata"].attrs["LDe"]
    wpi = f["/metadata"].attrs["wpi"]
    LDi = f["/metadata"].attrs["LDi"]


vthe = LDe*wpe
mass_ratio = 1.0 / 1836.0
vcs = vthe * np.sqrt(mass_ratio)
print(vcs)
velscale = (vthe/vcs)

files = [] 
for fname in os.listdir(field_dir):
    if fname.startswith("data_") and fname.endswith(".h5"):
        ts = int(fname[5:-3])
        #print(ts)
        files.append((ts, fname))
files.sort()

values = []
times = []
x_positions = None


# Load data
for ts, fname in files:

    with h5py.File(os.path.join(field_dir, fname), "r") as f:

        data = f["fielddata"][group_name][dataset_name][:]

        if field_type != "vel":
            velscale = 1
        else:
            velscale = (vthe / vcs)

        x = data[:, 0]
        value = data[:, 1]*velscale
        
       
        if x_positions is None:
            x_positions = x

        values.append(value)
        times.append(ts)


values = np.array(values).T

figsize = np.array([150,150/1.618])  # Figure size in mm
dpi = 300
ppi = np.sqrt(1920**2 + 1200**2) / 24

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

# Plot
fig, ax = plt.subplots(figsize=figsize / 25.4, constrained_layout=True, dpi=dpi)

# Now: x-axis = space, y-axis = time
extent = [x_positions[0], x_positions[-1],
          times[0] * DT_coeff, times[-1] * DT_coeff]

img = ax.imshow(
    values.T,         
    extent=extent,
    origin="lower",
    aspect="auto",
    cmap="plasma"
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\omega_{pi} t$")

cbar = plt.colorbar(img, ax=ax)
cbar.set_label(r"$n_e$")

# Save
name = field_type
if species:
    name += f"_{species}"

out_path = os.path.join(plot_dir, name + "_xt.pdf")

plt.savefig(out_path, dpi=dpi)
plt.show()
