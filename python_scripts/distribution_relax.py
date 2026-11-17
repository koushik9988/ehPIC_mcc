import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

if len(sys.argv) != 4:
    print("Usage: python3 plot_dfdv_time.py <path> <particle_type> <save_flag>")
    sys.exit(1)

base_path = os.path.abspath(sys.argv[1].strip())
particle_type = sys.argv[2].strip()
save_flag = int(sys.argv[3])

phase_dir = os.path.join(base_path, "phase")
meta_file = os.path.join(base_path, "metadata.h5")
plot_path = os.path.join(base_path, "plots_dfdv")

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

if not os.path.exists(meta_file):
    print("metadata.h5 not found")
    sys.exit(1)

if not os.path.isdir(phase_dir):
    print("phase directory not found")
    sys.exit(1)

# ---- Read metadata ----
with h5py.File(meta_file, 'r') as f_meta:
    metadata_group = f_meta['/metadata']
    NC = metadata_group.attrs['NC']
    NUM_TS = metadata_group.attrs['NUM_TS']
    DT_coeff = metadata_group.attrs['DT_coeff']
    wpe = metadata_group.attrs['wpe']
    wpi = metadata_group.attrs['wpi']

mFactor = wpi / wpe

# ---- Collect phase files ----
phase_files = []
for fname in os.listdir(phase_dir):
    if fname.startswith("phase_") and fname.endswith(".h5"):
        ts_num = int(fname[6:-3])
        phase_files.append((ts_num, fname))

phase_files.sort()
print(f"Found {len(phase_files)} phase files")

# ---- Histogram settings ----
NBINS = 100

# ----compute avg |df/dv| ----
time   = []
avg_dfdv   = []

for ts_num, fname in phase_files:

    path = os.path.join(phase_dir, fname)
    with h5py.File(path, 'r') as pf:
        data = pf[f"{particle_type}/phase"][:]

    vx = data[:, 1]

    f_v, bin_edges = np.histogram(vx, bins=NBINS, density=True)
    v = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dv = v[1] - v[0]

    dfdv = np.gradient(f_v, dv)

    t = ts_num * DT_coeff * mFactor
    time.append(t)
    avg_dfdv.append(np.mean(np.abs(dfdv)))

time = np.array(time)
avg_dfdv = np.array(avg_dfdv)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(time, avg_dfdv, lw=1.5, color='steelblue')

ax.set_xlabel(r"$\omega t$")
ax.set_ylabel(r"$\langle |df/dv| \rangle$")
ax.set_title(f"Mean $|df/dv|$ vs Time  [{particle_type}]")
ax.grid(True, alpha=0.4)

plt.tight_layout()

if save_flag == 1:
    out = os.path.join(plot_path, "avg_dfdv_vs_time.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

plt.show()