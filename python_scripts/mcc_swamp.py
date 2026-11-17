import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

# Usage: python3 swarm_from_vel.py <path> <species>
# e.g.   python3 swarm_from_vel.py ../datafiles/data_swarm_10 electron

path    = os.path.abspath(sys.argv[1])
species = sys.argv[2]

field_dir = os.path.join(path, "field")
meta_file = os.path.join(path, "metadata.h5")

# read metadata
with h5py.File(meta_file, "r") as f:
    DT_coeff = f["/metadata"].attrs["DT_coeff"]
    wpe      = f["/metadata"].attrs["wpe"]
    LDe      = f["/metadata"].attrs["LDe"]

vel_norm = wpe * LDe   # m/s

# load all field files
files = sorted(
    [(int(fn[5:-3]), fn)
     for fn in os.listdir(field_dir)
     if fn.startswith("data_") and fn.endswith(".h5")],
    key=lambda x: x[0]
)

times  = []
vdrift = []

for ts, fname in files:
    with h5py.File(os.path.join(field_dir, fname), "r") as f:
        data = f["fielddata"][f"vel_{species}"][species][:]
    # data[:,0] = x positions, data[:,1] = mean velocity (normalised)
    # spatial mean over all grid points = drift velocity
    vd = np.mean(data[:, 1]) * vel_norm
    times.append(ts * DT_coeff)
    vdrift.append(vd)

times  = np.array(times)
vdrift = np.array(vdrift)

# last 30% average
i_ss    = int(0.7 * len(vdrift))
mean_vd = np.mean(vdrift[i_ss:])

print(f"vel_norm = {vel_norm:.4e} m/s")
print(f"|vdrift| = {abs(mean_vd):.4e} m/s")

# plot
plt.figure(figsize=(8, 4))
plt.plot(times, np.abs(vdrift), lw=1.5, color='steelblue', label='|vdrift|')
plt.axvspan(times[i_ss], times[-1], alpha=0.12, color='green', label='Last 30%')
plt.axhline(abs(mean_vd), color='black', lw=1.5, linestyle='--',
            label=f'Mean = {abs(mean_vd):.3e} m/s')
plt.xlabel("Time")
plt.ylabel("Drift velocity (m/s)")
plt.title(f"Swarm drift velocity — {species}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("swarm_vdrift_vel.png", dpi=150)
plt.show()