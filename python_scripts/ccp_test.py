import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

path = os.path.abspath(sys.argv[1])

meta_file = os.path.join(path, "metadata.h5")

with h5py.File(meta_file, "r") as f:
    metadata = f["metadata"]
    DT_coeff = metadata.attrs["DT_coeff"]

# -------- Time interval --------
ts_start = 400000
ts_end   = 500000
ts_step  = 50
# -------------------------------

ne_sum = None
ni_sum = None
nn_sum = None
count = 0

for ts in range(ts_start, ts_end + 1, ts_step):

    field_file = os.path.join(path, "field", f"data_{ts}.h5")

    if not os.path.exists(field_file):
        continue

    with h5py.File(field_file, "r") as f:

        den_e = f["fielddata"]["den_electron"]["electron"][:]
        den_i = f["fielddata"]["den_ion"]["ion"][:]
        den_n = f["fielddata"]["vel_ion"]["ion"][:]#f["fielddata"]["pot"]["potential"][:]#f["fielddata"]["den_negion"]["negion"][:]

    x = den_e[:,0]
    ne = den_e[:,1]
    ni = den_i[:,1]
    nn = den_n[:,1]

    if ne_sum is None:
        ne_sum = np.zeros_like(ne)
        ni_sum = np.zeros_like(ni)
        nn_sum = np.zeros_like(nn)

    ne_sum += ne
    ni_sum += ni
    nn_sum += nn
    count += 1

# Compute averages
ne_avg = ne_sum / count
ni_avg = ni_sum / count
nn_avg = nn_sum / count

print(f"Averaged over {count} snapshots")
print(f"Time interval = {ts_start*DT_coeff:.3f} to {ts_end*DT_coeff:.3f}")

# Plot
plt.figure(figsize=(8,5))
#plt.plot(x, ne_avg, label=r"$\langle n_e \rangle$", lw=2)
#plt.plot(x, ni_avg, label=r"$\langle n_i \rangle$", lw=2)
#plt.plot(x, nn_avg, label=r"$\langle n_n \rangle$", lw=2)
plt.plot(x, nn_avg, label=r"$vel$", lw=2)

plt.xlabel(r"$x$")
plt.ylabel("Time-averaged quantity")
#plt.title("Time-averaged density profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()