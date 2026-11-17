import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py

path  = sys.argv[1]

energy_data = os.path.join(path,"logs" ,"collision_freq_log.csv")

energy = np.loadtxt(energy_data, delimiter="\t", comments="#")

meta_file = os.path.join(path, "metadata.h5")
with h5py.File(meta_file, 'r') as f_meta:
    metadata_group = f_meta['/metadata']
    wpi = metadata_group.attrs['wpi']

print(energy.shape)

time_step = energy[:, 1]
colle = energy[:, 2]
colli = energy[:, 3]
#TE = energy[:, 4]
print(energy)


plt.plot(time_step, colle, color = "blue",label = r"$\nu_e$")
plt.plot(time_step, colli, color = "red", label = r"$\nu_i$")
plt.xlabel(r"\omega_{pi} t")
plt.ylabel(r"$\nu$")
#plt.plot(time_step, TE, color = "green", label = "TE")

plt.legend()
plt.show()