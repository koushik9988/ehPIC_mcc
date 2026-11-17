import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys

# Usage:
# python plot_field.py <output_folder> <ts> <species>

path = os.path.abspath(sys.argv[1])
ts = int(sys.argv[2])
#species = sys.argv[3]

field_file = os.path.join(path, "field", f"data_{ts}.h5")
meta_file = os.path.join(path, "metadata.h5")
plot_dir = os.path.join(path, "plots")

os.makedirs(plot_dir, exist_ok=True)

with h5py.File(meta_file, 'r') as f_meta:

    metadata_group = f_meta['/metadata']

    NC = metadata_group.attrs['NC']
    NUM_TS = metadata_group.attrs['NUM_TS']

    write_interval = metadata_group.attrs['write_int']
    write_interval_phase = metadata_group.attrs['write_int_phase']

    DT_coeff = metadata_group.attrs['DT_coeff']

    wpe = metadata_group.attrs['wpe']
    wpi = metadata_group.attrs['wpi']

    LDe = metadata_group.attrs['LDe']
    LDi = metadata_group.attrs['LDi']

    normscheme = metadata_group.attrs['norm_scheme']
    n0 = metadata_group.attrs['density']



time = ts * DT_coeff

for ts in range(0, NUM_TS, write_interval):

    field_file = os.path.join(path, "field", f"data_{ts}.h5")

    # Skip if file doesn't exist
    if not os.path.exists(field_file):
        continue

    # Read field data
    with h5py.File(field_file, "r") as f:
        den_e = f["fielddata"]["den_electron"]["electron"][:]
        den_i = f["fielddata"]["den_ion"]["ion"][:]
        #den_n = f["fielddata"][f"den_negion"]["negion"][:]
        rho_data = f["fielddata"]["rho"]["rho"][:]
        pot_data = f["fielddata"]["pot"]["potential"][:]
        #pot_data = f["fielddata"]["efield"]["efield"][:]
        #data_vel = f["fielddata"][f"vel_electron"][species][:]

    # Extract columns
    x = rho_data[:, 0]
    #den_e = den_e[:, 1]
    #den_i = den_i[:, 1]
    #den_n = den_n[:,1]
    #rho = rho_data[:, 1]
    pot = pot_data[:,1]

    # Plot
    plt.figure(figsize=(8, 5))
    #plt.plot(x, pot, label=r"$\phi$")
    plt.plot(x, pot, label=r"$\phi$")

    #plt.plot(x, den_e, label=r"$n_e$")
    #plt.plot(x, den_i, label=r"$n_i$")
    #plt.plot(x, den_n, label = r"$n_n$")

    plt.xlabel(r"$x$")
    plt.ylabel(r"$\phi(x)$")
    plt.title(f"$\omega_{{pe}}t $= {ts*DT_coeff}")
    plt.legend()
    plt.grid()

    # Save
    out_path = os.path.join(plot_dir, f"space_charge_{ts}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

print("All plots saved.")


#"""

#plt.show()