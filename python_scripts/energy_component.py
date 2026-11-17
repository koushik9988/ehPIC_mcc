import numpy as np
import matplotlib.pyplot as plt
import os
import sys

path = sys.argv[1]

energy_data = os.path.join(path, "logs", "ke_components_log.csv")
data = np.loadtxt(energy_data, delimiter="\t", comments="#")

time       = data[:, 1]
ele_KEx    = data[:, 2]    # electron KEx  (was ion at col 5)
ele_KEy    = data[:, 3]    # electron KEy
ele_KEz    = data[:, 4]    # electron KEz

# normalization
vel_norm   = 1#132621#132621.0      # m/s  (your vel_norm)
mass_e     = 9.1e-31       # kg

# --------------------------------------------------
# YOUR CODE HAD A BUG: you divided by mass
# correct formula from your Compute_KE:
#   ke(0) = 0.5 * sum(vx^2) * vel_norm^2 / Th
# but for drift we just need mean vx, not from KE
#
# From KE:  KE = 0.5 * m * <vx^2>  (per particle, in Joules)
# so:       <vx^2> = 2 * KE_joules / mass
#           v_drift = sqrt(<vx^2>)
#
# But your KE is NORMALIZED by Th = kT*spwt*N
# so first un-normalize, then get velocity
# --------------------------------------------------

# your normalization: Th = temp(eV) * QE * spwt * numparticle
temp_eV    = 0.1           # from your config
QE         = 1.6e-19
spwt       = 9.44111e+07         # set your actual spwt
num        = 10000         # species_0.num from config
Th         = temp_eV * QE * spwt * num    # normalization factor in Joules

# un-normalize KE back to Joules
KEx_J = ele_KEx * Th
KEy_J = ele_KEy * Th
KEz_J = ele_KEz * Th

# get drift velocity in m/s
# KE = 0.5 * m * N * <vx^2>  → <vx^2> = 2*KE / (m*N)
vx_drift = np.sqrt(2.0 * KEx_J / (mass_e * num))
vy_drift = np.sqrt(2.0 * KEy_J / (mass_e * num))
vz_drift = np.sqrt(2.0 * KEz_J / (mass_e * num))
v_total  = np.sqrt(vx_drift**2 + vy_drift**2 + vz_drift**2)

# steady state average over last 30%
start     = int(0.7 * len(time))
vx_steady = np.mean(vx_drift[start:])
vy_steady = np.mean(vy_drift[start:])
vz_steady = np.mean(vz_drift[start:])
vt_steady = np.mean(v_total[start:])

print(f"Steady state electron drift:")
print(f"  vx = {vx_steady:.4e} m/s")
print(f"  vy = {vy_steady:.4e} m/s")
print(f"  vz = {vz_steady:.4e} m/s")
print(f"  |v| = {vt_steady:.4e} m/s")

# reference values at your E/N
print(f"\nReference (Phelps) at E/N = 100 Td:")
print(f"  vx = 3.2e+04 m/s")
print(f"  mean energy = 2.20 eV")

# plot drift vs time
plt.figure(figsize=(10, 4))
plt.plot(time, vx_drift, label='vx drift')
plt.axvline(time[start], color='r', linestyle='--', label='averaging start')
plt.axhline(3.4e3, color='k', linestyle='--', label='Phelps ref (10Td)')
plt.xlabel("Time (normalized)")
plt.ylabel("Electron drift vx (m/s)")
plt.title("Electron Swarm - Drift Velocity Convergence")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path, "swarm_drift.png"))
plt.show()