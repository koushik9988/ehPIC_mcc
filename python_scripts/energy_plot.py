import numpy as np
import matplotlib.pyplot as plt
import os
import sys

path  = sys.argv[1]

energy_data = os.path.join(path,"logs" ,"energy_log.csv")

energy = np.loadtxt(energy_data, delimiter="\t", comments="#")

print(energy.shape)

wpe = 1.77163E7
wpi =  4.16329E06
mfactor = 0.23585166451892622#wpi/wpe

time_step = energy[:, 1]*mfactor
KE = energy[:, 2]
PE = energy[:, 3]
TE = energy[:, 4]


plt.plot(time_step, KE, color = "blue",label = "KE")
plt.plot(time_step, PE, color = "red", label = "PE")
plt.plot(time_step, TE, color = "green", label = "TE")
plt.xlabel(r"$\omega_{pi}t$")
plt.ylabel(r"$Energy$")
plt.yscale("log")

plt.legend()
plt.show()
    