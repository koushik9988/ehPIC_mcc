import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib as mp

path1  = sys.argv[1]
path2  = sys.argv[2]

energy_data_1 = os.path.join(path1,"logs" ,"energy_log.csv")
energy_data_2 = os.path.join(path2,"logs" ,"energy_log.csv")

energy_1 = np.loadtxt(energy_data_1, delimiter="\t", comments="#")
energy_2 = np.loadtxt(energy_data_2, delimiter="\t", comments="#")  

#print(energy_1.shape)
#print(energy_2.shape)

wpe = 1.77163E7
wpi =  4.16329E06
mfactor = 0.23585166451892622#wpi/wpe

time1 = energy_1[:, 1]*mfactor
KE11 = energy_1[:, 2]
PE21 = energy_1[:, 3]
TE31 = energy_1[:, 4]

time2 = energy_2[:, 1]*mfactor
KE12 = energy_2[:, 2]
PE22 = energy_2[:, 3]
TE32 = energy_2[:, 4]

figsize = np.array([150,150/1.618])#Figure size in mm (FOR SINGLE FIGURE)
dpi = 300                        #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)


plt.figure(figsize= figsize / 25.4, constrained_layout=True, dpi=dpi)
#plt.plot(time1, d11, label = "CIC")
#plt.plot(time1, d21, label = "CIC")
plt.plot(time1, PE21, color = "blue", label = "Boltzman electron")

#plt.plot(time2, d12, label = "TSC")
#plt.plot(time2, d22, label = "TSC")
plt.plot(time2, PE22, color = "red", label = "kinetic electron")
plt.xlabel(r"$\omega_{pe}t$")
plt.ylabel(r"$PE$")
plt.ylim([0.00018, 0.03])

plt.yscale("log")
plt.legend()
plt.savefig("PE_comparison_hybrid_vs_kinetic.pdf", dpi=dpi)


plt.show()
