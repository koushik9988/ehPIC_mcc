import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib as mp

path1 = sys.argv[1]
path2 = sys.argv[2]
path3 = sys.argv[3]

wpe = 1.77163E7
wpi = 4.16329E06
mfactor = 0.23585166451892622  # wpi/wpe

def load_energy(path):
    fname = os.path.join(path, "logs", "energy_log.csv")
    data = np.loadtxt(fname, delimiter="\t", comments="#")

    time = data[:, 1] * mfactor
    KE = data[:, 2]
    PE = data[:, 3]
    TE = data[:, 4]

    return time, KE, PE, TE

time1, KE1, PE1, TE1 = load_energy(path1)
time2, KE2, PE2, TE2 = load_energy(path2)
time3, KE3, PE3, TE3 = load_energy(path3)

figsize = np.array([150, 150/1.618])
dpi = 300

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10)
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

plt.figure(figsize=figsize/25.4, constrained_layout=True, dpi=dpi)

plt.plot(time1, PE1, lw=1.5, label=r'$ppc = 200$')
plt.plot(time2, PE2, lw=1.5, label=r'$ppc = 600$')
plt.plot(time3, PE3, lw=1.5, label=r'$ppc = 1000$')

plt.xlabel(r'$\omega_{pe} t$')
plt.ylabel('PE')

# Uncomment if needed
plt.yscale('log')
plt.ylim([6.5e-5, 0.03])
plt.xlim([-10, 400])
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("PE_comparison_3cases.pdf", dpi=dpi)
plt.show()