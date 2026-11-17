import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_energy_data(path):
    energy_file = os.path.join(path, "logs", "energy_log.tsv")
    
    if not os.path.exists(energy_file):
        print(f"Warning: {energy_file} not found")
        return None, None
    
    energy = np.loadtxt(energy_file, delimiter="\t", comments="#")
    
    time = energy[:, 1]
    total_energy = energy[:, 4]
    
    return time, total_energy


def main():

    if len(sys.argv) < 2:
        print("Usage: python plot_energy.py path1 path2 ...")
        sys.exit(1)

    paths = sys.argv[1:]
    colors = plt.cm.tab10.colors

    for i, path in enumerate(paths):
        time, TE = load_energy_data(path)
        label = os.path.basename(os.path.normpath(path))
        color = colors[i % len(colors)]
        plt.plot(time, TE, label=label, color=color)

    plt.xlabel(r"$\omega_{pe}t$")
    plt.ylabel("Total Energy")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

main()
