import pandas as pd
import matplotlib.pyplot as plt
import os
import sys



path  = sys.argv[1]

energy_data = os.path.join(path,"logs" ,"ke_components_log.csv")

# Read file
df = pd.read_csv(energy_data , sep=r"\s+", engine="python")

# Time column
time = df["time"]

# Detect species automatically
species = []
for col in df.columns:
    if col.endswith("_KEx"):
        species.append(col[:-4])  # remove '_KEx'

n_species = len(species)

# Create figure with one axis per species
fig, axes = plt.subplots(
    n_species,
    1,
    figsize=(10, 3*n_species),
    sharex=True
)

# When only one species exists
if n_species == 1:
    axes = [axes]

for ax, sp in zip(axes, species):

    ax.plot(time, df[f"{sp}_KEx"], label="KEx")
    ax.plot(time, df[f"{sp}_KEy"], label="KEy")
    ax.plot(time, df[f"{sp}_KEz"], label="KEz")

    ax.set_title(sp)
    ax.set_ylabel("KE")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time")

plt.tight_layout()
plt.show()