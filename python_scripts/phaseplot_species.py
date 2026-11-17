import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.constants import value as constants
from os.path import join as pjoin
import os
import sys
import matplotlib.animation as animation

#----- CLI Argumnets ------
if len(sys.argv) != 4:
    print("Usage: python3 script.py <path> <particle_type> <save_flag>")
    sys.exit(1)

base_path = os.path.abspath(sys.argv[1].strip())
particle_type = sys.argv[2].strip()
save_flag = int(sys.argv[3])  # 1 = save plots, 0 = no save


phase_dir = os.path.join(base_path, "phase")
meta_file = os.path.join(base_path, "metadata.h5")
plot_path = os.path.join(base_path, "plots")

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

print("\nBase path :", base_path)
print("Species   :", particle_type)
print("Phase dir :", phase_dir)


if not os.path.exists(meta_file):
    print("metadata.h5 not found")
    sys.exit(1)

if not os.path.isdir(phase_dir):
    print("phase directory not found")
    sys.exit(1)

# ---- Read metadata ----
f_meta = h5py.File(meta_file, 'r')
metadata_group = f_meta['/metadata']

# --- Constants and attributes ---
NC = metadata_group.attrs['NC']
NUM_TS = metadata_group.attrs['NUM_TS']
DT_coeff = metadata_group.attrs['DT_coeff']
wpe = metadata_group.attrs['wpe']
wpi = metadata_group.attrs['wpi']
mFactor = wpi / wpe


phase_files = []

for fname in os.listdir(phase_dir):
    if fname.startswith("phase_") and fname.endswith(".h5"):
        ts_num = int(fname[6:-3])
        phase_files.append((ts_num, fname))

phase_files.sort()

print(f"Found {len(phase_files)} phase files")


DATA_TS_PHASE = len(phase_files)

# Create time array
ts = np.array([step * DT_coeff * mFactor for step, _ in phase_files])

# --- Plotting ---
#fig, (ax, ax_phase1) = plt.subplots(2, 1)
fig, ax_phase1 = plt.subplots()

def animate(i):
    ts_num, fname = phase_files[i]
    
    # Read phase space data from individual file
    phase_file_path = os.path.join(phase_dir, fname)
    phase_f = h5py.File(phase_file_path, 'r')
    
    data_phase = phase_f[f"{particle_type}/phase"][:]
    
    datax = data_phase[:, 0]
    datavx = data_phase[:, 1]

    ax_phase1.clear()
    
    # Plot phase space
    ax_phase1.scatter(datax, datavx, s=4, marker='o', color='blue')

    ax_phase1.set_xlabel('$x$')
    ax_phase1.set_ylabel('$v$')
    ax_phase1.set_xlim([0, NC])
    #ax_phase1.set_ylim([-3.5, 1.2])

    title_text = f"time : {ts[i]:.2f}, TS: {ts_num}"
    ax_phase1.set_title(title_text)

    phase_f.close()

    # Save if enabled
    if save_flag == 1 and i % 10 == 0:
        fig.savefig(os.path.join(plot_path, f"frame_{i:04d}.png"), dpi=150)

    return ax_phase1

# --- Keyboard controls ---
def on_key(event):
    if event.key == 'enter':
        on_key.frame = min(on_key.frame + 1, DATA_TS_PHASE - 1)
        animate(on_key.frame)
        plt.draw()
    elif event.key == 'backspace':
        on_key.frame = max(on_key.frame - 1, 0)
        animate(on_key.frame)
        plt.draw()
    elif event.key == 'tab':
        on_key.ani = animation.FuncAnimation(
            fig, animate, frames=DATA_TS_PHASE,
            blit=False, interval=100, repeat=False
        )
        plt.draw()

on_key.frame = 0
on_key.ani = None
fig.canvas.mpl_connect('key_press_event', on_key)

animate(on_key.frame)
plt.show()