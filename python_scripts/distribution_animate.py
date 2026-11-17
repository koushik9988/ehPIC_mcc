import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import matplotlib.animation as animation

if len(sys.argv) != 4:
    print("Usage: python3 script.py <path> <particle_type> <save_flag>")
    sys.exit(1)

base_path = os.path.abspath(sys.argv[1].strip())
particle_type = sys.argv[2].strip()
save_flag = int(sys.argv[3])

phase_dir = os.path.join(base_path, "phase")
meta_file = os.path.join(base_path, "metadata.h5")
plot_path = os.path.join(base_path, "plots_fv")

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

print("\nBase path :", base_path)
print("Species   :", particle_type)
print("Phase dir :", phase_dir)

# ---- Check files ----
if not os.path.exists(meta_file):
    print("metadata.h5 not found")
    sys.exit(1)

if not os.path.isdir(phase_dir):
    print("phase directory not found")
    sys.exit(1)

# ---- Read metadata ----
with h5py.File(meta_file, 'r') as f_meta:
    metadata_group = f_meta['/metadata']
    NC = metadata_group.attrs['NC']
    NUM_TS = metadata_group.attrs['NUM_TS']
    DT_coeff = metadata_group.attrs['DT_coeff']
    wpe = metadata_group.attrs['wpe']
    wpi = metadata_group.attrs['wpi']

mFactor = wpi / wpe

# ---- Collect phase files ----
phase_files = []
for fname in os.listdir(phase_dir):
    if fname.startswith("phase_") and fname.endswith(".h5"):
        ts_num = int(fname[6:-3])
        phase_files.append((ts_num, fname))

phase_files.sort()

print(f"Found {len(phase_files)} phase files")

DATA_TS_PHASE = len(phase_files)

# Time array
ts = np.array([step * DT_coeff * mFactor for step, _ in phase_files])

# ---- Histogram settings ----
NBINS = 100

# ---- Plot ----
fig, ax = plt.subplots()

def animate(i):

    ts_num, fname = phase_files[i]

    phase_file_path = os.path.join(phase_dir, fname)

    with h5py.File(phase_file_path, 'r') as phase_f:
        data_phase = phase_f[f"{particle_type}/phase"][:]

    vx = data_phase[:, 1]

    # Compute histogram
    f_v, bin_edges = np.histogram(vx, bins=NBINS, density=True)

    # Compute bin centers
    v = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax.clear()

    ax.plot(v, f_v)
    #ax.set_xlim(-230,230)

    ax.set_xlabel("v")
    ax.set_ylabel("f(v)")
    ax.set_title(f"time: {ts[i]:.2f}, TS: {ts_num}")

    ax.grid(True)

    # Save frame
    if save_flag == 1 and i % 10 == 0:
        fig.savefig(os.path.join(plot_path, f"fv_{i:04d}.png"), dpi=150)

    return ax

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
            fig,
            animate,
            frames=DATA_TS_PHASE,
            interval=100,
            repeat=False
        )
        plt.draw()


on_key.frame = 0
on_key.ani = None

fig.canvas.mpl_connect('key_press_event', on_key)

animate(on_key.frame)

plt.show()