import numpy as np
import h5py
import os
import sys
import glob
import re

# ============================================================
# Usage:
# python3 save_swarm_data.py <datafiles_dir> <species_name>
#
# Saves a 3-column txt file:
#   E/N(Td)   W_sim(m/s)   W_phelps(m/s)
# ============================================================

if len(sys.argv) != 3:
    print("Usage: python3 save_swarm_data.py <datafiles_dir> <species_name>")
    sys.exit(1)

datafiles_dir = os.path.abspath(sys.argv[1])
particle_type = sys.argv[2]

# ============================================================
# Phelps/LXCat reference data (300 K, Boltzmann two-term)
# ============================================================

phelps_EN = np.array([1, 1.5, 2, 3, 5, 7, 10, 20, 30, 50,
                       70, 100, 200, 300, 500])          # Td
phelps_W  = np.array([3010, 3340, 3580, 3930, 5170, 7290,
                       10540, 20050, 28300, 43100, 56600,
                       75800, 134800, 190000, 294000])   # m/s

# ============================================================
# Find all data_swarm_* directories
# ============================================================

pattern  = os.path.join(datafiles_dir, "data_swarm_*")
all_dirs = sorted(glob.glob(pattern))

if len(all_dirs) == 0:
    print(f"ERROR: no data_swarm_* directories found in {datafiles_dir}")
    sys.exit(1)

print(f"Found {len(all_dirs)} swarm directories in {datafiles_dir}")

# ============================================================
# Process each directory
# ============================================================

results = []

for run_dir in all_dirs:

    dirname = os.path.basename(run_dir)

    match = re.search(r'data_swarm_([0-9]+(?:[p.][0-9]+)?)', dirname)
    if not match:
        print(f"  SKIP {dirname}: cannot parse E/N from name")
        continue

    en_str = match.group(1).replace('p', '.')
    EN_Td  = float(en_str)

    field_dir = os.path.join(run_dir, "field")
    meta_file = os.path.join(run_dir, "metadata.h5")

    if not os.path.isfile(meta_file):
        print(f"  SKIP {dirname}: no metadata.h5")
        continue
    if not os.path.isdir(field_dir):
        print(f"  SKIP {dirname}: no field/ directory")
        continue

    try:
        with h5py.File(meta_file, "r") as meta:
            grp      = meta["/metadata"]
            DT_coeff = grp.attrs["DT_coeff"]
            wpe      = grp.attrs["wpe"]
            LDe      = grp.attrs["LDe"]
    except Exception as e:
        print(f"  SKIP {dirname}: metadata read error — {e}")
        continue

    vel_norm = wpe * LDe

    field_files = sorted(
        [(int(f[5:-3]), f)
         for f in os.listdir(field_dir)
         if f.startswith("data_") and f.endswith(".h5")],
        key=lambda x: x[0]
    )

    if len(field_files) == 0:
        print(f"  SKIP {dirname}: no field files")
        continue

    vdrift_ts = []

    for ts_num, fname in field_files:
        fpath = os.path.join(field_dir, fname)
        try:
            with h5py.File(fpath, "r") as f:
                data = f["fielddata"][f"vel_{particle_type}"][particle_type][:]
        except Exception as e:
            print(f"  WARN {dirname} ts={ts_num}: {e}")
            continue

        vdrift_ts.append(np.mean(data[:, 1]) * vel_norm)

    if len(vdrift_ts) == 0:
        print(f"  SKIP {dirname}: no valid snapshots")
        continue

    vdrift_arr = np.array(vdrift_ts)

    N    = len(vdrift_arr)
    i_ss = int(0.7 * N)
    vd_ss = vdrift_arr[i_ss:]

    mean_vd = np.mean(vd_ss)
    std_vd  = np.std(vd_ss)
    sem_vd  = std_vd / np.sqrt(len(vd_ss))

    converged = True
    if mean_vd != 0:
        ts_axis = np.arange(len(vd_ss), dtype=float)
        p = np.polyfit(ts_axis, vd_ss, 1)
        slope_pct = abs(p[0]) * len(vd_ss) / abs(mean_vd) * 100
        if slope_pct > 10:
            print(f"  WARN {dirname}: drift trend {slope_pct:.1f}% — may not be converged")
            converged = False

    results.append({
        "EN_Td"    : EN_Td,
        "mean_vd"  : mean_vd,
        "sem_vd"   : sem_vd,
        "converged": converged,
    })

    status = "OK" if converged else "WARN"
    print(f"  [{status}] E/N={EN_Td:6.1f} Td  |vdrift|={abs(mean_vd):.4e} m/s")

if len(results) == 0:
    print("ERROR: no valid results found.")
    sys.exit(1)

results.sort(key=lambda r: r["EN_Td"])

EN_sim = np.array([r["EN_Td"]        for r in results])
W_sim  = np.array([abs(r["mean_vd"]) for r in results])

# Interpolate Phelps reference at simulation E/N points
W_phelps_interp = np.interp(EN_sim, phelps_EN, phelps_W)

# ============================================================
# Save to txt file
# ============================================================

out_path = "swarm_data.txt"

header = (
    "# Electron swarm drift velocity comparison\n"
    "# Species: " + particle_type + "\n"
    "# Columns:\n"
    "#   1. E/N (Td)\n"
    "#   2. W_sim (m/s)   — PIC-MCC simulation (|mean drift|, steady-state last 30%)\n"
    "#   3. W_phelps (m/s) — Phelps/LXCat reference (Boltzmann two-term, 300 K)\n"
    "#\n"
    f"# {'E/N_Td':>12}  {'W_sim_m/s':>15}  {'W_phelps_m/s':>15}\n"
)

with open(out_path, "w") as fout:
    fout.write(header)
    for en, ws, wp in zip(EN_sim, W_sim, W_phelps_interp):
        fout.write(f"  {en:>12.4f}  {ws:>15.6e}  {wp:>15.6e}\n")

print(f"\nSaved: {out_path}")
print(f"Rows written: {len(EN_sim)}")

# ── preview ────────────────────────────────────────────────
print("\n" + "="*55)
print("  OUTPUT PREVIEW")
print("="*55)
print(f"  {'E/N (Td)':>12}  {'W_sim (m/s)':>15}  {'W_phelps (m/s)':>15}")
print("-"*55)
for en, ws, wp in zip(EN_sim, W_sim, W_phelps_interp):
    print(f"  {en:>12.4f}  {ws:>15.6e}  {wp:>15.6e}")
print("="*55)