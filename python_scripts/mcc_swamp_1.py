import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import glob
import re

# ============================================================
# Usage:
# python3 plot_swarm_EN.py <datafiles_dir> <species_name>
#
# Example:
# python3 plot_swarm_EN.py ../datafiles electron
#
# Automatically scans for all data_swarm_* directories,
# extracts E/N from folder name, computes steady-state
# drift velocity (last 30%), and plots W vs E/N.
# ============================================================

if len(sys.argv) != 3:
    print("Usage: python3 plot_swarm_EN.py <datafiles_dir> <species_name>")
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

pattern = os.path.join(datafiles_dir, "data_swarm_*")
all_dirs = sorted(glob.glob(pattern))

if len(all_dirs) == 0:
    print(f"ERROR: no data_swarm_* directories found in {datafiles_dir}")
    sys.exit(1)

print(f"Found {len(all_dirs)} swarm directories in {datafiles_dir}")

# ============================================================
# Process each directory
# ============================================================

results = []   # list of (EN_Td, mean_vd, std_vd, sem_vd, mean_energy_eV)

for run_dir in all_dirs:

    dirname = os.path.basename(run_dir)

    # Extract E/N value from folder name — handles integers and floats
    # e.g. data_swarm_10  → 10.0
    #      data_swarm_2p5 → 2.5  (if you use 'p' for decimal point)
    #      data_swarm_2.5 → 2.5
    match = re.search(r'data_swarm_([0-9]+(?:[p.][0-9]+)?)', dirname)
    if not match:
        print(f"  SKIP {dirname}: cannot parse E/N from name")
        continue

    en_str = match.group(1).replace('p', '.')
    EN_Td  = float(en_str)

    field_dir = os.path.join(run_dir, "field")
    meta_file = os.path.join(run_dir, "metadata.h5")

    # ── sanity checks ──────────────────────────────────────
    if not os.path.isfile(meta_file):
        print(f"  SKIP {dirname}: no metadata.h5")
        continue
    if not os.path.isdir(field_dir):
        print(f"  SKIP {dirname}: no field/ directory")
        continue

    # ── read metadata ──────────────────────────────────────
    try:
        with h5py.File(meta_file, "r") as meta:
            grp      = meta["/metadata"]
            DT_coeff = grp.attrs["DT_coeff"]
            wpe      = grp.attrs["wpe"]
            LDe      = grp.attrs["LDe"]
    except Exception as e:
        print(f"  SKIP {dirname}: metadata read error — {e}")
        continue

    vel_norm = wpe * LDe    # normalised → m/s

    # ── collect field files ────────────────────────────────
    field_files = sorted(
        [(int(f[5:-3]), f)
         for f in os.listdir(field_dir)
         if f.startswith("data_") and f.endswith(".h5")],
        key=lambda x: x[0]
    )

    if len(field_files) == 0:
        print(f"  SKIP {dirname}: no field files")
        continue

    # ── compute vdrift at each snapshot ───────────────────
    vdrift_ts = []

    for ts_num, fname in field_files:
        fpath = os.path.join(field_dir, fname)
        try:
            with h5py.File(fpath, "r") as f:
                data = f["fielddata"][f"vel_{particle_type}"][particle_type][:]
        except Exception as e:
            print(f"  WARN {dirname} ts={ts_num}: {e}")
            continue

        # data[:, 0] = x positions, data[:, 1] = mean velocity (normalised)
        # spatial mean over all grid points = drift velocity
        vdrift_ts.append(np.mean(data[:, 1]) * vel_norm)

    if len(vdrift_ts) == 0:
        print(f"  SKIP {dirname}: no valid snapshots")
        continue

    vdrift_arr = np.array(vdrift_ts)

    # ── steady-state: last 30% ─────────────────────────────
    N    = len(vdrift_arr)
    i_ss = int(0.7 * N)

    vd_ss = vdrift_arr[i_ss:]

    mean_vd = np.mean(vd_ss)
    std_vd  = np.std(vd_ss)
    sem_vd  = std_vd / np.sqrt(len(vd_ss))

    # ── convergence check ──────────────────────────────────
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
        "std_vd"   : std_vd,
        "sem_vd"   : sem_vd,
        "converged": converged,
        "dirname"  : dirname,
    })

    status = "OK" if converged else "WARN"
    print(f"  [{status}] E/N={EN_Td:6.1f} Td  |vdrift|={abs(mean_vd):.4e} m/s"
          f"  (N_ss={len(vd_ss)})")

if len(results) == 0:
    print("ERROR: no valid results found.")
    sys.exit(1)

# ── sort by E/N ────────────────────────────────────────────
results.sort(key=lambda r: r["EN_Td"])

EN_sim  = np.array([r["EN_Td"]          for r in results])
W_sim   = np.array([abs(r["mean_vd"])   for r in results])
W_err   = np.array([r["sem_vd"]         for r in results])
conv    = np.array([r["converged"]       for r in results])

# ============================================================
# Print summary table
# ============================================================

print("\n" + "="*55)
print("  SUMMARY TABLE")
print("="*55)
print(f"  {'E/N (Td)':>10}  {'|W| sim (m/s)':>15}  {'SEM (m/s)':>12}")
print("-"*55)
for r in results:
    flag = "" if r["converged"] else "  ← not converged"
    print(f"  {r['EN_Td']:>10.2f}  {abs(r['mean_vd']):>15.4e}"
          f"  {r['sem_vd']:>12.4e}{flag}")
print("="*55)

# ============================================================
# Figure 1 — vdrift vs E/N  (main benchmark plot)
# ============================================================

fig1, ax1 = plt.subplots(figsize=(9, 5.5))

# Phelps reference
ax1.plot(phelps_EN, phelps_W, 'r-', lw=2, zorder=3,
         label='Phelps/LXCat (Boltzmann, 300 K)')

# simulation — converged points
mask_ok   = conv
mask_warn = ~conv

if mask_ok.any():
    ax1.errorbar(EN_sim[mask_ok], W_sim[mask_ok], yerr=W_err[mask_ok],
                 fmt='o', color='steelblue', markersize=7, capsize=4,
                 lw=1.5, zorder=4, label='PIC-MCC (converged)')

if mask_warn.any():
    ax1.errorbar(EN_sim[mask_warn], W_sim[mask_warn], yerr=W_err[mask_warn],
                 fmt='s', color='orange', markersize=7, capsize=4,
                 lw=1.5, zorder=4, label='PIC-MCC (not converged)')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$E/N$  (Td)', fontsize=13)
ax1.set_ylabel(r'$v_\mathrm{drift}$  (m/s)', fontsize=13)
ax1.set_title('Electron drift velocity in Ar — PIC-MCC vs Phelps', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, which='both', linestyle='--', alpha=0.35)
plt.tight_layout()
plt.savefig('swarm_vdrift_vs_EN.png', dpi=150)
print("\nSaved: swarm_vdrift_vs_EN.png")

# ============================================================
# Figure 2 — ratio W_sim / W_phelps (error diagnostic)
# ============================================================

# interpolate Phelps at simulation E/N points
W_ref_interp = np.interp(EN_sim, phelps_EN, phelps_W)
ratio = W_sim / W_ref_interp

fig2, ax2 = plt.subplots(figsize=(9, 4))

ax2.axhline(1.0, color='black', lw=1.5, linestyle='--', label='Perfect agreement')
ax2.axhspan(0.9, 1.1, alpha=0.1, color='green', label='±10% band')
ax2.axhspan(0.8, 1.2, alpha=0.07, color='orange', label='±20% band')

if mask_ok.any():
    ax2.plot(EN_sim[mask_ok], ratio[mask_ok], 'o', color='steelblue',
             markersize=7, label='PIC-MCC (converged)')
if mask_warn.any():
    ax2.plot(EN_sim[mask_warn], ratio[mask_warn], 's', color='orange',
             markersize=7, label='PIC-MCC (not converged)')

#ax2.set_xscale('log')
ax2.set_xlabel(r'$E/N$  (Td)', fontsize=13)
ax2.set_ylabel(r'$W_\mathrm{sim}\ /\ W_\mathrm{Phelps}$', fontsize=13)
ax2.set_title('Ratio: PIC-MCC / Phelps reference', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, which='both', linestyle='--', alpha=0.35)
plt.tight_layout()
plt.savefig('swarm_ratio_vs_EN.png', dpi=150)
print("Saved: swarm_ratio_vs_EN.png")

plt.show()