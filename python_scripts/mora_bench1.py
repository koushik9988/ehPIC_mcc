import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import os
import sys

# ══════════════════════════════════════════════════════════════════════════════
# Mora Theory vs PIC Benchmark
# Tracks: front velocity, front position, peak E-field, density profile
#
# Usage:
#   python plot_mora_benchmark.py <output_folder> <species> <ts1> <ts2> ...
#
# Example:
#   python plot_mora_benchmark.py ./output ion 200 500 1000 2000 5000
# ══════════════════════════════════════════════════════════════════════════════

if len(sys.argv) < 4:
    print("Usage: python plot_mora_benchmark.py <output_folder> <species> <ts1> <ts2> ...")
    sys.exit(1)

path    = os.path.abspath(sys.argv[1])
species = sys.argv[2]
ts_list = [int(t) for t in sys.argv[3:]]

meta_file = os.path.join(path, "metadata.h5")
plot_dir  = os.path.join(path, "plots")
os.makedirs(plot_dir, exist_ok=True)

# ── Metadata ──────────────────────────────────────────────────────────────────
with h5py.File(meta_file, "r") as f:
    DT_coeff = f["/metadata"].attrs["DT_coeff"]
    wpe      = f["/metadata"].attrs["wpe"]
    wpi      = f["/metadata"].attrs["wpi"]
    LDe      = f["/metadata"].attrs["LDe"]

mass_ratio = 1.0 / 1836.0
vcs_code   = np.sqrt(mass_ratio)   # cs / v_the
x_0_code   =128.0                  # vacuum interface (ions placed 0→64 LDe)

# ══════════════════════════════════════════════════════════════════════════════
#  Mora Theory Functions
# ══════════════════════════════════════════════════════════════════════════════

def mora_vfront(t_wpe):
    """
    Mora (2003) ion front velocity in units of cs.
    v_front = 2*cs*sqrt( ln(wpe*t) - 0.5*ln(2*pi*ln(wpe*t)) )
    Valid for wpe*t >> 1.
    """
    if t_wpe <= np.e:        # need ln > 1 for sqrt to be real
        return 0.0
    log_t = np.log(t_wpe)
    inner = log_t - 0.5 * np.log(2.0 * np.pi * log_t)
    if inner <= 0:
        return 0.0
    return 2.0 * np.sqrt(inner)   # in units of cs

def mora_xi_front(t_wpe):
    """Self-similar coordinate of ion front: xi_front = v_front/cs - 1"""
    return mora_vfront(t_wpe) - 1.0

def mora_xfront_phys(t_wpe):
    """Physical front position in code units: x_front = x0 + cs*t * xi_front"""
    t_code = t_wpe          # t*wpe (dimensionless time in code units)
    return x_0_code + vcs_code * t_code * mora_xi_front(t_wpe)

def mora_Efront(t_wpe, n0):
    """
    Mora peak electric field at the rarefaction front (xi = -1):
      E_max = (n0 * Te) / (e * cs * t)
    In code units where v_the=1, Te/me = v_the^2 = 1, so Te = me = mass_ratio*mi
    We return E in code units: E_code = E / (n0_ref * Te / (e * LDe))
    Simplified: E_front_norm = 1 / (wpe * t)  (in units of cs*me*wpe/e)
    """
    if t_wpe <= 0:
        return 0.0
    # E_front in code units ~ cs / (wpe * t * LDe) * n0_ratio
    # Since cs = vcs_code * v_the, and we normalise to n0:
    return vcs_code / t_wpe   # dimensionless, scales as 1/t

# ══════════════════════════════════════════════════════════════════════════════
#  Extract PIC quantities at each timestep
# ══════════════════════════════════════════════════════════════════════════════

# Storage arrays
t_wpe_arr    = []
t_wpi_arr    = []

# PIC measurements
pic_vfront   = []   # velocity of foremost ion  [cs units]
pic_xfront   = []   # position of foremost ion  [code units]
pic_xi_front = []   # self-similar coord of front

# Mora predictions
mora_vf_arr  = []
mora_xf_arr  = []
mora_xif_arr = []

print(f"\n{'ts':>8}  {'t·wpe':>8}  {'t·wpi':>7}  "
      f"{'PIC v_f/cs':>12}  {'Mora v_f/cs':>12}  "
      f"{'PIC xi_f':>10}  {'Mora xi_f':>10}")
print("─" * 85)

for ts in ts_list:
    field_file = os.path.join(path, "field", f"data_{ts}.h5")
    if not os.path.exists(field_file):
        print(f"[WARN] Missing: {field_file} — skipping")
        continue

    with h5py.File(field_file, "r") as f:
        data_vel = f["fielddata"][f"vel_{species}"][species][:]
        data_den = f["fielddata"][f"den_{species}"][species][:]

    x_code   = data_vel[:, 0]
    vel_code = data_vel[:, 1]
    den_code = data_den[:, 1]

    time_code = ts * DT_coeff           # t * wpe
    wpit      = time_code * wpi / wpe   # t * wpi

    # Bulk density (leftmost 5%)
    bulk_end = max(len(den_code) // 20, 1)
    n_0      = np.mean(den_code[:bulk_end])

    # ── PIC front detection ──────────────────────────────────────────────────
    # Method: front = position where density first drops below threshold
    threshold = 0.01 * n_0   # 1% of bulk density

    # Only look in the expansion region (x > x0)
    mask_right = x_code >= x_0_code
    x_right    = x_code[mask_right]
    n_right    = den_code[mask_right]
    v_right    = vel_code[mask_right]

    # Find foremost cell above threshold
    above_thresh = n_right > threshold
    if np.any(above_thresh):
        front_idx   = np.where(above_thresh)[0][-1]   # last cell above threshold
        x_front_pic = x_right[front_idx]
        v_front_pic = v_right[front_idx] / vcs_code   # convert to cs units
    else:
        # Fallback: use maximum velocity position
        front_idx   = np.argmax(v_right)
        x_front_pic = x_right[front_idx]
        v_front_pic = v_right[front_idx] / vcs_code

    xi_front_pic = (x_front_pic - x_0_code) / (vcs_code * time_code)

    # ── Mora predictions ─────────────────────────────────────────────────────
    mv_front  = mora_vfront(time_code)
    mxi_front = mora_xi_front(time_code)
    mx_front  = mora_xfront_phys(time_code)

    # Store
    t_wpe_arr.append(time_code)
    t_wpi_arr.append(wpit)
    pic_vfront.append(v_front_pic)
    pic_xfront.append(x_front_pic)
    pic_xi_front.append(xi_front_pic)
    mora_vf_arr.append(mv_front)
    mora_xf_arr.append(mx_front)
    mora_xif_arr.append(mxi_front)

    print(f"{ts:>8}  {time_code:>8.1f}  {wpit:>7.2f}  "
          f"{v_front_pic:>12.3f}  {mv_front:>12.3f}  "
          f"{xi_front_pic:>10.3f}  {mxi_front:>10.3f}")

# Convert to arrays
t_wpe_arr    = np.array(t_wpe_arr)
t_wpi_arr    = np.array(t_wpi_arr)
pic_vfront   = np.array(pic_vfront)
pic_xfront   = np.array(pic_xfront)
pic_xi_front = np.array(pic_xi_front)
mora_vf_arr  = np.array(mora_vf_arr)
mora_xf_arr  = np.array(mora_xf_arr)
mora_xif_arr = np.array(mora_xif_arr)

# Mora continuous curves (fine time axis)
t_fine     = np.linspace(max(t_wpe_arr.min() * 0.5, np.e + 1), t_wpe_arr.max() * 1.2, 500)
mora_vf_c  = np.array([mora_vfront(t) for t in t_fine])
mora_xf_c  = np.array([mora_xfront_phys(t) for t in t_fine])
mora_xif_c = np.array([mora_xi_front(t) for t in t_fine])

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Front Velocity vs Time
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(12, 9))
fig1.suptitle(
    rf"Mora (2003) Theory vs PIC — Ion Front Quantities ({species}, Boltzmann $e^-$)",
    fontsize=13, fontweight='bold'
)

# ── Panel A: Front Velocity vs t·wpe ─────────────────────────────────────────
ax = axes[0, 0]
ax.plot(t_fine, mora_vf_c, 'k--', lw=2, label=r"Mora: $2c_s\sqrt{\ln(\omega_{pe}t)}$")
ax.scatter(t_wpe_arr, pic_vfront, color='tab:blue', s=60, zorder=5,
           label="PIC (front detection)")
ax.plot(t_wpe_arr, pic_vfront, color='tab:blue', lw=1.2, alpha=0.6)
ax.set_xlabel(r"$t \cdot \omega_{pe}$", fontsize=11)
ax.set_ylabel(r"$v_{front} / c_s$", fontsize=11)
ax.set_title("Ion Front Velocity vs Time", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, linestyle=':', alpha=0.5)

# Annotate each point with t·wpi
for i, (tw, vf) in enumerate(zip(t_wpi_arr, pic_vfront)):
    ax.annotate(rf"$\omega_{{pi}}t$={tw:.1f}", (t_wpe_arr[i], vf),
                textcoords="offset points", xytext=(5, 4), fontsize=7, color='tab:blue')

# ── Panel B: Front Velocity vs t·wpi (more physical) ─────────────────────────
ax = axes[0, 1]
t_wpi_fine = t_fine * wpi / wpe
ax.plot(t_wpi_fine, mora_vf_c, 'k--', lw=2, label="Mora Theory")
ax.scatter(t_wpi_arr, pic_vfront, color='tab:red', s=60, zorder=5, label="PIC")
ax.plot(t_wpi_arr, pic_vfront, color='tab:red', lw=1.2, alpha=0.6)
ax.set_xlabel(r"$t \cdot \omega_{pi}$", fontsize=11)
ax.set_ylabel(r"$v_{front} / c_s$", fontsize=11)
ax.set_title("Ion Front Velocity vs Ion Time", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, linestyle=':', alpha=0.5)

# ── Panel C: Front Position (self-similar ξ) vs time ─────────────────────────
ax = axes[1, 0]
ax.plot(t_fine, mora_xif_c, 'k--', lw=2,
        label=r"Mora: $\xi_{front} = 2\sqrt{\ln(\omega_{pe}t)}-1$")
ax.scatter(t_wpe_arr, pic_xi_front, color='tab:green', s=60, zorder=5, label="PIC")
ax.plot(t_wpe_arr, pic_xi_front, color='tab:green', lw=1.2, alpha=0.6)
ax.set_xlabel(r"$t \cdot \omega_{pe}$", fontsize=11)
ax.set_ylabel(r"$\xi_{front} = (x_{front} - x_0)/(c_s t)$", fontsize=11)
ax.set_title(r"Self-Similar Front Position $\xi_{front}$ vs Time", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, linestyle=':', alpha=0.5)
ax.axhline(0, color='gray', lw=0.8, ls=':')

# ── Panel D: Relative error PIC vs Mora ──────────────────────────────────────
ax = axes[1, 1]
err_v  = 100.0 * (pic_vfront   - mora_vf_arr)  / (mora_vf_arr  + 1e-10)
err_xi = 100.0 * (pic_xi_front - mora_xif_arr) / (mora_xif_arr + 1e-10)

ax.plot(t_wpi_arr, err_v,  'o-', color='tab:blue',  lw=1.5, ms=6, label=r"$v_{front}$ error")
ax.plot(t_wpi_arr, err_xi, 's-', color='tab:green', lw=1.5, ms=6, label=r"$\xi_{front}$ error")
ax.axhline(0,   color='black', lw=1.0, ls='--')
ax.axhspan(-10, 10, alpha=0.08, color='green', label="±10% band")
ax.set_xlabel(r"$t \cdot \omega_{pi}$", fontsize=11)
ax.set_ylabel("Relative Error (%)", fontsize=11)
ax.set_title("PIC vs Mora: Relative Error", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_ylim(-50, 50)

# Note on convergence
ax.text(0.97, 0.05,
        "Error → 0 as\n" + r"$t\omega_{pi} \to \infty$",
        transform=ax.transAxes, fontsize=8, ha='right',
        bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.8))

plt.tight_layout()
out1 = os.path.join(plot_dir, f"mora_front_benchmark_{species}.png")
plt.savefig(out1, dpi=300, bbox_inches='tight')
print(f"\nSaved → {out1}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — v_front vs ln(wpe*t) straight-line check (Mora's key prediction)
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(7, 5))

ln_t      = np.log(t_wpe_arr)
ln_t_fine = np.log(t_fine)

ax.plot(ln_t_fine, mora_vf_c, 'k--', lw=2,
        label=r"Mora: $2c_s\sqrt{\ln(\omega_{pe}t)}$")
ax.scatter(ln_t, pic_vfront, color='tab:blue', s=70, zorder=5,
           label="PIC front velocity")
ax.plot(ln_t, pic_vfront, color='tab:blue', lw=1.2, alpha=0.5)

# Fit PIC to sqrt(ln t) form: v = A * sqrt(ln t)
from numpy.polynomial import polynomial as P
sqrt_ln_t = np.sqrt(ln_t)
# Linear fit: v = A * sqrt(ln t)  → fit through origin
A_fit = np.dot(sqrt_ln_t, pic_vfront) / np.dot(sqrt_ln_t, sqrt_ln_t)
ax.plot(ln_t_fine, A_fit * np.sqrt(ln_t_fine), 'r-', lw=1.5, alpha=0.7,
        label=rf"PIC fit: ${A_fit:.2f}\,\sqrt{{\ln(\omega_{{pe}}t)}}$ (Mora predicts 2.0)")

ax.set_xlabel(r"$\ln(\omega_{pe}\,t)$", fontsize=12)
ax.set_ylabel(r"$v_{front} / c_s$", fontsize=12)
ax.set_title(
    r"Mora's Key Prediction: $v_{front} \propto \sqrt{\ln(\omega_{pe}t)}$" + "\n"
    rf"({species}, Boltzmann $e^-$, $x_0 = {x_0_code:.0f}\,\lambda_{{De}}$)",
    fontsize=11
)
ax.legend(fontsize=10)
ax.grid(True, linestyle=':', alpha=0.5)

# Annotate coefficient
ax.text(0.05, 0.88,
        rf"Mora coefficient = 2.00" + "\n" + rf"PIC fit coefficient = {A_fit:.3f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

plt.tight_layout()
out2 = os.path.join(plot_dir, f"mora_sqrt_scaling_{species}.png")
plt.savefig(out2, dpi=300, bbox_inches='tight')
print(f"Saved → {out2}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Self-similar profile overlay (density + velocity at all times)
# ══════════════════════════════════════════════════════════════════════════════
fig3, (ax_n, ax_v) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
fig3.subplots_adjust(hspace=0.08)

cmap   = plt.cm.plasma
colors = [cmap(i / max(len(ts_list) - 1, 1)) for i in range(len(ts_list))]

xi_mora = np.linspace(-1.0, 6.5, 600)
n_mora  = np.exp(-(1.0 + xi_mora))
v_mora  = 1.0 + xi_mora

for idx, ts in enumerate(ts_list):
    field_file = os.path.join(path, "field", f"data_{ts}.h5")
    if not os.path.exists(field_file):
        continue

    with h5py.File(field_file, "r") as f:
        data_vel = f["fielddata"][f"vel_{species}"][species][:]
        data_den = f["fielddata"][f"den_{species}"][species][:]

    x_code   = data_vel[:, 0]
    vel_code = data_vel[:, 1]
    den_code = data_den[:, 1]

    time_code = ts * DT_coeff
    wpit      = time_code * wpi / wpe

    bulk_end = max(len(den_code) // 20, 1)
    n_0      = np.mean(den_code[:bulk_end])

    xi     = (x_code - x_0_code) / (vcs_code * time_code)
    n_norm = den_code / n_0
    v_norm = vel_code / vcs_code

    # Front marker
    xi_f = mora_xi_front(time_code)

    label = rf"$t\omega_{{pi}}$ = {wpit:.1f}"
    col   = colors[idx]

    ax_n.plot(xi, n_norm, color=col, lw=1.6, alpha=0.8, label=label)
    ax_v.plot(xi, v_norm, color=col, lw=1.6, alpha=0.8, label=label)

    # Mark predicted front position on both panels
    ax_n.axvline(xi_f, color=col, lw=0.8, ls='-.', alpha=0.5)
    ax_v.axvline(xi_f, color=col, lw=0.8, ls='-.', alpha=0.5)

# Mora theory
ax_n.plot(xi_mora, n_mora, 'k--', lw=2.2, label="Mora Theory", zorder=10)
ax_v.plot(xi_mora, v_mora, 'k--', lw=2.2, label="Mora Theory", zorder=10)

ax_n.set_ylabel(r"$n_i / n_0$", fontsize=12)
ax_n.set_ylim(-0.05, 1.20)
ax_n.set_xlim(-3.0, 7.0)
ax_n.axvline(-1.0, color='gray', lw=1.0, ls=':', alpha=0.6)
ax_n.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
ax_n.grid(True, linestyle=':', alpha=0.5)
ax_n.set_title(
    rf"Self-Similar Collapse ({species}) — Dashed verticals = Mora $\xi_{{front}}$",
    fontsize=11
)

ax_v.set_xlabel(r"$\xi = (x - x_0)/(c_s t)$", fontsize=12)
ax_v.set_ylabel(r"$v_i / c_s$", fontsize=12)
ax_v.set_ylim(-0.5, 8.0)
ax_v.axvline(-1.0, color='gray', lw=1.0, ls=':', alpha=0.6)
ax_v.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
ax_v.grid(True, linestyle=':', alpha=0.5)

# Colorbar
sm   = plt.cm.ScalarMappable(cmap=cmap,
        norm=plt.Normalize(vmin=t_wpi_arr[0], vmax=t_wpi_arr[-1]))
sm.set_array([])
cbar = fig3.colorbar(sm, ax=[ax_n, ax_v], orientation='vertical',
                     fraction=0.03, pad=0.02)
cbar.set_label(r"$t \cdot \omega_{pi}$", fontsize=11)

plt.tight_layout()
out3 = os.path.join(plot_dir, f"mora_selfsimilar_{species}.png")
plt.savefig(out3, dpi=300, bbox_inches='tight')
print(f"Saved → {out3}")

plt.show()
print("\nAll benchmark plots complete.")