import os
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.ndimage import gaussian_filter1d


if len(sys.argv) != 3:
    print("Usage: python analyze_kubo_ensemble.py <path_to_output> <particle_type>")
    sys.exit(1)

path = sys.argv[1]
particle_type = sys.argv[2]

# ---------------------- LOAD FILE ----------------------
#fname = pjoin(path, f"kubo_ensemble_time_{particle_type}.h5")
fname = pjoin(path, "analysis", "kubo_analysis", f"kubo_ensemble_{particle_type}.h5")
path_fig = pjoin(path, "analysis", "kubo_analysis")
os.makedirs(path_fig, exist_ok=True)

with h5py.File(fname, "r") as hf:
    v_centers = hf["v_centers"][:]
    tau       = hf["tau"][:]
    acf       = hf["acf"][:]
    tau_c     = hf["tau_c"][:]
    Ku        = hf["Ku"][:]
    counts    = hf["counts"][:]

print(f"Loaded file: {fname}")
print(f"acf shape = {acf.shape}  (nv_bins * window_steps)")

# ---------------------- BASIC CHECK ----------------------
valid_bins = counts > 0
print(f"Valid velocity bins: {np.sum(valid_bins)} / {len(v_centers)}")

# ---------------------- ACF HEATMAP ----------------------
plt.figure(figsize=(8, 6))
plt.imshow(
    acf,
    aspect="auto",
    origin="lower",
    interpolation= "bilinear",
    extent=[tau[0], tau[-1], v_centers[0], v_centers[-1]],
    cmap="inferno"
)
plt.colorbar(label="ACF")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$v$")
plt.title("Velocity-resolved ACF heatmap")
plt.tight_layout()
plt.savefig(pjoin(path_fig, "acf_heatmap.png"), dpi=200)
plt.show()


nv_bins = len(v_centers)
ib1 = nv_bins // 2  # choose middle velocity bin
ib2 = nv_bins // 4  # choose lower velocity bin
ib3 = 3 * nv_bins // 4  # choose higher velocity bin
ib4 = nv_bins - 4  # choose highest velocity bin

plt.figure(figsize=(7, 5))
plt.plot(tau, (acf[ib1]), lw=2)
plt.plot(tau, (acf[ib2]), lw=2)
plt.plot(tau, (acf[ib3]), lw=2)
plt.plot(tau, (acf[ib4]), lw=2)
plt.xlabel(r"$\tau$")
plt.ylabel("ACF")
plt.title(f"ACF vs τ (v = {v_centers[ib1]:.3f})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(pjoin(path_fig, f"acf_vs_tau_bin_{ib1}.png"), dpi=200)
plt.show()



nv_bins = len(v_centers)
ib1 = nv_bins // 2
ib2 = nv_bins // 4
ib3 = 3 * nv_bins // 4
ib4 = nv_bins - 4
plot_bins = [ib1, ib2, ib3, ib4]

from scipy.optimize import curve_fit

def exp_decay(t, tc):
    return np.exp(-t / tc)

plt.figure(figsize=(8, 5))

colors = ['steelblue', 'orange', 'green', 'red']

for color, ib in zip(colors, plot_bins):
    # ACF dots
    plt.plot(tau, acf[ib],'*',color=color, label=f'ACF v={v_centers[ib]:.3f}')
    # Exp fit line same color
    try:
        fit_end = len(tau) // 3
        popt, _ = curve_fit(exp_decay, tau[:fit_end], acf[ib, :fit_end],
                            p0=[tau[fit_end] * 0.3])
        tc_fit = popt[0]
        plt.plot(tau[:fit_end], exp_decay(tau[:fit_end], tc_fit),'-', color=color, lw=2, label=rf'Fit $\tau_c$={tc_fit:.2f}')
    except Exception:
        pass

plt.xlabel(r'$\tau$')
plt.ylabel('ACF')
plt.title('ACF vs τ with Exponential Fit')
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(pjoin(path_fig, "acf_with_expfit.png"), dpi=200)
plt.show()

# ---------------------- OPTIONAL: Ku vs v ----------------------
plt.figure(figsize=(7, 5))
plt.plot(v_centers, Ku, "o-", lw=2)
plt.xlabel(r"$v$")
plt.ylabel(r"$Ku(v)$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(pjoin(path_fig, "Ku_vs_v.png"), dpi=200)
plt.show()

# ---------------------- OPTIONAL: tau_c vs v ----------------------
plt.figure(figsize=(7, 5))
plt.plot(v_centers, tau_c, "o-", lw=2)
plt.xlabel(r"$v$")
plt.ylabel(r"$\tau_c(v)$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(pjoin(path_fig, "tau_c_vs_v.png"), dpi=200)
plt.show()

# ---------------------- OPTIONAL: counts ----------------------
plt.figure(figsize=(7, 5))
plt.bar(v_centers, counts, width=v_centers[1] - v_centers[0])
plt.xlabel(r"$v$")
plt.ylabel("Number of samples")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(pjoin(path_fig, "counts_vs_v.png"), dpi=200)
plt.show()


