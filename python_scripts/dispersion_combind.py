"""
Merged dispersion analysis script.
- Data loading & FFT pipeline : Code 2 (hpic-style, separate metadata.h5 + field dir)
- Analytic dispersion solver  : Code 1 (symbolic 8th-order polynomial via sympy)

@author : merged version
"""

import numpy as np
from copy import deepcopy
from scipy.constants import value as constants
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib as mp
import matplotlib.pyplot as plt
import os, sys
from os.path import join as pjoin
from sympy import *
import h5py
from tqdm import tqdm

# ─────────────────────────── physical constants ──────────────────────────────
eps0 = constants('electric constant')
kb   = constants('Boltzmann constant')
me   = constants('electron mass')
AMU  = constants('atomic mass constant')
e    = constants('elementary charge')

# ─────────────────────────── CLI argument ────────────────────────────────────
if len(sys.argv) != 2:
    print("Usage: python dispersion_merged.py <path_to_data>")
    sys.exit(1)

base_path = os.path.abspath(sys.argv[1].strip())

phase_dir = os.path.join(base_path, "phase")
field_dir = os.path.join(base_path, "field")
meta_file = os.path.join(base_path, "metadata.h5")

path_fig = os.path.join(base_path, "analysis")
os.makedirs(path_fig, exist_ok=True)

print("\nBase path :", base_path)

# ─────────────────────────── sanity checks ───────────────────────────────────
if not os.path.exists(meta_file):
    print("metadata.h5 not found"); sys.exit(1)
if not os.path.isdir(field_dir):
    print("field directory not found"); sys.exit(1)

# ─────────────────────────── read metadata ───────────────────────────────────
with h5py.File(meta_file, 'r') as f_meta:
    mg = f_meta['/metadata']

    NC                  = mg.attrs['NC']
    NUM_TS              = mg.attrs['NUM_TS']
    write_interval      = mg.attrs['write_int']
    write_interval_phase= mg.attrs['write_int_phase']
    DT_coeff            = mg.attrs['DT_coeff']

    # frequencies / lengths (stored directly in Code-2 metadata)
    wpe_meta  = mg.attrs['wpe']
    wpi_meta  = mg.attrs['wpi']
    LDe       = mg.attrs['LDe']
    LDi       = mg.attrs['LDi']

    normscheme = mg.attrs['norm_scheme']
    n0         = mg.attrs['density']

    # ── particle / plasma parameters needed for analytic solver ──────────────
    # These mirror Code-1's metadata attrs.  Add try/except so the script
    # still runs (without analytic curves) if they are absent.
    try:
        Te   = 1#mg.attrs['Te']          # electron temperature  [J or eV – match Code1]
        Ti   = 0.026#mg.attrs['Ti']
        Tb   = Ti#mg.attrs['Tb']          # beam negative-ion temperature
        Tn   = Ti#mg.attrs['Tn']          # background negative-ion temperature
        mi   = 1.6726219e-27#mg.attrs['mI']          # ion mass [kg]
        mn   = 1.6726219e-27#mg.attrs['mN']
        mb   = 1.6726219e-27#mg.attrs['mB']
        alp  = 10#mg.attrs['alpha']       # nn0/ne0
        beta = 0.4#mg.attrs['beta']        # nb0/ne0
        vb   = 20#mg.attrs['vb']          # beam speed in normalised units
        have_analytic_params = True
    except KeyError as err:
        print(f"[WARN] Analytic parameter missing in metadata: {err}")
        print("       Analytic dispersion curves will be skipped.")
        have_analytic_params = False

# ─────────────────────────── normalisation ───────────────────────────────────
mFactor = wpi_meta / wpe_meta
if normscheme in [1, 2]:
    mFactor = 1

L = LDe;  W = wpe_meta
if normscheme in [2, 4]:
    L = LDi; W = wpi_meta

DT = DT_coeff * (1.0 / W)

print("normscheme :", normscheme)
print("mFactor    :", mFactor)

# ─────────────────────────── derived plasma quantities ───────────────────────
if have_analytic_params:
    ni0  = n0
    ne0  = n0 / (1 + alp + beta)
    nn0  = alp  * ne0
    nb0  = beta * ne0

    # thermal velocities
    vthi = np.sqrt(Ti / mi)
    vthe = np.sqrt(Te / me)
    vthn = np.sqrt(Tn / mn)
    vthb = np.sqrt(Tb / mb)

    # plasma frequencies  (re-derived; cross-check against stored wpe/wpi)
    wpe  = np.sqrt(ne0 * e**2 / (eps0 * me))
    wpi  = np.sqrt(ni0 * e**2 / (eps0 * mi))
    wpn  = np.sqrt(nn0 * e**2 / (eps0 * mn))
    wpnb = np.sqrt(nb0 * e**2 / (eps0 * mb))

    LDE  = np.sqrt(eps0 * Te / (ne0 * e**2))   # electron Debye length
    LDI  = np.sqrt(eps0 * Ti / (ni0 * e**2))   # ion Debye length

    # Characteristic scales for analytic branch
    L_an = LDI
    W_an = wpi

    print(f"wpi/wpe = {wpi/wpe:.4f}")

# ─────────────────────────── symbolic dispersion coefficients ────────────────
def symbolic_coeffs():
    """
    Return polynomial coefficients of the 8th-order dispersion relation
    (background ions + electrons + negative ions + beam negative ions).
    Identical to Code-1.
    """
    w, k, vthi_, vthe_, vthn_, vthb_, vnb0_, wpi_, wpe_, wpn_, wpnb_ = symbols(
        'w k vthi vthe vthn vthb vnb0 wpi wpe wpn wpnb')

    A = w**2 - k**2 * vthi_**2                       # background ions
    B = w**2 - k**2 * vthe_**2                       # background electrons
    C = w**2 - k**2 * vthn_**2                       # background negative ions
    D = (w - k*vnb0_)**2 - k**2 * vthb_**2           # beam negative ions

    expr = A*B*C*D - B*C*D*wpi_**2 - A*C*D*wpe_**2 - A*B*D*wpn_**2 - A*B*C*wpnb_**2
    p    = Poly(expr, w)
    return p.all_coeffs()

# ─────────────────────────── root-ordering helper ────────────────────────────
def correct(roots):
    """
    Re-order roots sample-by-sample so consecutive samples have the
    smallest possible jump (written by Dr. Sigvald Marholm).
    """
    nSamples, nRoots = roots.shape
    for i in range(1, nSamples):
        for j in range(nRoots):
            dist    = np.abs(roots[i, :] - roots[i-1, j])
            closest = np.argmin(dist)
            if closest != j:
                tmp               = deepcopy(roots[i:, j])
                roots[i:, j]     = roots[i:, closest]
                roots[i:, closest] = tmp
    return roots

# ─────────────────────────── analytic EPW solver ─────────────────────────────
def EPW_dispersion(k_array, coff,
                   vthi, vthe, vthn, vthb, vnb0,
                   wpi, wpe, wpn, wpnb):
    """
    Solve the 8th-order polynomial at each k in k_array.
    coff  – symbolic coefficient list from symbolic_coeffs()
    Returns roots array  shape (len(k_array)-1, 8).
    """
    # evaluate symbolic coefficients numerically
    coeff_num = [eval(str(c)) for c in coff]

    roots = []
    for i in range(1, len(k_array)):
        # substitute the current k value where needed
        coeffs = []
        for c in coeff_num:
            if callable(getattr(c, '__len__', None)):   # array
                coeffs.append(c[i])
            else:                                        # scalar (coeff1 is always scalar)
                coeffs.append(c)
        root = np.roots(coeffs)
        roots.append(root)

    roots = np.array(roots)
    roots = correct(roots)
    return roots


# ─────────────────────────── load electric field data ────────────────────────
field_files = []
for fname in os.listdir(field_dir):
    if fname.startswith("data_") and fname.endswith(".h5"):
        ts = int(fname[5:-3])
        field_files.append((ts, fname))
field_files.sort()

electric_field_data = []
for ts, fname in field_files:
    with h5py.File(os.path.join(field_dir, fname), 'r') as f:
        data = f["fielddata"]["efield"]["efield"][:]
        electric_field_data.append(data[:, 1])

EF      = np.vstack(electric_field_data)   # shape (Nt, Nx)
Nt, Nx  = EF.shape
x       = np.linspace(0, NC, Nx, endpoint=False)
dx      = x[1] - x[0]
print(f"Data shape: EF({Nt}, {Nx})")

# ─────────────────────────── 2-D FFT → power spectrum ────────────────────────
efield_norm    = (n0 * e * L) / eps0
F_k            = np.fft.fft(EF, axis=1, norm='ortho') * efield_norm * (2 / np.sqrt(NC))
kfft           = 2 * np.pi * np.fft.fftfreq(Nx, dx)

F_k_t          = np.fft.fft(F_k, axis=0, norm='ortho')
omega_fft      = 2 * np.pi * np.fft.fftfreq(Nt, DT_coeff * write_interval * mFactor)

F_k_t_shifted  = np.fft.fftshift(F_k_t, axes=(0, 1))
k_shifted      = np.fft.fftshift(kfft)
omega_shifted  = np.fft.fftshift(omega_fft)

dispersion_power = np.abs(F_k_t_shifted)**2
dispersion_power /= np.max(dispersion_power)

power_log  = np.log10(dispersion_power)
power_log  = power_log[:, ::-1]
k_shifted  = -k_shifted[::-1]

# ─────────────────────────── analytic branches ───────────────────────────────
if have_analytic_params:
    print("\nBuilding symbolic dispersion coefficients …")
    coff = symbolic_coeffs()
    print(f"Highest power of omega: {len(coff)-1}\n")

    # un-normalised k array (same convention as Code-1)
    k_un   = 2 * np.pi * np.arange(NC + 1) / (NC * dx * L_an)   # [rad/m]
    kappa  = k_un[1:] * LDI                                        # k·λDi  (normalised)
    vnb0_  = vb * (L_an * W_an)                                    # beam drift [m/s]

    # Evaluate symbolic coefficients at each k value
    # (replicates Code-1's EPW_dispersion inner logic but cleaner)
    print("Evaluating polynomial coefficients …")
    coeff_arrays = []
    for sym_c in coff:
        sym_str = str(sym_c)
        # replace symbolic names with numpy expressions evaluated over k_un array
        # We use the same variable names as in symbolic_coeffs()
        val = eval(sym_str, {
            'k'    : k_un,
            'vthi' : vthi,  'vthe' : vthe,  'vthn' : vthn,  'vthb' : vthb,
            'vnb0' : vnb0_, 'wpi'  : wpi,   'wpe'  : wpe,   'wpn'  : wpn,
            'wpnb' : wpnb,
        })
        coeff_arrays.append(val)

    print("Solving 8th-order polynomial at each k …")
    roots_list = []
    for i in range(1, len(k_un)):
        coeffs = []
        for c in coeff_arrays:
            if np.ndim(c) == 0:          # scalar coefficient (e.g. leading term = 1)
                coeffs.append(float(c))
            else:
                coeffs.append(float(c[i]))
        roots_list.append(np.roots(coeffs))

    roots_EPW = np.array(roots_list)      # shape (NC, 8)
    roots_EPW = correct(roots_EPW)

    # split into real / imaginary parts normalised by wpi
    ep_re = np.real(roots_EPW) / wpi      # shape (NC, 8)
    ep_im = np.imag(roots_EPW) / wpi

    # fastest-growing mode
    idx  = np.argmax(ep_im)
    i_fg, j_fg = np.unravel_index(idx, ep_im.shape)
    print(f"Fastest-growing mode: k·λDi = {kappa[i_fg]:.4g}, "
          f"ω/ωpi = {ep_re[i_fg, j_fg]:.4g}+{ep_im[i_fg, j_fg]:.4g}i")

    # ── additional simple dispersion curves ──────────────────────────────────
    k1      = k_un[1:]                           # avoid k=0
    CF      = 1.0 / (1.0 + k1**2 * LDE**2)
    CS      = np.sqrt((Te * CF + Ti) / mi)
    w_ia    = CS * k1
    w_beam  = vnb0_ * k1

    alpha1  = alp / (1 + alp + beta)
    w_fast  = (np.sqrt(
        ((Ti / mi) * (1 + alpha1 + 3*(k1**2 * LDI**2) + 3*(1 - alpha1)*(Ti/Te))) /
        ((k1 * LDI)**2 + (Ti/Te)*(1 - alpha1))
    ) * k1)

# ─────────────────────────── IAW analytic (Code-2 style) ─────────────────────
EV_TO_K    = e / kb
k_analytic = k_shifted[k_shifted > 0] / LDi

alpha1_c2  = 10 / (1 + 10 + 0.4)          # Code-2 hard-coded values; adjust if needed
omega_IAW  = (np.sqrt(
    ((Ti * EV_TO_K * kb / mi) * (1 + alpha1_c2 + 3*(k_analytic**2 * LDi**2)
      + 3*(1 - alpha1_c2)*(Ti/Te))) /
    ((k_analytic * LDi)**2 + (Ti/Te)*(1 - alpha1_c2))
) * k_analytic) if have_analytic_params else None

# ─────────────────────────── plotting ────────────────────────────────────────
figsize = np.array([150, 150/1.618])
dpi     = 300
ppi     = np.sqrt(1920**2 + 1200**2) / 24

mp.rc('text',   usetex=False)
mp.rc('font',   family='sans-serif', size=10)
mp.rc('axes',   titlesize=10, labelsize=10)
mp.rc('xtick',  labelsize=10)
mp.rc('ytick',  labelsize=10)
mp.rc('legend', fontsize=9)

ms = 0.8   # marker / line size for analytic curves

# ── Figure 1 : real-part dispersion (power spectrum + analytic real roots) ───
fig, ax = plt.subplots(figsize=figsize/25.4, dpi=ppi, constrained_layout=True)
extent  = [k_shifted[0], k_shifted[-1], omega_shifted[0], omega_shifted[-1]]

ax.imshow(power_log, extent=extent, origin='lower', aspect='auto',
          interpolation='bilinear', cmap='RdBu_r')

if have_analytic_params:
    for col in range(ep_re.shape[1]):
        ax.plot(kappa,  ep_re[:, col], 'k-', markersize=ms, linewidth=ms)
        ax.plot(-kappa, ep_re[:, col], 'k-', markersize=ms, linewidth=ms)   # mirror
        ax.plot(kappa, -ep_re[:, col], 'k-', markersize=ms, linewidth=ms)

    # Highlight fastest-growing mode
    #ax.plot(kappa[i_fg], ep_re[i_fg, j_fg], 'xr', markersize=6,label=f'Fastest mode (k={kappa[i_fg]:.3g})')

    # simple dispersion curves
    #ax.plot(kappa, w_ia   / wpi, 'r--', lw=0.8, alpha=0.7, label='IAW')
    #ax.plot(kappa, w_beam / wpi, 'b--', lw=0.8, alpha=0.7, label='BW')
    #ax.plot(kappa, w_fast / wpi, 'm--', lw=0.8, alpha=0.7, label='FW')

#if omega_IAW is not None:
    #ax.plot( k_analytic*LDi,  omega_IAW/wpi, 'c--', lw=1.2, label='IAW (analytic)')
    #ax.plot(-k_analytic*LDi, -omega_IAW/wpi, 'c--', lw=1.2)
    #ax.plot( k_analytic*LDi, -omega_IAW/wpi, 'c--', lw=1.2)
    #ax.plot(-k_analytic*LDi,  omega_IAW/wpi, 'c--', lw=1.2)

ax.set_xlabel(r'$k\,\lambda_{Di}$')
ax.set_ylabel(r'$\omega_r/\omega_{pi}$')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-4, 4)
ax.legend(loc='upper left', framealpha=0.5)
ax.grid(False)

cbar = fig.colorbar(ax.images[0], ax=ax)
cbar.set_label(r'$\log|E(k,\omega)|^2$')

out_real = pjoin(path_fig, "dispersion_real.png")
plt.savefig(out_real, dpi=dpi)
print(f"Saved → {out_real}")

# ── Figure 2 : imaginary-part (growth-rate) plot ─────────────────────────────
if have_analytic_params:
    fig2, ax2 = plt.subplots(figsize=figsize/25.4, dpi=ppi, constrained_layout=True)

    for col in range(ep_im.shape[1]):
        ax2.plot(kappa, ep_im[:, col], 'k-', markersize=ms, linewidth=ms)

    ax2.plot(kappa[i_fg], ep_im[i_fg, j_fg], 'xr', markersize=6,
             label=f'Max growth (k={kappa[i_fg]:.3g})')

    ax2.set_xlabel(r'$k\,\lambda_{Di}$')
    ax2.set_ylabel(r'$\gamma/\omega_{pi}$')
    ax2.set_xlim(0, 1.0)
    ax2.legend(loc='upper right', framealpha=0.5)
    ax2.grid(True)

    out_imag = pjoin(path_fig, "dispersion_imaginary.png")
    plt.savefig(out_imag, dpi=dpi)
    print(f"Saved → {out_imag}")

plt.show()