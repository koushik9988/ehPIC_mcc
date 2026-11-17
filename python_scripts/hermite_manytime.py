"""
Hermite spectrum for multiple timesteps, overlaid on a single plot.

Usage:
    python hermite_multi.py <data_path> <species> [drift] -- <ts1> <ts2> ...

    data_path : directory containing metadata.h5 and phase/ subfolder
    species   : particle type string  (e.g. "ion", "electron")
    drift     : optional float, bulk drift velocity to subtract (default 0)
    ts1 ts2…  : one or more integer timestep numbers after the -- separator

Example:
    python hermite_multi.py /data/run01 ion 0.5 -- 2000 3000 4000 5000
    python hermite_multi.py /data/run01 electron -- 1000 2000 3000
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.cm as mcm
import h5py
import sys
import os
from os.path import join as pjoin


# ─────────────────────────────────────────────────────────────
# Hermite basis  ψ_m(v)  (physics / probabilist normalisation)
# ─────────────────────────────────────────────────────────────
def compute_hermite_basis(v, M):
    """
    Returns psi[m, iv] = ψ_m(v[iv]).
    Recurrence: ψ_{m+1} = sqrt(2/(m+1)) v ψ_m - sqrt(m/(m+1)) ψ_{m-1}
    """
    Nv = len(v)
    psi = np.zeros((M, Nv))
    psi[0] = np.pi**(-0.25) * np.exp(-v**2 / 2)
    if M > 1:
        psi[1] = np.sqrt(2) * v * psi[0]
    for m in range(1, M - 1):
        psi[m + 1] = (np.sqrt(2.0 / (m + 1)) * v * psi[m]
                      - np.sqrt(m       / (m + 1)) * psi[m - 1])
    return psi


# ─────────────────────────────────────────────────────────────
# Compute E_m = <|f_m(x)|^2>_x  for one phase file
# ─────────────────────────────────────────────────────────────
def hermite_spectrum(phase_path, species, drift, NC, Nv=1024, M=200):
    """
    Parameters
    ----------
    phase_path : str   – full path to phase_XXXX.h5
    species    : str
    drift      : float – bulk velocity to subtract
    NC         : int   – number of spatial cells
    Nv         : int   – velocity grid points
    M          : int   – number of Hermite modes

    Returns
    -------
    E_m : ndarray, shape (M,)
    """
    with h5py.File(phase_path, 'r') as pf:
        data = pf[f"{species}/phase"][:]

    x_raw = data[:, 0]
    v_raw = data[:, 1] - drift

    # ── velocity grid (add 10 % buffer)
    vmin = v_raw.min()
    vmax = v_raw.max()
    buf  = 0.1 * (vmax - vmin)
    vmin -= buf
    vmax += buf

    v_grid    = np.linspace(vmin, vmax, Nv)
    v_centers = 0.5 * (v_grid[:-1] + v_grid[1:])
    dv        = v_centers[1] - v_centers[0]

    # ── spatial grid & bin
    Nx     = NC
    x_grid = np.linspace(x_raw.min(), x_raw.max(), Nx)
    dx     = x_grid[1] - x_grid[0]

    ix    = np.searchsorted(x_grid, x_raw) - 1
    valid = (ix >= 0) & (ix < Nx)
    ix    = ix[valid]
    v     = v_raw[valid]

    # f(x, v): normalised distribution at each spatial cell
    f_xv = np.zeros((Nx, Nv - 1))
    for i in range(Nx):
        vi = v[ix == i]
        if vi.size:
            f_xv[i], _ = np.histogram(vi, bins=v_grid, density=True)
    # divide by Nx so that integrating over all x and v gives ~1
    f_xv /= Nx

    norm = np.sum(f_xv) * dx * dv
    print(f"  Normalisation check (should ≈ 1): {norm:.4f}")

    # ── Hermite projection  f_m(x) = ∫ f(x,v) ψ_m(v) dv
    psi  = compute_hermite_basis(v_centers, M)   # (M, Nv-1)
    f_m  = f_xv @ (psi * dv).T                  # (Nx, M)

    # ── x-averaged power spectrum
    E_m = np.mean(np.abs(f_m)**2, axis=0)        # (M,)
    return E_m


# ─────────────────────────────────────────────────────────────
# Parse CLI  –  split on "--" to separate options from timesteps
# ─────────────────────────────────────────────────────────────
def parse_args(argv):
    if "--" in argv:
        sep       = argv.index("--")
        ts_list   = [int(t) for t in argv[sep + 1:]]
        opts      = argv[1:sep]
    else:
        # No "--": assume last N pure-integer tokens are timesteps
        opts    = []
        ts_list = []
        for a in argv[1:]:
            try:
                ts_list.append(int(a))
            except ValueError:
                opts.append(a)

    if len(opts) < 2:
        print(__doc__)
        sys.exit(1)

    data_path = os.path.abspath(opts[0].strip())
    species   = opts[1].strip()
    drift     = float(opts[2]) if len(opts) > 2 else 0.0

    if not ts_list:
        print("Error: no timestep numbers supplied.")
        sys.exit(1)

    return data_path, species, drift, ts_list


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    data_path, species, drift, ts_list = parse_args(sys.argv)

    print(f"\nData path : {data_path}")
    print(f"Species   : {species}")
    print(f"Drift     : {drift}")
    print(f"Timesteps : {ts_list}\n")

    # ── metadata
    meta_file = pjoin(data_path, "metadata.h5")
    if not os.path.exists(meta_file):
        print("metadata.h5 not found"); sys.exit(1)

    with h5py.File(meta_file, 'r') as f:
        meta     = f['/metadata']
        NC       = int(meta.attrs['NC'])
        DT_coeff = float(meta.attrs['DT_coeff'])
        wpe      = float(meta.attrs['wpe'])
        wpi      = float(meta.attrs['wpi'])

    mfactor = wpi / wpe

    # ── Hermite parameters (tune here)
    M  = 200
    Nv = 1024

    # ── colour map: one colour per timestep
    cmap   = mcm.get_cmap("plasma", len(ts_list))
    colors = [cmap(i) for i in range(len(ts_list))]

    # ── output dir
    outdir = pjoin(data_path, "paper_plots")
    os.makedirs(outdir, exist_ok=True)

    mp.rc('font', family='sans-serif', size=10)
    mp.rc('axes', titlesize=10, labelsize=10)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    # ── reference power-law  m^{-5/2}  anchored at m=40 of first timestep
    anchor_E = None

    for idx, ts_num in enumerate(ts_list):
        phase_filename = f"phase_{ts_num}.h5"
        phase_path     = pjoin(data_path, "phase", phase_filename)

        if not os.path.exists(phase_path):
            print(f"  [SKIP] not found: {phase_path}")
            continue

        time_val = ts_num * DT_coeff * mfactor
        print(f"Processing ts={ts_num}  (t = {time_val:.3f} wpi⁻¹) …")

        E_m = hermite_spectrum(phase_path, species, drift, NC, Nv=Nv, M=M)

        label = rf"$\omega_{{pi}} t = {time_val:.1f}$"
        ax.plot(np.arange(M), E_m, lw=1.5, color=colors[idx], label=label)

        if anchor_E is None:
            anchor_E = E_m[40]   # anchor the reference slope to first curve

    # reference slope
    if anchor_E is not None:
        m_ref = np.arange(40, M)
        ax.plot(m_ref,anchor_E * (m_ref / 40)**(-3 / 2),"k:", lw=2, label=r"$m^{-5/2}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$\langle |f_m|^2 \rangle_x$")
    ax.set_title(rf"Hermite spectrum — {species}  (drift = {drift})")
    ax.legend(fontsize=8, ncol=2)

    ts_tag = "_".join(str(t) for t in ts_list)
    out    = pjoin(outdir, f"hermite_spectrum_multi_{ts_tag}.pdf")
    fig.savefig(out, dpi=300)
    plt.show()
    print(f"\nSaved: {out}")
    print("Done.")


if __name__ == "__main__":
    main()