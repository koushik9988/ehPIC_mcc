"""
delta_omega_nl.py
=================
Nonlinear frequency shift analysis for electrostatic plasma simulations.

Pipeline:
  1. Extract  E(x,t)  from HDF5
  2. Filter   E_k(t)  for a target k-mode via spatial FFT
  3. Analyse  ω(t)    via STFT (spectrogram) OR Hilbert instantaneous frequency
  4. Subtract δω_NL(t) = ω_measured(t) − ω_linear(k)

Usage
-----
    python delta_omega_nl.py <path_to_data> [options]

Options
-------
    --k          Target k-mode value (default: auto-detect dominant peak)
    --method     Frequency method: 'hilbert' or 'stft'  (default: hilbert)
    --wlinear    Override linear frequency ω_L manually (default: auto-estimated)
    --window     STFT window length in snapshots (default: Nt//8)
    --overlap    STFT overlap fraction 0–1  (default: 0.75)
    --ylim       y-axis limits for δω plot, e.g.  --ylim -0.05 0.05
"""

import os
import sys
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import h5py

from os.path       import join as pjoin
from scipy.constants import value as constants
from scipy.signal  import hilbert, spectrogram, find_peaks
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path',        help='Directory containing metadata.h5, field/, phase/')
    p.add_argument('--k',         type=float, default=None,
                   help='Target k-mode (normalised units)')
    p.add_argument('--method',    choices=['hilbert', 'stft'], default='hilbert',
                   help='Instantaneous-frequency method')
    p.add_argument('--wlinear',   type=float, default=None,
                   help='Known linear frequency ω_L (override auto-estimate)')
    p.add_argument('--window',    type=int,   default=None,
                   help='STFT window length [snapshots]')
    p.add_argument('--overlap',   type=float, default=0.75,
                   help='STFT overlap fraction')
    p.add_argument('--ylim',      type=float, nargs=2, default=None,
                   metavar=('LO', 'HI'))
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  HDF5 loader  — matches modes.py exactly
# ─────────────────────────────────────────────────────────────────────────────
 
def load_simulation(path):
    print(f"Loading simulation data from: ")

    eps0 = constants('electric constant')
    e    = constants('elementary charge')

    # ── metadata ────────────────────────────────────────────────────────────
    meta_file = os.path.join(path, "metadata.h5")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"metadata.h5 not found in {path}")

    with h5py.File(meta_file, 'r') as f_meta:
        metadata_group = f_meta['/metadata']

        NC                   = metadata_group.attrs['NC']
        NUM_TS               = metadata_group.attrs['NUM_TS']
        write_interval       = metadata_group.attrs['write_int']
        write_interval_phase = metadata_group.attrs['write_int_phase']
        DT_coeff             = metadata_group.attrs['DT_coeff']
        wpe                  = metadata_group.attrs['wpe']
        wpi                  = metadata_group.attrs['wpi']
        LDe                  = metadata_group.attrs['LDe']
        LDi                  = metadata_group.attrs['LDi']
        normscheme           = metadata_group.attrs['norm_scheme']
        n0                   = metadata_group.attrs['density']

    # ── normalisation (mirrors modes.py exactly) ────────────────────────────
    mfactor = wpi / wpe

    mFactor = 1          # default (normscheme 1 or 2)
    L = LDe
    W = wpe

    if normscheme in [2, 4]:
        L = LDi
        W = wpi

    DT = DT_coeff * (1.0 / W)

    # ── field files ─────────────────────────────────────────────────────────
    field_dir = os.path.join(path, "field")
    if not os.path.isdir(field_dir):
        raise FileNotFoundError(f"field directory not found in {path}")

    efield_norm = (n0 * e * L) / eps0

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
            electric_field_data.append(data[:, 1])  # column 1 = E-field value

    EF = np.vstack(electric_field_data)   # (Nt, Nx)
    Nt, Nx = EF.shape

    x  = np.linspace(0, NC, Nx, endpoint=False)
    dx = x[1] - x[0]

    time_arr = np.arange(Nt) * DT_coeff * write_interval * mfactor

    meta = dict(n0=n0, NC=NC,
                wpi=wpi, wpe=wpe, LDe=LDe, LDi=LDi,
                mFactor=mfactor,
                DT_coeff=DT_coeff, write_int=write_interval,
                write_int_phase=write_interval_phase,
                normscheme=normscheme, efield_norm=efield_norm)

    return EF, time_arr, x, dx, meta


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 – Spatial FFT → isolate k-mode
# ─────────────────────────────────────────────────────────────────────────────

def spatial_fft(EF, dx, NC, norm=True):
    """
    Returns:
        k_modes   : positive k values (Nx//2,)
        Fk_pos    : complex mode amplitudes (Nt, Nx//2)
        energy    : |Fk|² per mode per snapshot (Nt, Nx//2)
    """
    Nt, Nx = EF.shape
    Fk     = np.fft.fft(EF, axis=1, norm='ortho')
    kfft   = 2 * np.pi * np.fft.fftfreq(NC, dx)

    k_modes = kfft[:NC // 2]
    Fk_pos  = Fk[:, :NC // 2] * (2 / np.sqrt(NC))   # one-sided normalisation

    energy  = np.abs(Fk_pos) ** 2
    return k_modes, Fk_pos, energy


def select_k_mode(k_modes, energy, k_target=None):
    """
    If k_target is given, return the closest index.
    Otherwise auto-detect the dominant peak in time-averaged spectrum.
    """
    avg = np.mean(energy, axis=0)
    if k_target is not None:
        idx = int(np.argmin(np.abs(k_modes - k_target)))
        print(f"  k_target = {k_target:.5f}  →  k_actual = {k_modes[idx]:.5f}")
    else:
        peaks, _ = find_peaks(avg, height=np.max(avg) * 0.05, distance=2)
        if len(peaks) == 0:
            idx = int(np.argmax(avg))
        else:
            idx = peaks[np.argmax(avg[peaks])]      # strongest peak
        print(f"  Auto-selected dominant mode k = {k_modes[idx]:.5f}")
    return idx


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3a – Hilbert instantaneous frequency
# ─────────────────────────────────────────────────────────────────────────────

def hilbert_inst_freq(Ek_t, time_arr):
    signal   = np.real(Ek_t)
    analytic = hilbert(signal)
    phase    = np.unwrap(np.angle(analytic))
    omega_t  = np.gradient(phase, time_arr)
    omega_t  = uniform_filter1d(omega_t, size=max(3, len(omega_t) // 50))
    return omega_t


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3b – STFT (spectrogram) peak frequency
# ─────────────────────────────────────────────────────────────────────────────

def stft_inst_freq(Ek_t, time_arr, nperseg=None, overlap=0.75):
    Nt = len(Ek_t)
    dt = (time_arr[-1] - time_arr[0]) / (Nt - 1)

    if nperseg is None:
        nperseg = max(16, Nt // 8)

    noverlap = int(nperseg * overlap)
    signal   = np.real(Ek_t)

    f_stft, t_stft, Sxx = spectrogram(signal, fs=1.0/dt,
                                       nperseg=nperseg,
                                       noverlap=noverlap,
                                       window='hann',
                                       scaling='spectrum')

    omega_axis  = 2 * np.pi * f_stft
    t_stft_phys = time_arr[0] + t_stft * dt

    peak_idx = np.argmax(Sxx, axis=0)
    omega_t  = omega_axis[peak_idx]

    return t_stft_phys, omega_t, f_stft, t_stft, Sxx


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 – Linear frequency estimate  ω_L(k)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_omega_linear(Ek_t, time_arr, omega_override=None):
    if omega_override is not None:
        print(f"  Using supplied ω_L = {omega_override:.5f}")
        return float(omega_override)

    Nt = len(Ek_t)

    wspec   = np.fft.fft(np.real(Ek_t))
    freqs   = 2 * np.pi * np.fft.fftfreq(Nt, d=(time_arr[1] - time_arr[0]))
    pos     = slice(1, Nt // 2)
    omega_L = freqs[pos][np.argmax(np.abs(wspec[pos]))]

    n_early = Nt // 4
    if np.std(np.real(Ek_t[:n_early])) > 1e-20:
        wspec_e = np.fft.fft(np.real(Ek_t[:n_early]))
        freqs_e = 2 * np.pi * np.fft.fftfreq(n_early,
                                               d=(time_arr[1] - time_arr[0]))
        pos_e   = slice(1, n_early // 2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            omega_early = freqs_e[pos_e][np.argmax(np.abs(wspec_e[pos_e]))]
        if 0.5 * abs(omega_L) < abs(omega_early) < 2.0 * abs(omega_L):
            omega_L = omega_early

    print(f"  Estimated ω_L = {omega_L:.5f}")
    return float(omega_L)


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_mpl():
    mp.rc('text',   usetex=False)
    mp.rc('font',   family='sans-serif', size=10)
    mp.rc('axes',   titlesize=10, labelsize=10)
    mp.rc('xtick',  labelsize=10)
    mp.rc('ytick',  labelsize=10)
    mp.rc('legend', fontsize=9)

FIGSIZE_MM = np.array([160, 160 / 1.618])
DPI_SAVE   = 300
DPI_SCREEN = int(np.sqrt(1920**2 + 1200**2) / 24)


def fig_in(mm=FIGSIZE_MM):
    return (mm / 25.4, DPI_SCREEN)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    path = os.path.abspath(args.path.strip())
    setup_mpl()

    path_out = os.path.join(path, "analysis")
    os.makedirs(path_out, exist_ok=True)

    # ── 1. Load ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading simulation data …")
    EF, time_arr, x, dx, meta = load_simulation(path)
    Nt, Nx = EF.shape
    print(f"  Shape: Nt={Nt}, Nx={Nx}")
    print(f"  Time range: [{time_arr[0]:.2f}, {time_arr[-1]:.2f}] ω_pi·t")

    # ── 2. Spatial FFT + mode selection ─────────────────────────────────────
    print("\n[2/4] Spatial FFT & mode selection …")
    k_modes, Fk_pos, energy = spatial_fft(EF, dx, int(meta['NC']))
    k_idx = select_k_mode(k_modes, energy, k_target=args.k)
    k_sel = k_modes[k_idx]

    Ek_t = Fk_pos[:, k_idx]

    # ── 3. Instantaneous frequency ──────────────────────────────────────────
    print(f"\n[3/4] Estimating ω(t) via '{args.method}' …")

    if args.method == 'hilbert':
        omega_t  = hilbert_inst_freq(Ek_t, time_arr)
        t_omega  = time_arr
        stft_pkg = None

    else:   # stft
        nperseg = args.window if args.window else max(16, Nt // 8)
        t_omega, omega_t, f_stft, t_stft, Sxx = stft_inst_freq(
            Ek_t, time_arr, nperseg=nperseg, overlap=args.overlap)
        stft_pkg = (f_stft, t_stft, Sxx, time_arr)

    # ── 4. Linear frequency & nonlinear shift ───────────────────────────────
    print("\n[4/4] Computing δω_NL(t) …")
    omega_L     = estimate_omega_linear(Ek_t, time_arr, omega_override=args.wlinear)
    delta_omega = omega_t - omega_L

    # ─────────────────────────────────────────────────────────────────────────
    #  FIGURES
    # ─────────────────────────────────────────────────────────────────────────
    fw, fd = fig_in()

    # ── Figure 1 : Mode energy  +  ω(t) ─────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=fw, dpi=fd,
                             constrained_layout=True, sharex=False)

    ax0 = axes[0]
    Ek_energy = np.abs(Ek_t) ** 2
    Ek_log    = np.where(Ek_energy > 0, np.log10(Ek_energy), np.nan)
    ax0.plot(time_arr, Ek_log, color='steelblue', lw=1.2)
    ax0.set_ylabel(r"$\log_{10}|E_k|^2$")
    ax0.set_xlabel(r"$\omega_{pi} t$")
    ax0.set_title(f"Mode energy  $k = {k_sel:.4f}$", pad=4)
    ax0.grid(alpha=0.4)

    ax1 = axes[1]
    ax1.plot(t_omega, omega_t, color='darkorange', lw=1.2, label=r"$\omega(t)$")
    ax1.axhline(omega_L, color='black', lw=1.0, ls='--',
                label=rf"$\omega_L = {omega_L:.4f}$")
    ax1.set_ylabel(r"$\omega$ (norm.)")
    ax1.set_xlabel(r"$\omega_{pi} t$")
    ax1.set_title(r"Instantaneous frequency $\omega(t)$", pad=4)
    ax1.legend(loc='upper right', framealpha=0.5)
    ax1.grid(alpha=0.4)

    fig.savefig(pjoin(path_out, "omega_vs_time.png"), dpi=DPI_SAVE)
    print("  Saved → omega_vs_time.png")

    # ── Figure 2 : δω_NL(t) ─────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=fw, dpi=fd, constrained_layout=True)

    ax2.plot(t_omega, delta_omega, color='crimson', lw=1.2)
    ax2.axhline(0, color='k', lw=0.8, ls=':')
    ax2.set_xlabel(r"$\omega_{pi} t$")
    ax2.set_ylabel(r"$\delta\omega_{NL}(t) = \omega(t) - \omega_L$")
    ax2.set_title(rf"Nonlinear frequency shift  $k = {k_sel:.4f}$", pad=4)
    if args.ylim:
        ax2.set_ylim(args.ylim)
    ax2.grid(alpha=0.4)

    fig2.savefig(pjoin(path_out, "delta_omega_nl.png"), dpi=DPI_SAVE)
    print("  Saved → delta_omega_nl.png")

    # ── Figure 3 : STFT spectrogram (only for stft method) ──────────────────
    if args.method == 'stft' and stft_pkg is not None:
        f_stft, t_stft_raw, Sxx, time_arr_full = stft_pkg
        dt_snap     = time_arr_full[1] - time_arr_full[0]
        t_stft_phys = time_arr_full[0] + t_stft_raw * dt_snap
        omega_axis  = 2 * np.pi * f_stft

        fig3, ax3 = plt.subplots(figsize=fw, dpi=fd, constrained_layout=True)
        pcm = ax3.pcolormesh(t_stft_phys, omega_axis,
                             np.log10(Sxx + 1e-30),
                             shading='auto', cmap='plasma')
        ax3.axhline(omega_L, color='white', lw=1.0, ls='--',
                    label=rf"$\omega_L = {omega_L:.4f}$")
        ax3.set_xlabel(r"$\omega_{pi} t$")
        ax3.set_ylabel(r"$\omega$ (norm.)")
        ax3.set_title(rf"STFT spectrogram  $k = {k_sel:.4f}$", pad=4)
        ax3.legend(loc='upper right', framealpha=0.5)
        plt.colorbar(pcm, ax=ax3, label=r"$\log_{10} S$")

        fig3.savefig(pjoin(path_out, "stft_spectrogram.png"), dpi=DPI_SAVE)
        print("  Saved → stft_spectrogram.png")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n─── Summary ─────────────────────────────────────────")
    print(f"  Selected mode       k  = {k_sel:.5f}")
    print(f"  Linear frequency  ω_L  = {omega_L:.5f}")
    print(f"  δω_NL  mean            = {np.nanmean(delta_omega):.5f}")
    print(f"  δω_NL  std             = {np.nanstd(delta_omega):.5f}")
    print(f"  δω_NL  max |shift|     = {np.nanmax(np.abs(delta_omega)):.5f}")
    print("─────────────────────────────────────────────────────\n")

    plt.show()


if __name__ == '__main__':
    main()