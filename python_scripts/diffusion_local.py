"""
@kaushik kalita
@Date : 16/02/2025
"""
import numpy as np
import h5py
import sys
import os
from os.path import join as pjoin
from scipy.constants import value as constants
from scipy.optimize import curve_fit

#----------------constants--------------------
eps0 = constants('electric constant')
kb = constants('Boltzmann constant')
me = constants('electron mass')
AMU = constants('atomic mass constant')
e = constants('elementary charge')
#----------------------------------------------

if len(sys.argv) != 3:
    print("Usage: python diffusion_local.py <path_to_data> <particle_type>")
    sys.exit(1)

base_path = os.path.abspath(sys.argv[1].strip())
particle_type = sys.argv[2].strip()

output_dir = 'diffusion_analysis'
phase_dir = os.path.join(base_path, "phase")
meta_file = os.path.join(base_path, "metadata.h5")
path_fig = os.path.join(base_path,"analysis", output_dir)

os.makedirs(path_fig, exist_ok=True)

print("\nBase path :", base_path)
print("Species   :", particle_type)
print("Phase dir :", phase_dir)

#--- Check paths ---
if not os.path.exists(meta_file):
    print("metadata.h5 not found")
    sys.exit(1)

if not os.path.isdir(phase_dir):
    print("phase directory not found")
    sys.exit(1)

#---------------------Read Metadata-----------------------------------------
f_meta = h5py.File(meta_file, 'r')
metadata_group = f_meta['/metadata']

NC = metadata_group.attrs['NC']
NUM_TS = metadata_group.attrs['NUM_TS']
write_interval_field = metadata_group.attrs['write_int']
write_interval_phase = metadata_group.attrs['write_int_phase']
DT_coeff = metadata_group.attrs['DT_coeff']
wpe = metadata_group.attrs['wpe']
wpi = metadata_group.attrs['wpi']
LDe = metadata_group.attrs['LDe']
LDi = metadata_group.attrs['LDi']
normscheme = metadata_group.attrs['norm_scheme']
density = metadata_group.attrs['density']

species_metadata_path = f'/metadata_species/{particle_type}'
metadata_species = f_meta[species_metadata_path]

species_mass = metadata_species.attrs['mass']
species_charge = metadata_species.attrs['charge']
species_temp = metadata_species.attrs['temperature']
species_spwt = metadata_species.attrs['spwt']
species_num = metadata_species.attrs['num_particles']



#--------- Normalization-----------------------------
mfactor = wpi/wpe
if normscheme == 2 or normscheme == 1:
    mfactor = 1
print("mfactor:", mfactor)

L = LDe
W = wpe

if normscheme == 2 or normscheme == 4:
    L = LDi
    W = wpi

DT = DT_coeff * (1.0 / W)

f_meta.close()
# ------Collect phase .h5 files--------------------
phase_files = []

for fname in os.listdir(phase_dir):
    if fname.startswith("phase_") and fname.endswith(".h5"):
        ts_num = int(fname[6:-3])
        phase_files.append((ts_num, fname))

phase_files.sort()

print(f"Found {len(phase_files)} phase files")

if not phase_files:
    print("No phase files found")
    sys.exit(1)

nt = len(phase_files)

dt_phase = DT_coeff * write_interval_phase * mfactor


# ------------------- Load Particle Data -------------------
print("Loading particle data...")

vel_list = []
for ts_num, fname in phase_files:
    phase_file_path = os.path.join(phase_dir, fname)
    with h5py.File(phase_file_path, 'r') as phase_f:
        data_phase = phase_f[f"{particle_type}/phase"][:]
        v_data = data_phase[:, 1].astype(float)  # Column 1 is velocity
        vel_list.append(v_data)

vel = np.array(vel_list)
nt, npart = vel.shape

print(f"Loaded {nt} phase-snapshots with {npart} particles each.")

# ------------------- Velocity Binning -------------------
#v_flat = vel.flatten()

#vmin = np.percentile(v_flat, 5)
#vmax = np.percentile(v_flat, 95)
#nv_bins = 50

#v_edges = np.linspace(vmin, vmax, nv_bins + 1)
#v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

# ---------------- Define Analysis Time Window ----------------
t_start_index = 2000 #in wpe/wpi*t unit
t_end_index = 5000 #in wpe/wpi*t unit

t0_idx = int(round(t_start_index / dt_phase))
n_ti = int(round(t_end_index / dt_phase))

# Ensure indices are within bounds
t0_idx = max(0, min(t0_idx, nt - 1))
n_ti = min(n_ti, nt - t0_idx)

###
# ------------------- Velocity Binning -------------------
# ------------------- Velocity Binning -------------------
nv_bins = 50

v_flat_list = []
for idx in range(t0_idx, t0_idx + n_ti, 10):  # every 10th snapshot within analysis window
    v_flat_list.append(vel[idx, :])

v_flat = np.concatenate(v_flat_list)
vmin, vmax = np.percentile(v_flat, [1, 99])

v_edges = np.linspace(vmin, vmax, nv_bins + 1)
v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

# Use velocities at analysis start time (t0_idx), not all timesteps
#v_flat = vel[t0_idx, :]

#vmin = np.percentile(v_flat, 5)
#vmax = np.percentile(v_flat, 95)
#nv_bins = 50

#v_edges = np.linspace(vmin, vmax, nv_bins + 1)
#v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
#--------------------------------------------------------------
###

print(f"Diffusion Analysis Time Window: Start Index = {t0_idx}")
print(f"Diffusion Analysis Time Window: End Index = {t0_idx + n_ti}")


# Define Lag Steps (Tau) in terms of phase-file indices (like phase_index.h5 here "index" is the time index of the phase snapshot, not the physical time)
part1 = np.arange(1, 6) #small lags
part2 = np.arange(6, 31, 2) #medium lags
#part3 = np.arange(32, 101, 8) #large lags
tau_steps = np.unique(np.concatenate([part1, part2]))
print(f"Tau steps (in phase-file index units): {tau_steps}")
#tau_steps = np.array([2,4,6,8,10,12])
#tau_steps = np.unique(np.concatenate([part1, part2, part3]))
tau_phys = tau_steps * dt_phase

# ---- Decorrelation time window for independent sampling ----
# The maximum lag used in any Δv measurement is tau_max steps. A particle
# selected as a "starting" particle at time ti and again at ti+1 produces
# two Δv samples that share up to (tau_max - 1) future snapshots in common.
# To guarantee statistical independence between starting times we must
# advance ti by at least tau_max steps so that NO snapshot is shared between
# consecutive (ti, ti+ decorr_window) measurement windows.
#
# decorr_window = tau_max (in units of phase-file index steps)
#
# This is the minimum safe decorr_window.  It may be increased further if we have long runs.

tau_max = int(tau_steps[-1])          # largest lag in index units
decorr_window = tau_max               # non-overlapping windows decorr gap
print(f"Decorrelation window : {decorr_window} phase-file steps  "f"({decorr_window * dt_phase:.4f} simulation time units)")

# Create range of independent starting indices.
# Each window [ti, ti+tau_max] must not overlap the next [ti+decorr_window, ti+decorr_window+tau_max].
# With decorr_window = tau_max the windows are exactly non-overlapping, guaranteeing that
# NO snapshot is reused across consecutive starting times -> independent dv samples.
# ti_max_start ensures the largest tau is always completable (ti + tau_max < nt).

ti_max_start = nt - tau_max - 1
ti_idxs = np.arange(t0_idx,min(t0_idx + n_ti, ti_max_start + 1),decorr_window)

if len(ti_idxs) == 0:
    print("ERROR: No valid starting indices after applying decorrelation decorr_window.")
    print(f"       Need at least {tau_max + 1} phase snapshots in the analysis window.")
    print(f"       Current window has {n_ti} snapshots. Reduce tau_steps or extend n_ti.")
    sys.exit(1)

print(f"Independent starting times (ti_idxs): {len(ti_idxs)}  "f"(decorr_window={decorr_window}, window=[{ti_idxs[0]}, {ti_idxs[-1]}])")

# ---------------- Initialize Storage Arrays ----------------
D1 = np.zeros(nv_bins)
D2 = np.zeros(nv_bins)
D3 = np.zeros(nv_bins)
D2_from_sigma = np.zeros(nv_bins)

n_pdf_bins = 100
all_var = []

all_pdfs = np.zeros((nv_bins, len(tau_steps), n_pdf_bins))
all_sigma_fit = np.zeros((nv_bins, len(tau_steps)))
all_mu_fit = np.zeros((nv_bins, len(tau_steps)))
all_amp_fit = np.zeros((nv_bins, len(tau_steps))) 
all_D2_from_sigma_tau = np.zeros((nv_bins, len(tau_steps)))
all_sigma_msd = np.zeros((nv_bins, len(tau_steps)))

pdf_centers_storage = [] 

# Number of points to use for linear slope fitting
Nfit = len(tau_steps) // 4
if Nfit < 6:
    Nfit = 6

#Gaussian function for curve_fit
def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

#--------------main loop---------------
print("Processing velocity bins...")


# Total 3 loops runs here
# 1st loop: velocity bins
# 2nd loop: independent starting times for Δv sampling
# 3rd loop: tau steps for statistics and PDF fitting
# And to make each ti start time independent we have a decorrelation window (decorr_window) 
# that ensures no shared snapshots between consecutive ti's. Each ti produces 
# many Δv samples across all taus, which are then used to compute PDFs and 
# fit diffusion coefficients for that velocity bin.



# 1st loop for bins
for ib in range(nv_bins):
    if ib % 10 == 0:
        print(f"Processing bin {ib}/{nv_bins}")
    
    v_low = v_edges[ib]
    v_high = v_edges[ib + 1]

    # Dictionary to hold velocity differences for each tau
    dv_per_tau = {}
    for tau in tau_steps:
        dv_per_tau[tau] = []

    # ti_idxs is pre-clipped to [t0_idx, ti_max_start] with decorr_window=tau_max,
    # so ti + tau_max < nt is guaranteed for every ti in ti_idxs.
    # These starting times are also separated by decorr_window >= tau_max, ensuring
    # NO snapshot index is shared between consecutive measurement windows.
    # 2nd loop for independent starting times
    for ti in ti_idxs:
        # Identify particles in this bin at time ti 
        in_bin_mask = (vel[ti] >= v_low) & (vel[ti] < v_high)
        idxs = np.flatnonzero(in_bin_mask)

        #checks
        if idxs.size == 0:
            continue

        for tau in tau_steps:
            t2 = ti + tau
            # t2 < nt is guaranteed by ti_max_start construction, but keep
            # guard for safety in case tau_steps is later extended manually.
            if t2 >= nt:
                continue

            #Δv = v(t+τ) − v(t)
            v_final   = vel[t2, idxs]
            v_initial = vel[ti, idxs]
            dv = v_final - v_initial

            dv_per_tau[tau].append(dv)

    # Flatten data for range determination
    all_dv_samples_for_bin = []
    for tau in tau_steps:
        if len(dv_per_tau[tau]) > 0:
            concatenated_dv = np.concatenate(dv_per_tau[tau])
            all_dv_samples_for_bin.append(concatenated_dv)

    if len(all_dv_samples_for_bin) == 0:
        continue

    all_dv_concat = np.concatenate(all_dv_samples_for_bin)
    
    # Establish PDF Bins for this vecloity bin based on the range of Δv values observed across all taus
    dv_min = np.min(all_dv_concat)
    dv_max = np.max(all_dv_concat)
    dv_range = dv_max - dv_min
    
    buffer_width = 0.05 * dv_range
    
    pdf_bins = np.linspace(dv_min - buffer_width, dv_max + buffer_width, n_pdf_bins + 1)
    pdf_bin_centers = 0.5 * (pdf_bins[:-1] + pdf_bins[1:])
    
    pdf_centers_storage.append(pdf_bin_centers)

    msd_list = []
    var_list = []
    
    #3rd loop for tau steps to calculate statistics and fit PDFs
    for i_tau, tau in enumerate(tau_steps):
        dv_list = dv_per_tau[tau]
        
        if len(dv_list) == 0:
            continue
                 
        dv_cat = np.concatenate(dv_list)
        
        # Calculate MSD and VAR
        msd_val = np.mean(dv_cat**2)
        var_val = np.var(dv_cat)
        
        msd_list.append(msd_val)
        var_list.append(var_val)
        
        # Calculate Histogram
        hist_vals, _ = np.histogram(dv_cat, bins=pdf_bins, density=True)
        all_pdfs[ib, i_tau, :] = hist_vals
        
        # Find the Peak (Mode) index
        peak_idx = np.argmax(hist_vals)
        
        mu0 = pdf_bin_centers[peak_idx] 
        A0 = hist_vals[peak_idx]
        sigma0 = np.std(dv_cat) 
        
        p0 = [A0, mu0, sigma0]

        # Define Bounds
        lower_bounds = [0, pdf_bins[0], 1e-12]
        upper_bounds = [np.inf, pdf_bins[-1], np.inf]
        bounds_tuple = (lower_bounds, upper_bounds)

        # Perform Fit
        try:
            popt, pcov = curve_fit(gauss, pdf_bin_centers, hist_vals, p0=p0, bounds=bounds_tuple, maxfev=8000)

            A_fit, mu_f, sigma_f = popt
            all_amp_fit[ib, i_tau] = A_fit
            all_mu_fit[ib, i_tau] = mu_f
            all_sigma_fit[ib, i_tau] = abs(sigma_f)
            tau_val = tau_phys[i_tau]
            all_D2_from_sigma_tau[ib, i_tau] = (sigma_f**2) / (2.0 * tau_val)

        except RuntimeError:
            all_amp_fit[ib, i_tau] = np.nan
            all_mu_fit[ib, i_tau] = np.nan
            all_sigma_fit[ib, i_tau] = np.nan
            all_D2_from_sigma_tau[ib, i_tau] = np.nan
            continue

    # Store Variance Array
    all_var.append(np.array(var_list))

    msd_array = np.array(msd_list)
    var_array = np.array(var_list)

    sigma_msd = np.sqrt(msd_array)
    all_sigma_msd[ib, :] = sigma_msd

    # 1) Diffusion from VAR slope:  Var = 2Dτ + C
    if len(var_array) >= Nfit:
        slope_var, intercept_var = np.polyfit(tau_phys[:Nfit], var_array[:Nfit], 1)
        D_var = slope_var / 2.0
        D1[ib] = D_var
    else:
        D1[ib] = np.nan

    # 2) Diffusion from MSD:  MSD = 2Dτ + C
    if len(msd_array) >= Nfit:
        slope_msd, intercept_msd = np.polyfit(tau_phys[:Nfit], msd_array[:Nfit], 1)
        D_msd = slope_msd / 2.0
        D2[ib] = D_msd
    else:
        D2[ib] = np.nan

    # 3) Diffusion from sigma² using curve-fit: σ² = 2Dτ + C
    sigma_fit_arr = all_sigma_fit[ib, :Nfit]
    sigma2 = sigma_fit_arr**2
    
    # Remove NaN values before fitting
    valid_mask = ~np.isnan(sigma2)
    if np.sum(valid_mask) >= 2:
        slope_sigma, intercept_sigma = np.polyfit(tau_phys[:Nfit][valid_mask], sigma2[valid_mask], 1)
        D_sigma = slope_sigma / 2.0
        D3[ib] = D_sigma
        D2_from_sigma[ib] = D_sigma
    else:
        D3[ib] = np.nan
        D2_from_sigma[ib] = np.nan

print("Velocity bin processing complete.")

# ----------------- SAVE TO HDF5 -----------------
save_h5 = pjoin(path_fig, f"local_velocity_diffusion_{particle_type}.h5")

print(f"\nSaving results to: {save_h5}")

with h5py.File(save_h5, "w") as file:
    # ---------------- Metadata ----------------
    metadata = file.create_group("metadata")
    metadata.attrs["particle_type"] = particle_type
    metadata.attrs["nv_bins"] = nv_bins
    metadata.attrs["n_tau"] = len(tau_steps)
    metadata.attrs["dt_phase"] = dt_phase
    metadata.attrs["analysis_window_start"] = t0_idx
    metadata.attrs["analysis_window_end"] = t0_idx + n_ti
    metadata.attrs["Nfit"] = Nfit
    metadata.attrs["DT_phase"] = dt_phase
    metadata.attrs["decorr_window"] = decorr_window  #Decorrelation decorr_window (index steps)
    metadata.attrs["tau_max"] = tau_max              # max lag used (index steps)
    metadata.attrs["n_independent_ti"] = len(ti_idxs)  # number of independent starts

    # ---------------- Axes info for plotting later ----------------
    axes = file.create_group("axes")
    axes.create_dataset("v_edges", data=v_edges)
    axes.create_dataset("v_centers", data=v_centers)
    axes.create_dataset("tau_steps", data=tau_steps)
    axes.create_dataset("tau_phys", data=tau_phys)

    # Each velocity bin has its own Δv axis
    pdf_axes = axes.create_group("pdf_centers")
    for ib in range(nv_bins):
        if ib < len(pdf_centers_storage):
            pdf_axes.create_dataset(f"bin_{ib}", data=pdf_centers_storage[ib])

    # ---------------- PDFs ----------------
    pdf = file.create_group("pdfs")
    pdf.create_dataset("P_dv_tau_v", data=all_pdfs, compression="gzip")

    # ---------------- Gaussian Fits ----------------
    fit = file.create_group("gaussian_fit")
    fit.create_dataset("A", data=all_amp_fit)
    fit.create_dataset("mu", data=all_mu_fit)
    fit.create_dataset("sigma", data=all_sigma_fit)

    # ---------------- Diffusion from sigma(τ) ----------------
    Dsig = file.create_group("diffusion_from_sigma_tau")
    Dsig.create_dataset("D_tau", data=all_D2_from_sigma_tau)

    # ---------------- Final Local Diffusion Coefficients ----------------
    D = file.create_group("local_diffusion")
    D.create_dataset("D_from_var", data=D1)
    D.create_dataset("D_from_msd", data=D2)
    D.create_dataset("D_from_sigma", data=D2_from_sigma)

    # ---------------- Variance / MSD ----------------
    stats = file.create_group("statistics")
    stats.create_dataset("variance", data=np.array(all_var))
    stats.create_dataset("sigma_from_msd", data=all_sigma_msd)
