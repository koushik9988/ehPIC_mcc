# Electrostatic 1D Particle-in-Cell (PIC) Code , ePIC++

This repository contains an electrostatic 1D Particle-in-Cell (PIC) code developed for simulating plasma systems. The code is capable of simulating basic plasma phenomena, Beam_Plasma Interactions, Monte-Carlo collision (electron-Neutral).


<p align="center">
  <img src="https://github.com/koushik9988/particle-in-cell/assets/55924787/5d278d78-2755-4293-bf18-4f8a09789b8c" height="300">
  <img src="https://github.com/user-attachments/assets/1cbc78c4-6244-4015-a1a3-4aa8835ac5db" height="300">
</p>


### Benchamrk and validation Plots

* [**Landau Damping Analysis**](https://github.com/user-attachments/files/34965185/landau.1.pdf)
* [**Upper Hybrid Dispersion**](https://github.com/user-attachments/files/34965230/disp_uh.pdf)
* [**Ion cyclotron Wave Dispersion**](https://github.com/user-attachments/files/34965215/dispersion_icw.pdf)
* [**Shape Function**](https://github.com/user-attachments/files/34965227/shap_diag.pdf)
* [**Electron Swarm Parameters**](https://github.com/user-attachments/files/34965223/swarm_plot.pdf)




## Requirements
- Python3 : Required for data processing, and data visualization. 
- python3-dev : Provides Python development headers needed for matplotlibcpp.
- GNU C++ compiler / clang (C++17 standards)
- [CMake](https://cmake.org/)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
- [matplotlibcpp](https://github.com/lava/matplotlib-cpp)
- [Git](https://git-scm.com/)
- Matplotlib
- NumPy
- Scipy

## Directory Structure

```
project_root/
├── CMakeLists.txt         # Build configuration
├── Doxyfile               # Doxygen documentation settings
├── include/               # Header files
├── src/                   # Core source files
├── linearalgebra/         # Custom matrix and solver routines
├── cross_section/         # Cross section data for different gases
├── python_scripts/        # Postprocessing and visualization scripts
├── inputfiles/            # Sample INI configuration files
├── tests/                 # Unit and performance tests
├── build/                 # Out-of-source build directory
├── data/                  # Example output and plots
├── LICENSE
└── README.md
```


## Installation (LINUX Distro)
1. Clone the repository:
    ```bash
    git clone https://github.com/koushik9988/ePIC_mcc.git
    ```

2. Navigate to the directory:
    ```bash
    cd ePIC_mcc
    ```

3. Build the code using cmake:
    ```bash
    mkdir build && cd build
    ```
    ```bash
    cmake ..                       # Normal build
    ```
    ```bash
   cmake -DENABLE_PLOTTING=ON ..   # build with runtime visualization using matplotlibcpp
    ```
    ```bash
    cmake --build .
    ```
## Installation (Windows)
1. Clone the repository:
    ```bash
    git clone https://github.com/koushik9988/ePIC_mcc.git
    ```
2. Navigate to the directory:
    ```bash
    cd ePIC_mcc
    ```

3. Build the code using cmake for MinGW:
    ```bash
    mkdir build && cd build
    ```
    ```bash
    cmake -G "MinGW Makefiles" ..          
    ```
    ```bash
    cmake --build .
    ```

#### MinGW
1. Open MSYS2 UCRT64 shell
2. Upgrade MinGW 
  ```bash
   pacman -Syu
  ```
4. Install All dependency in MinGW
```bash

pacman -S --needed \
mingw-w64-ucrt-x86_64-gcc \
mingw-w64-ucrt-x86_64-cmake \
mingw-w64-ucrt-x86_64-make \
mingw-w64-ucrt-x86_64-hdf5 \
mingw-w64-ucrt-x86_64-fftw \
mingw-w64-ucrt-x86_64-python \
mingw-w64-ucrt-x86_64-python-numpy

```
5. For runtime visualization Add Python to PATH (needed for matplotlibcpp)
```bash
export PATH="/c/Users/<YOUR_USERNAME>/AppData/Local/Programs/Python/Python312:$PATH"
```


## Running
 
```bash
./ePIC_mcc <input_file> [options]
```
 
| Option | Meaning |
|---|---|
| `--threads <n>` | number of OpenMP threads (default and upper limit: hardware threads) |
| `--restart-file <file>` | resume from a dump/phase file (positions and velocities are read from it) |
| `-h`, `--help` | help |
| `-v`, `--version` | version |
 
Examples:
 
```bash
./ePIC_mcc ../inputfiles/input.ini
./ePIC_mcc ../inputfiles/sheath/input_sheath2.ini --threads 8
./ePIC_mcc ../inputfiles/input.ini --restart-file ../datafiles/data/phase_1000.h5 --threads $(nproc)
```
 
---

# Explanation of `input.ini` File Parameters

The `input.ini` file contains parameters for configuring the simulation. Each section corresponds to a different aspect of the simulation setup.

## `[file]`

| Parameter | Description |
|----------|-------------|
| `output` | Path to directory where simulation output data in hdf5 format is saved |

## `[time]`
| Parameter | Description |
|----------|-------------|
| `NUM_TS` | Total number of time steps to run |
| `DT_coeff` | Time step as a fraction of (1/frequency): `dt = DT_coeff / ω_pe` or `dt = DT_coeff / ω_pi` |


## `[diagnostics]`
| Parameter | Description |
|----------|-------------|
| `write_interval` | Interval (in steps) to write density/field data |
| `write_interval_phase` | Interval to write phase-space data |
| `write_diagnostics` | Interval to output diagnostics in screen (energies, phase plot etc.) |
| `write_flag` | What data to write in disk: `0 = none`, `1 = all`, `2 = fields only`, `3 = phase-space only` |
| `save_fig` | Save plots as images (`1 = yes`, `0 = no`) (deprecated) |
| `sub_cycle_interval` | Frequency of ion sub-cycling  |
| `precision` | floating point precision level of diagnostics output in screen|
| `diagtype` | Type of diagnostics :`off(print just time step)`, `basic (print just time and max_phi)` ,`full (print evryting with live plot at runtime)`|



## `[visualplot]`

| Flag | Description |
|------|-------------|
| `Energy_plot` | Flag for plotting energy :`Energy_plot = 1 (on)` and `Energy_plot = 0 (off)` |
| `keflag` | Flag for kinetic energy plot `(1 = on, 0 = off)` |
| `peflag` | Flag for potential energy plot. `(1 = on, 0 = off)`|
| `teflag` | Flag for potential energy plot. `(1 = on, 0 = off)`|
| `Potentialfield_plot` | Flag for plotting potential field `(1 = on, 0 = off)`|
| `Chargedensity_plot` | Flag for plotting charge density `(1 = on, 0 = off)`|
| `phase_plot` | Flag for plotting phase space `(1 = on, 0 = off)`|
| `species_index` | Index of species to use for phase-space and density plots starting from index 0 as in species section of the input file (e.g 0 = electrons, 1 = ion etc) |
|`dft_rho`|Flag for plotting Fourier transformed charge density `(1 = onn, 0 = off)` |


## `[domain]`

| Parameter | Description |
|----------|-------------|
| `NC` | Number of grid points |
| `x0` | Origin coordinate of the domain |


## `[normalization]`

| Parameter | Description |
|-----------|-------------|
| `norm_scheme` | Type of normalization (`1 = electron scale (time and space is normalized by electron plasma frequency and debye lenghts)`, `2 = ion scale`, `3 = subcycling`, `4 = mixed scale(time is normalized by electron plasma frequency and space is normalized by ion debye length)`, `5 = user defined scale( input is given below)`) |
| `vel_norm_scheme` | Velocity normalization type (`1 = vthe ( velocity is normalized by electron thermal speed`, `2 = vthi (velocity is normalized by ion thermal speed)`, `3 = vcs(ion-acoustic speed)`) |
| `lenght_scale` | User defined Characteristic length scale |
| `time_scale` | User Defined Characteristic time scale (`1/ω_pe`) |
| `energy_scale` | User defined Characteristic energy scale (not used) |


## `[simulation]`

| Parameter | Description |
|-----------|-------------|
| `shapefunction` | Interpolation scheme: `NGP` and  `CIC` |
| `push_parallal` | Use parallel particle pushing (`1 = enabled`) |
| `deposit_parallal` | Use parallel charge deposition (`1 = enabled`) |
| `density` | Plasma density |
| `bc` | Boundary condition: `pbc` (periodic) or `open` |
| `see_rate` | Secondary electron emission rate *(not tested/used)* |
| `tempwall` | Wall temperature *(not tested/used)* |
| `ionfixed` | Fixed background ions (`1 = yes (ions donot move)`, `0 = no (ions moves)`) |


## `[solver]`

| Parameter | Description |
|----------|-------------|
| `solvertype` | Solver method: `gs`, `pcg`,`spectral` `(use spectral for periodic boundary)`|
| `tolerance` | Convergence tolerance (for iterative solvers) |
| `max_iteration` | Maximum iterations (for iterative solvers) |


## `[ExternalField]`

| Parameter | Description |
|----------|-------------|
| `B` | Magnitude of magnetic field|
| `theta` | Polar angle (`0 - PI`)|
| `azimuth` | Azimuthal angle (`0 - 2*PI`)|



## `[collision]`
| Parameter | Description |
|----------|-------------|
| `elastic` | Enable elastic collisions (`true` and `false`) |
| `excitation` | Enable excitation collisions |
| `ionization` | Enable ionization |
| `pion_elastic` | Enable ionization |
| `e_detach_collision` | Enable ionization |
| `GAS_TYPE` | GAS TYPE ( currently argon and hydrogen  is supported) |
| `GAS_DENSITY` | Neutral gas density (e.g., `1e20`) |
| `collgroup ` | particle collision group in a pair of two|

|`Collgroup defines pairs of species(for reference see [species] section below)for collision interactions using two-digit codes. For example, collgroup = 01, 21 means: first  Species with index 0 (as in coding index start from 0,1,2 etc) (e.g., electrons) will collide with the neutral gas having properties of 2nd  Species with index 1 (e.g., argon ion)and 3rd Species ( index 2) will similarly interact with the neutral gas defined by 2nd Species (index 1). This simplifies modeling by treating the gas as a fluid using the physical properties of an existing species (typically ions). To explicitly define neutral atoms (e.g., neutral argon), you can add a separate species in the [species] section with zero normalized density and dummy values for other parameters, ensuring it does not participate in the simulation as particles. Note: Neutrals are not simulated as particles but only act as background gas for MCC purposes.`|

## `[Sepcies]`

| Parameter                   | Description                                                                |
| ----------------------------| ---------------------------------------------------------------------------|
| `count`                     | Total number of species to define                                          |
| `species_X.name`            | Name of the species (e.g., `electron`, `ion`)                              |
| `species_X.mass`            | Mass of the species in kg                                                  |
| `species_X.num`             | Number of particles for the species                                        |
| `species_X.temp`            | Initial temperature in eV                                                  |
| `species_X.charge_sign`     | Charge sign (`-1` for electrons, `+1` for ions, `0` for neutrals)          |
| `species_X.normden`         | Normalized density (with respect to electron density)                      |
| `species_X.vs`              | Streaming velocity (normalized)                                            |
| `species_X.loadtype_pos`    | Particle position load type: `uniform` or other supported types    |
| `species_X.loadtype_vel`    | Particle initial vecocity load type: `uniform` or other supported types    |

|`X is the species index starting from 0. The first two species must be electron and ion respectively. Additional species (e.g., beams or negative ions) follow. Neutrals are specified with charge_sign = 0 and normden = 0 and are used only for background collisions.`|
# `Example`
```
[Species]
#number of species
count = 5

species_0.name = electron
species_0.mass = 9.10938215E-31
species_0.num = 20000
species_0.temp = 1
species_0.charge_sign = -1
species_0.normden = 1
species_0.vs = 10
species_0.loadtype_pos = uniform
species_0.loadtype_vel = 0.0sin(0)

species_1.name = ion
species_1.mass = 1.661E-27
species_1.num = 20000
species_1.temp = 0.0
species_1.charge_sign = 1
species_1.normden = 0
species_1.vs = 0
species_1.loadtype_pos = uniform
species_1.loadtype_vel = 0.0sin(0)

species_2.name = negion
species_2.mass = 1.661E-27
species_2.num = 20000
species_2.temp = 0.026
species_2.charge_sign = -1
species_2.normden = 0.5
species_2.vs = 0
species_2.loadtype_pos = uniform
species_2.loadtype_vel = 0.0sin(0)

species_3.name = beam
species_3.mass = 1.661E-27
species_3.num = 20000
species_3.temp = 0.026
species_3.charge_sign = -1
species_3.normden = 0.4
species_3.vs = 10
species_3.loadtype_pos = uniform
species_3.loadtype_vel = 0.0sin(0)

species_4.name = neutralgas
species_4.mass = 3.347E-27
species_4.num = 10
species_4.temp = 0.000
species_4.charge_sign = 0
species_4.normden = 0
species_4.vs = 0
species_4.loadtype_pos = uniform
species_4.loadtype_vel = 0.0sin(0)
```
## `Normalized density :`
`ion density , n_i0 = plasma density, so for two component electron-ion plasma n_e0 = n_i0 => 1 = n_i0/n_e0 => normalized electron density is 1 by default and normalized ion density (wrt electron) set to zero as ion density is set equal to plasma density and  so it remains fixed and doesnot change with respect to electon density. For example if our system is multicomponent and consist of 5 species as below`
```
  electron, 9.10938215E-31, 50000, 1, -1, 1, -10, uniform
  ion,6.63352090e-26,50000,0,1,0,0,uniform
  species_a,9.10938215E-31, 50000,1,-1,0.1,0,uniform
  species_b,9.10938215E-31,50000,1,1,0.3,0,uniform
  species_c,9.10938215E-31, 50000,1,-1,0.4,0,uniform
```
`now normalized density equation become` 
```
n_a + n_c + n_e0 = n_i0 + n_b
```
```
n_a/n_e0 + n_c/n_e0 + n_e0/n_e0 = n_i0/n_e0 + n_b/n_e0
```
`Let n_a/_ne0 = a , n_b/n_e0 = b and n_c/n_e0 = c then n_e0 = n_i0/(1 + a - b + c) this equation can be rewritten as`
```
n_e0 = n_i0/(1 - charge_sign_a * a - charge_sign_b * b - charge_sign_c * c) 
```
`with normalized density values : a = 0.1, b= 0.3 and c = 0.4 by taking charge sign as : species_a: -1, species_b: +1, species_c: -1. As mentioned above by default normalized electron and ion density are set to 1 and 0 respectively.
Above equation can be written like this for any number of species with different charges`

```
double k = 0;
for(int i = 0; i < species_no; i++)
{
    k += (-charge_signs[i]) * frac_densities[i];
}

// display::print(k);

normden[1] = den; // ion density
normden[0] = den / k; // electron density

for(int i = 2; i < species_no; i++)
{
    normden[i] = frac_densities[i] * normden[0];
}
```

## Boundary conditions
 
| `bc` | Left | Right | Particles |
|---|---|---|---|
| `pbc` | periodic | periodic | wrap around |
| `open` | Dirichlet `voltage_left` | Dirichlet `voltage_right` | absorbed at both walls |
| `dirneu` | Dirichlet `voltage_left` (wall/electrode) | Neumann `dphi_dx` (plasma source side) | absorbed at the left wall; right side is a **source/reflux plane** (see below) |
 
For `dirneu`:
- Particles that reach `x ≥ x0 + xL` are **re-thermalised** and sent back into the plasma (`Species::source_reemit`, flux-weighted half-Maxwellian at the species temperature, perpendicular components resampled).
- Heavy negative species (every negatively charged species except the electron) are **removed** in the last 10 % of the box (`x ≥ 0.9 xL`) so that negative ions do not accumulate at the source.
- Emitters and the plasma source are active only in this mode.
---
 
## Emitters and sources
 
> This part of the code was recently extended. There are **three independent mechanisms**, all for `bc = dirneu`.
 
### 1. Wall emitters — `[Emitters]` section (`Emitter` class)
 
Each emitter injects particles from a plane at `emitter_loc` every step. The injection mode is chosen **automatically from the keys you give**:
 
| Mode | Selected when | Particle count | Weight |
|---|---|---|---|
| **Surface (thermal emission)** | `spwt` is given and > 0 | one-sided thermal flux Γ = n·√(kT/2πm) with `n = density` | emitter's own `spwt` |
| **Current / drift** | `drift_velocity` is given (and no `spwt`) | J = n·q·v_d | species default weight |
| **Reservoir** | neither of the above | one-sided thermal flux with the species' bulk density (`normden`) | species default weight |
 
Velocities are always half-Maxwellian in the normal direction (flux-weighted, `vx = v_t √(−2 ln(1−U))`) and full Maxwellian in the two transverse directions. Each particle is advanced a random fraction of a time step so injection is not clustered on the boundary. The fractional part of the macro-particle count is injected stochastically.
 
```ini
[Emitters]
count = 1                       # number of emitters (0 disables the section)
 
emitter_0.emitter_loc  = 0      # position of the emitter plane (normalized grid units)
emitter_0.density      = 1e15   # emitter plasma density [m^-3]
emitter_0.emitter_area = 1      # emitting area [m^2]
emitter_0.species_id1  = 2      # first emitted species (index in [Species]); required
emitter_0.species_id2  = 0      # second emitted species (optional, -1 = none)
emitter_0.direction    = 1      # +1 inject toward +x, -1 toward -x, 0 = auto (left half -> +1, right half -> -1)
emitter_0.temp1        = 0.01   # injection temperature of species_id1 [eV] (<=0 -> species temperature)
emitter_0.temp2        = 0.01   # injection temperature of species_id2 [eV] (optional)
emitter_0.spwt         = 1e8    # macro-particle weight -> selects surface-emission mode
#emitter_0.drift_velocity = 1   # normalized drift speed -> selects current-injection mode
```
 
| Key | Required | Default | Description |
|---|---|---|---|
| `emitter_loc` | yes | – | emitter plane position |
| `density` | yes | – | density used in the flux/current formula |
| `emitter_area` | yes | – | emitting area |
| `species_id1` | yes | – | species index to emit |
| `species_id2` | no | `-1` | optional second species emitted from the same plane (ignored if equal to `species_id1`) |
| `direction` | no | `0` | injection direction (see above) |
| `temp1`, `temp2` | no | `-1` | injection temperatures in eV; `≤ 0` uses the species temperature |
| `spwt` | no | – | emitter-specific weight; **turns on surface-emission mode** |
| `drift_velocity` | no | – | **turns on current-injection mode** (used when `spwt` is absent) |
| `flux_weighted` | no | `1` | read for compatibility; flux-weighted sampling is always used |
| `temp`, `numparticle` | no | `0` | **deprecated**, kept for old input files |
 
Note: `domain.current_density` (used for diagnostics) is computed from the first emitter's `density` and `drift_velocity`.


## Output
 
Everything is written below `[file] output`:
 
```
<output>/
├── metadata.h5            DT_coeff, wpe, wpi, LDe, LDi, species info (read by the Python scripts)
├── field/data_<ts>.h5     potential, field, densities, ...   (every write_interval)
├── phase/phase_<ts>.h5    particle x, v, ...                  (every write_interval_phase)
├── restart dump (dump.h5) for --restart-file
└── logs/                  CSV diagnostics
    ├── energy_log.csv, ke_components_log.csv, momentum_log.csv
    ├── collision_freq_log.csv, domain_collision_freq_log.csv
    ├── electronegativity_log.csv   (ts, time, alpha = n-/ne)
    └── particle_number_log.csv
```

 ## Data processing and visualization
 1. Plot kinetic enegy ,potential enegy and total enegy
     ```bash
    python3 ke_plot.py ../name_of_outputfolder
    ```
 2. Plot dispersion
     ```bash
    python3 dispersion.py ../name_of_outputfolder
    ```
 3. Plot/Animate phase-space and potential data
     ```bash
    python3 phase_pot_plot.py ../name_of_outputfolder
    ```

## Contributors
- [Kaushik Kalita](https://github.com/koushik9988)
- [Rakesh Moulick](https://github.com/rakeshmoulick)

## Extras
 
- Keep species order **electron → ion → negion → others**; the source, recombination and α diagnostic hard-code indices 0, 1, 2.
- Several keys are read **without defaults** and the run aborts if missing: all of `[time]`, `[diagnostics]`, `[domain]`, `[normalization]`, `[simulation]` (including `dphi_dx`, `ionfixed`), `[solver]` (including `solver_state`), `[ExternalField]` (including `E_ext`), `[boundary_potential]`, `[collision]`, `[visualplot]`. Copy a working file from `inputfiles/` as a template — note `input_run_negion_emitter.ini` predates these keys and lacks `dphi_dx`, `solver_state` and `E_ext` (and `[visualplot]`-style additions), so add them before use.
- `dphi_dx` and `x0` are parsed as integers.
- For emitter runs, `spwt` should keep the macro-particle count manageable: the injected count per step is `n·√(kT/2πm)·A·Δt / spwt`.
- Set `GAS_DENSITY = 0` to switch off all collisions (cross-section files are still loaded from `../cross_section`).
- Use `--threads 1` to force the serial push/deposit path (useful for debugging reproducibility).
 
  



