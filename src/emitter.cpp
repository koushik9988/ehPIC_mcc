/*
function name all small letter with clear readble name use underscore
*/

#include "emitter.h"
#include "domain.h"
#include "init.h"
#include "species.h"
#include <cmath>

Emitter::Emitter(const EmitterParams &params, Domain &domain) : params(params), domain(domain) {}

// ---------------------------------------------------------------
//  How many macro-particles to inject at current timestep
//  current I = |J| * area = (real_particle * |charge|) / time   
//  real_particles/s = |J| * area / |charge|
//  macro_particles/s = real_particles/s / spwt
//  macro_particles at current dt = macro_particles/s * dt
// ---------------------------------------------------------------

int Emitter::compute_num_inject(double J, double charge, double spwt) const
{
    double real_per_sec   = std::abs(J) * params.emitter_area / std::abs(charge);
    double macro_per_sec  = real_per_sec / spwt;
    double macro_this_dt  = macro_per_sec * (domain.DT/domain.W); // convert to second

    int    n_floor  = static_cast<int>(macro_this_dt);
    double fraction = macro_this_dt - n_floor;

    // Inject the fraction with some probability 
    return n_floor + (rnd() < fraction ? 1 : 0);
}


double Emitter::compute_current_density(double drift_vel, double charge) const
{
    double n = params.density;
    //double q = std::abs(charge);
    //J = n * q * v_d
    double current_density = n * std::abs(charge) * std::abs(drift_vel * domain.vel_norm); //in ampere/m^2
    //display::print("current density",current_density);
    return current_density;
}

double Emitter::get_current_density()
{
    return compute_current_density(params.vdrift, Const::QE);
}


void Emitter::inject(std::vector<Species> &species_list)
{
    if (params.has_spwt)        inject_surface(species_list);      // weighted thermal emission
    else if (params.has_drift)  inject_current(species_list);
    else                        inject_reservoir(species_list);
}

// inward normal: left half of domain injects +x, right half injects -x
int Emitter::inward_direction() const
{
    if (params.direction != 0) return (params.direction > 0) ? +1 : -1;
    return (params.emitter_loc <= domain.x0 + 0.5 * domain.xL) ? +1 : -1;
}

// vt = sqrt(kT/m)
// flux-weighted: pdf(vx) ~ vx exp(-vx^2/2vt^2)  -> vx = vt*sqrt(-2 ln(1 -U))   (particles crossing a plane)
double Emitter::sample_halfmaxwellian_vx(Species &sp, double temp_eV, double vt) const
{
    
    return vt * std::sqrt(-2.0 * std::log(1.0 - rnd()));
   
}

// one-sided flux of a stationary Maxwellian: Gamma = n * sqrt(kT/(2 pi m))
double Emitter::reservoir_flux(double sigma, double density) const
{
    //return params.density * sigma / std::sqrt(2.0 * Const::PI);
    return density * sigma / std::sqrt(2.0 * Const::PI);
}

// Create n particles at the emitter boundary moving in direction `dir`
// vx: half-Maxwellian, vy,vz: full Maxwellian via Init::SampleVel.  Each particle is advanced by a random
// fraction of a time step so injection is not artificially clustered on the boundary.
void Emitter::inject_halfmaxwellian(Species &sp, int n, double temp_eV, int dir, double spwt)
{
    if (n <= 0) return;
    if (temp_eV <= 0) temp_eV = sp.temp;

    const double vt   = std::sqrt(temp_eV * Const::eV / sp.defaultmass);  // [m/s]
    const double Pos_Const = domain.vel_norm / (domain.L * domain.W);
    const double xmin = domain.x0;
    const double xmax = domain.x0 + domain.xL;

    for (int p = 0; p < n; p++)
    {
        double vx = dir * sample_halfmaxwellian_vx(sp, temp_eV, vt) / domain.vel_norm;   // normalized, inward
        double vy = Init::SampleVel(sp, temp_eV) / domain.vel_norm;
        double vz = Init::SampleVel(sp, temp_eV) / domain.vel_norm;

        double x = params.emitter_loc + rnd() * Pos_Const * vx * domain.DT;
        //double x = params.emitter_loc + Pos_Const * vx * domain.DT;
        //display::print("emitted x:",x);
        //if (x < xmin || x >= xmax) continue;

        //sp.AddParticle(Particle(x, vx, vy, vz, 0));
        sp.AddParticle(Particle(x, vx, vy, vz, 0, NAN, spwt));
    }
}


// spwt given: surface emission of a thermal one-sided flux. Gamma = n*sqrt(kT/2pi m).
// The count uses the emitter's own weight, and each particle carries that weight.
// No drift velocity: the sheath field accelerates the ions.
void Emitter::inject_surface(std::vector<Species> &species_list)
{
    const int dir = inward_direction();
    const int ids[2]      = { params.species_id1, params.species_id2 };
    const double temps[2] = { params.temp1, params.temp2 };

    for (int k = 0; k < 2; k++)
    {
        if (ids[k] < 0) continue;
        if (k == 1 && ids[1] == ids[0]) continue;

        Species &sp  = species_list[ids[k]];
        double T     = (temps[k] > 0) ? temps[k] : sp.temp;
        double sigma = std::sqrt(T * Const::eV / sp.defaultmass);

        double J = std::abs(sp.charge) * reservoir_flux(sigma, params.density);   // q*Gamma
        int    n = compute_num_inject(J, sp.charge, params.spwt);                 // stochastic rounding
        inject_halfmaxwellian(sp, n, T, dir, params.spwt);
    }
}

// drift_velocity given : number of particles from J = n q v_d , velocities half-Maxwellian
void Emitter::inject_current(std::vector<Species> &species_list)
{
    const int dir = inward_direction();
    const int ids[2]     = { params.species_id1, params.species_id2 };
    const double temps[2] = { params.temp1, params.temp2 };

    for (int k = 0; k < 2; k++)
    {
        if (ids[k] < 0) continue;
        if (k == 1 && ids[1] == ids[0]) continue;

        Species &sp = species_list[ids[k]];
        double J = compute_current_density(params.vdrift, sp.charge);
        
        
        int n = compute_num_inject(J, sp.charge, sp.defaultspwt);
        //int n = compute_num_inject(J, sp.charge, 10E8);
        //display::print("aaa",n);
        inject_halfmaxwellian(sp, n, temps[k], dir,sp.defaultspwt);
    }
}

// no drift_velocity : plasma reservoir. Particles reaching this wall are removed by the OPEN
// boundary; the reservoir re-injects the thermal flux Gamma = n*sqrt(kT/2pi m)
void Emitter::inject_reservoir(std::vector<Species> &species_list)
{
    const int dir = inward_direction();
    const int ids[2]     = { params.species_id1, params.species_id2 };
    const double temps[2] = { params.temp1, params.temp2 };

    for (int k = 0; k < 2; k++)
    {
        if (ids[k] < 0) continue;
        if (k == 1 && ids[1] == ids[0]) continue;

        Species &sp = species_list[ids[k]];
        double T     = (temps[k] > 0) ? temps[k] : sp.temp;
        double sigma = std::sqrt(T * Const::eV / sp.defaultmass);

        // reuse the stochastic rounding: J = q*Gamma  ->  computeNumInject uses |J|*A/|q|
        //double J = std::abs(sp.charge) * reservoir_flux(sigma,sp.bulkden);
        double J = std::abs(sp.charge) * reservoir_flux(sigma,sp.bulkden);
        //display::print("flux",reservoir_flux(sigma,sp.bulkden));
        int n = compute_num_inject(J, sp.charge, sp.defaultspwt);
        //display::print(" computeNumInject",n);
        inject_halfmaxwellian(sp, n, T, dir,sp.defaultspwt);
    }
}


void source_emitter(Species &ion, Species &el, Domain &domain, double emitter_area, Species *neg, double neg_frac)   // newww: neg, neg_frac added
{
    const double x_src             = domain.x0 + domain.xL;
    const double Ls                = 0.3 * domain.xL;
    const double x_lo              = x_src - Ls;

    const double gamma             = 1 * ion.bulkden * domain.IAW_vel;
    const double real_ions_this_dt = gamma * emitter_area * (domain.DT / domain.W);
    const double ion_macro_total   = real_ions_this_dt / ion.defaultspwt;
    const int    ion_floor         = static_cast<int>(std::floor(ion_macro_total));
    const int    n_pairs           = ion_floor + ((rnd() < ion_macro_total - ion_floor) ? 1 : 0);

    //(1-neg_frac) so that S+ = Se + S-
    //  electron total contribution = el_per_ion*electron_spwt = (1-neg_frac)*ion_spwt
    //  negative ion total constribution = neg_per_ion*neg_spwt = neg_frac*ion_spwt
    //  total contribition = ion_spwt
    const double el_per_ion        = (1.0 - neg_frac) * ion.defaultspwt / el.defaultspwt;   
    const double neg_per_ion       = (neg != nullptr) ? neg_frac * ion.defaultspwt / neg->defaultspwt : 0.0;
    const int    neg_floor         = static_cast<int>(std::floor(neg_per_ion));
    const double neg_fr            = neg_per_ion - neg_floor;
    const double neg_vt            = (neg != nullptr) ? std::sqrt(neg->temp * Const::eV / neg->defaultmass) : 0.0;
    const int    el_floor          = static_cast<int>(std::floor(el_per_ion));
    const double el_frac           = el_per_ion - el_floor;

    const double ion_vt            = std::sqrt(ion.temp * Const::eV / ion.defaultmass);
    const double el_vt             = std::sqrt(el.temp  * Const::eV / el.defaultmass);

    auto flux_vx = [&](double vt)
    {   double sgn = (rnd() < 0.5) ? -1.0 : 1.0;
        return sgn * std::sqrt(-2.0 * std::log(1.0 - rnd())) * vt / domain.vel_norm;
    };

    for (int p = 0; p < n_pairs; p++)
    {
        double x = x_lo + rnd() * Ls;
        if (x >= x_src)
        {
            x = x_src - 1e-9;
        }

        ion.AddParticle(Particle(x, flux_vx(ion_vt),Init::SampleVel(ion, ion.temp) / domain.vel_norm, Init::SampleVel(ion, ion.temp) / domain.vel_norm, 0));

        const int n_el = el_floor + ((rnd() < el_frac) ? 1 : 0);
        for (int k = 0; k < n_el; k++)
        {
            el.AddParticle(Particle(x, flux_vx(el_vt),Init::SampleVel(el, el.temp) / domain.vel_norm, Init::SampleVel(el, el.temp) / domain.vel_norm, 0));
        }

        // negative ion should born at the same x as the ion/electron (keeps the source neutral)
        if (neg != nullptr)
        {
            const int n_neg = neg_floor + ((rnd() < neg_fr) ? 1 : 0);
            for (int k = 0; k < n_neg; k++)
            {   
                neg->AddParticle(Particle(x, flux_vx(neg_vt), Init::SampleVel(*neg, neg->temp) / domain.vel_norm,Init::SampleVel(*neg, neg->temp) / domain.vel_norm, 0));
            }
                
        }
    }
}

//  ion-ion recombination (volume loss of negative ions, as in Oudini et al., Phys. Plasmas 2013)
//  Loss rate per negative ion = k*n+(x), per positive ion = k*n-(x)  -> equal real-particle losses,
//  so charge is conserved on average. den is the normalized grid density (physical n = den*density).
void ion_ion_recombination(Species &ion, Species &neg, Domain &domain, double k_rec)
{
    if (k_rec <= 0) return;
    const double dt = domain.DT / domain.W;                       // [s]

    auto dens_at = [&](Species &s, double x)
    {
        double u = (x - domain.x0) / domain.dx;
        int i = static_cast<int>(std::floor(u));
        if (i < 0) i = 0;
        if (i > domain.ni - 2) i = domain.ni - 2;
        double f = u - i;
        return (s.den(i) * (1.0 - f) + s.den(i + 1) * f) * domain.density;
    };

    for (Particle &p : neg.part_list)
    {
        if (p.alive && rnd() < 1.0 - std::exp(-k_rec * dens_at(ion, p.x) * dt)) p.alive = false;
    }
    for (Particle &p : ion.part_list)
    {
        if (p.alive && rnd() < 1.0 - std::exp(-k_rec * dens_at(neg, p.x) * dt)) p.alive = false;
    }     
}

static void rethermalize_bulkboundary(Species &sp, Particle &p, Domain &domain, double dt, double x_loc)
{
    const double sigma = std::sqrt(sp.temp * Const::eV / sp.defaultmass);  // [m/s]
    const double Pos_Const = domain.vel_norm / (domain.L * domain.W);
    const double xmax = domain.x0 + domain.xL;

    //p.vx = -sigma * std::sqrt(-2.0 * std::log(1.0 - rnd())) / domain.vel_norm;
    p.vx = ((rnd() < 0.5) ? -1.0 : +1.0) * sigma * std::sqrt(-2.0 * std::log(1.0 - rnd())) / domain.vel_norm;
    //display::print("vx",p.vx);
    p.vy = Init::SampleVel(sp, sp.temp) / domain.vel_norm;
    p.vz = Init::SampleVel(sp, sp.temp) / domain.vel_norm;

    double x = x_loc + Pos_Const * p.vx * dt;   // p.vx<0 -> x inside domain
    //display::print("x",x);
    //if (x < domain.x0 || x >= xmax) x = xmax - 1e-9;   // safety for very fast particle
    p.x = x;
    p.alive = true;
}

void source_boundary_reemit(Species &sp, Domain &domain, double dt, double x_loc)
{
    //const double xmax = domain.x0 + domain.xL;
    for (Particle &p : sp.part_list)
    {
        if (p.x >= x_loc)
        {
            rethermalize_bulkboundary(sp, p, domain, dt, x_loc);
        }
    }    
}
//*/
