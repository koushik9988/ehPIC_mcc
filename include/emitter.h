#ifndef _EMITTER_H
#define _EMITTER_H

#include <vector>

class Domain;
class Species;


void source_boundary_reemit(Species &sp, Domain &domain, double dt, double x_loc);

struct EmitterParams
{
    double emitter_loc;
    double temp         =    0;
    int numparticle     =    0; //deprecat
    double density;
    double emitter_area;
    double vdrift       =    0;
    int species_id1;
    int species_id2     =   -1;

    //bulk injection parameters
    double spwt         =    -1;
    bool has_spwt       = false;
    bool has_drift      = false;
    int direction       =     0; // +1 inject toward +x,
    //-1 toward -x, 0 = auto (left half -> +1, right half -> -1)
    int flux_weighted   =     1;    // 1: vx ~ vx*exp(-vx^2/2s^2) (true flux from reservoir), 0: plain half-Gaussian
    double temp1        =    -1;           // injection temperature [eV] of species_id1 (<=0 -> use species temp)
    double temp2        =    -1;           // injection temperature [eV] of species_id2 (<=0 -> use species temp)


};

class Emitter
{
    public:
    Emitter(const EmitterParams &params, Domain &domain);
    void inject(std::vector<Species> &species_list);
    double get_current_density();

    /// newly added section 2026-10-12: point particle source at x0 + 0.75*xL
    //currenly not in use so commented
    //void source_emitter(Species &ion, Species &el);
    // emitter.h
    //void source_emitter(std::vector<Species> &sp, int drv = 1);   // drv = driver (positive ion) index

    private:
    EmitterParams params;
    Domain &domain;
    int compute_num_inject(double J, double charge, double spwt) const;
    double compute_current_density(double drift_vel, double charge) const;

    //new
    int    inward_direction() const;
    double sample_halfmaxwellian_vx(Species &sp, double temp_eV, double vt) const;   // [m/s], always >= 0
    double reservoir_flux(double sigma,double density) const;         // [particles/m^2/s] = n*sigma/sqrt(2pi)
    void   inject_halfmaxwellian(Species &sp, int n, double temp_eV, int dir, double spwt);
    void   inject_current(std::vector<Species> &species_list);    // drift_velocity given
    void   inject_reservoir(std::vector<Species> &species_list);  // no drift_velocity
    void   inject_surface(std::vector<Species> &species_list);

};


// newww: optional 3rd species (negative ion) in the source; neg_frac = S_neg/S_ion = alpha/(1+alpha)  (S+ = Se + S-)
void source_emitter(Species &ion, Species &el, Domain &domain, double emitter_area, Species *neg = nullptr, double neg_frac = 0.0);
//void source_emitter(Species &ion, Species &el, Domain &domain, double emitter_area);

// newww: volume ion-ion recombination  A+ + B- -> neutrals, rate k_rec [m^3/s]
void ion_ion_recombination(Species &ion, Species &neg, Domain &domain, double k_rec);

#endif
