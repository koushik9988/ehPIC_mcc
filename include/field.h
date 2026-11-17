#ifndef _FIELD_H_
#define _FIELD_H_

#include <cstring>
#include <cassert>
#include "domain.h"
#include "linalg.h"
#include "function.h"
#include "solvers.h"
#include <fftw3.h>

class Domain; // Forward declaration of Domain class

class FieldSolve
{
    public:
    //Constructor
    FieldSolve(Domain &domain);//:domain(domain),nr_a(domain.ni),nr_b(domain.ni),nr_c(domain.ni),nr_rhs(domain.ni){};

    enum class SolverType
    {
        DIRECT,
        PCG,
        CG,
        GS,
        NLP
    };

    void PotentialSolver(int ts, Domain::BoundaryType bc_flag);
    void Direct(int ts, Domain::BoundaryType bc_flag);
    void NewtonRaphsonBoltzmann(int ts, Domain::BoundaryType bc_flag);
    void CalculateEfield(Domain::BoundaryType bc_flag);
    void pcgsolver();
    void cgsolver();
    void GaussElim();
    void Spectral();
    void fluid_den_update(vector<Species> &species_list);

    private:
    Domain &domain;
    vec<double> nr_a;
    vec<double> nr_b;
    vec<double> nr_c;
    vec<double> nr_rhs;
};

#endif // _FIELD_H_
