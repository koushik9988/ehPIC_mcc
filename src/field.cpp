#include "field.h"

//mo problem here(numerical error problem )
//void FieldSolve::SolvePotDirect(double *x, double *rho)


FieldSolve::FieldSolve(Domain &domain): domain(domain),nr_a(domain.ni),nr_b(domain.ni),nr_c(domain.ni),nr_rhs(domain.ni){}


void FieldSolve::Direct(int ts, Domain::BoundaryType bc_flag)
{
    
    //const Domain::BoundaryType bc_flag = domain.bc_type;
    /* Set coefficients, precompute them*/
    int ni = domain.ni;
    double dx2 = domain.dx*domain.dx;

    vec<double> rhov(ni);

    ////RF
    auto boundary_potential = [&](double amp, double freq, double phase)
    {
        double t = ts * (domain.DT / domain.W); // physical seconds
        double omega = 2.0 * Const::PI * freq;
        return  amp * cos(omega * t + phase); //cos beause making omega zeo make it 1
    };
    ////

    switch(bc_flag)
    {
        case Domain::BoundaryType::PBC:
        {
            rhov(0) = 0;
            rhov(ni-1) = 0;
            break;
        }

        case Domain::BoundaryType::OPEN:
        {
            double v_left = boundary_potential(domain.vL, domain.fL, domain.phaseL);
            double v_right = boundary_potential(domain.vR, domain.fR, domain.phaseR);
            rhov(0) = v_left * domain.phi_norm;
            rhov(ni-1) = v_right * domain.phi_norm; 
            break;
        }

        case Domain::BoundaryType::DIRNEU:
        {
            //double v_left = boundary_potential(domain.vL, domain.fL, domain.phaseL);
            //double v_right = boundary_potential(domain.vR, domain.fR, domain.phaseR);
            //rhov(0) = v_left * domain.phi_norm;
            //rhov(ni-1) = v_right * domain.phi_norm; 
            //break;
            rhov(0) = boundary_potential(domain.vL, domain.fL, domain.phaseL) * domain.phi_norm;
            // dphi/dx is normalized in input so we have to make it normalized when passed in pic solver.
            rhov(ni-1) = -domain.rho(ni-1)*dx2*((domain.L*domain.L)/(domain.LDe*domain.LDe))- 
            2.0*domain.dx*domain.grad_phi*domain.phi_norm*domain.L;
            break;
        }
    }
       
    for( int i = 1 ; i < ni-1 ; i++)
    {
        rhov(i) = -domain.rho(i)*dx2*((domain.L*domain.L)/(domain.LDe*domain.LDe));
        //rhov(i) += 0.1*sin((2*3.14*i)/domain.xL)*((domain.L*domain.L)/(domain.LD*domain.LD));
    }

    //vec<double> sol = direct(rhov,ni);
    vec<double> sol = (bc_flag == Domain::BoundaryType::DIRNEU) ? direct(rhov,ni,true) : direct(rhov,ni);

    domain.phi = 0;
    domain.phi = sol;
}

// -----------------------------------------------------------------------------
// Newton-Raphson solver for nonlinear Poisson + Boltzmann fluid electrons
//
//   d²phi/dx² = -(L²/LDe²) * [rho(i) - n0_ref * exp(phi(i))]
//
//   phi       = normalized electrostatic potential
//   rho       = summed charge density from all non-fluid kinetic species
//   n0_ref    = reference fluid electron density = normden[0] / domain.density
//   J * delta_phi = -F
//   Trancendebatl equation to solve
//   F_i = (phi_{i+1} - 2*phi_i + phi_{i-1})/dx² + (L²/LDe²) * (rho_ion_i - n0_ref * exp(phi_i))
//   J = dF/dphi
//   J_ii     = -2/dx² - (L²/LDe²) * n0_ref * exp(phi_i)   (diagonal)
//   J_{i,i±1}= 1/dx²                                       (off-diagonals)
//   Newton raphson logic
//   phi_k+1 = phi_k - J^-1*F
//   Let delta_phi =  J^-1*F
//   We  get linear system Jy = F where J is tridiagonal 
// -----------------------------------------------------------------------------
void FieldSolve::NewtonRaphsonBoltzmann(int ts, Domain::BoundaryType bc_flag)
{
    //display::print("density ref",domain.ne0_ref);
    int    ni   = domain.ni;
    double dx2  = domain.dx * domain.dx;
    double Debye_ratio = (domain.L * domain.L) / (domain.LDe * domain.LDe);
 
    auto boundary_potential = [&](double amp, double freq, double phase)
    {
        double t = ts * (domain.DT / domain.W); // physical seconds
        double omega = 2.0 * Const::PI * freq;
        return  amp * cos(omega * t + phase);
    };
 
    double phi_left  = 0.0,  phi_right = 0.0;

    if (bc_flag == Domain::BoundaryType::OPEN)
    {
        phi_left  = boundary_potential(domain.vL, domain.fL, domain.phaseL) * domain.phi_norm;
        phi_right = boundary_potential(domain.vR, domain.fR, domain.phaseR) * domain.phi_norm;
    }
 
    // Enforce boundary values on phi
    domain.phi(0)    = phi_left;
    domain.phi(ni-1) = phi_right;
 
    // Newton-Raphson iteration
    const int    NR_MAX = domain.max_iteration;
    const double NR_TOL = domain.tolerance;
 
    for (int iter = 0; iter < NR_MAX; iter++)
    {

        nr_a(0)=0;  nr_b(0)=1;  nr_c(0)=0;  nr_rhs(0)=0;
        nr_a(ni-1)=0;  nr_b(ni-1)=1;  nr_c(ni-1)=0;  nr_rhs(ni-1)=0;
 
        double max_F = 0.0;
 
        for (int i = 1; i < ni-1; i++)
        {
        
            //Trancendental function F_i/Residual should be zero but will not
            double F_i = (domain.phi(i+1) - 2.0*domain.phi(i) + domain.phi(i-1)) / dx2 + Debye_ratio * (domain.rho(i) - domain.ne0_ref * std::exp(domain.phi(i)));
 
            // Jacobian or Derivative of F_i => J_{i,i-1}=1/dx², J_{ii}=-2/dx²-L2_LDe2*n0_ref*ep, J_{i,i+1}=1/dx²
            nr_a(i)   =  1.0 / dx2;
            nr_b(i)   = -2.0 / dx2 - Debye_ratio * domain.ne0_ref * std::exp(domain.phi(i));
            nr_c(i)   =  1.0 / dx2;
            nr_rhs(i) = -F_i;     // solve J*delta_phi = -F
 
            max_F = std::max(max_F, std::fabs(F_i));
        }
 
        // Solve tridiagonal system J * delta_phi = nr_rhs using Thomas algorithm
        vec<double> dphi = direct(nr_a, nr_b, nr_c, nr_rhs);
 
        // Update phi (boundaries stay fixed)
        for (int i = 1; i < ni-1; i++)
        {
            //earlier we solve J*delta_phi = -F to plus sign here
            domain.phi(i) += dphi(i);
        }
 
        if (max_F < NR_TOL) break;
    }

}


void FieldSolve::fluid_den_update(vector<Species> &species_list)
{
    for (Species &sp : species_list)
    {
        if (!sp.isfluid) continue;
        for (int i = 0; i < domain.ni; i++)
        {
            sp.den(i) = domain.ne0_ref*std::exp(domain.phi(i));
        }
    }
}


///
void FieldSolve::Spectral()
{
    // NOTE: FFT solver assumes PERIODIC boundary conditions only.
    // Do not call this with OPEN boundaries.
    assert(domain.bc_type == Domain::BoundaryType::PBC);

    int ni = domain.ni;
    double L = domain.xL;
    int nr = (ni / 2) + 1;
    double norm = 1.0 / ni;

    double *rho_buf = fftw_alloc_real(ni);
    double *phi_buf = fftw_alloc_real(ni);
    fftw_complex *rho_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nr);
    fftw_complex *phi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nr);

    double scale = (domain.L * domain.L) / (domain.LDe * domain.LDe);
    for (int i = 0; i < ni; i++)
        rho_buf[i] = -domain.rho(i) * scale;

    fftw_plan forward = fftw_plan_dft_r2c_1d(ni, rho_buf, rho_k, FFTW_ESTIMATE);
    fftw_execute(forward);
    fftw_destroy_plan(forward);

    double dx = domain.dx;
    for (int k = 0; k < nr; k++)
    {
        double kx = 2.0 * M_PI * k / L;

        // Use discrete Laplacian eigenvalue to match the FD stencil in Direct()
        // kx_fd² = (4/dx²) * sin²(π k / ni)
        double sin_arg = std::sin(M_PI * k / ni);
        double denom = (k == 0) ? 1.0 : -(4.0 / (dx * dx)) * sin_arg * sin_arg;

        // k=0: enforce zero mean potential (requires charge-neutral plasma)
        phi_k[k][0] = (k == 0) ? 0.0 : rho_k[k][0] / denom;
        phi_k[k][1] = (k == 0) ? 0.0 : rho_k[k][1] / denom;

        domain.dft_k(k)     = kx;
        domain.dft_value(k) = std::sqrt(rho_k[k][0]*rho_k[k][0] + rho_k[k][1]*rho_k[k][1]);
    }

    fftw_plan backward = fftw_plan_dft_c2r_1d(ni, phi_k, phi_buf, FFTW_ESTIMATE);
    fftw_execute(backward);
    fftw_destroy_plan(backward);

    domain.phi = 0;
    for (int i = 0; i < ni; i++)
        domain.phi(i) = phi_buf[i] * norm;

    fftw_free(rho_k);
    fftw_free(phi_k);
    fftw_free(rho_buf);
    fftw_free(phi_buf);
}


/*
void FieldSolve::Spectral()
{
    int ni = domain.ni;
    double L = domain.xL;
    int nr = (ni/2) + 1;
    double norm = 1.0 /ni;
 
    double *rho = fftw_alloc_real(ni);
    double *phi = fftw_alloc_real(ni);
   
    fftw_complex *rho_k= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (nr));
    fftw_complex *phi_k= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (nr));

    for(int i = 0 ; i < ni ; i++)
    {
        rho[i] =  -domain.rho(i)*((domain.L*domain.L)/(domain.LDe*domain.LDe));
    }

    // Forward FFT
    fftw_plan forward = fftw_plan_dft_r2c_1d(ni, rho , rho_k, FFTW_ESTIMATE);
    fftw_execute(forward);
    fftw_destroy_plan(forward);

    //display::print(phi_k[0][0]);
    // Solve in Fourier space
    for (int k = 0; k <= ni/2; ++k)
    {
        //double kx = 2.0 * M_PI * k / L;
        //double denom = (k == 0) ? 1.0 : -(kx * kx); // Avoid divide-by-zero at k=0
        double dx = domain.dx;
        double kx = 2.0 * M_PI * k / L;
        double denom = (k == 0) ? 1.0 : -kx*kx;//- (4/(dx*dx))*sin(kx*dx*0.5)*sin(kx*dx*0.5);

        phi_k[k][0] = (k == 0) ? 0.0 : rho_k[k][0] / denom;
        phi_k[k][1] = (k == 0) ? 0.0 : rho_k[k][1] / denom;

        domain.dft_k(k) = kx;
    }

    //display::print(phi_k[0][0]);

    for( int i = 0 ; i < nr ; i++)
    {
        domain.dft_value(i) = sqrt(rho_k[i][0]*rho_k[i][0] + rho_k[i][1]*rho_k[i][1]);
    }

    // Inverse FFT
    fftw_plan backward = fftw_plan_dft_c2r_1d(ni, phi_k, phi, FFTW_ESTIMATE);
    fftw_execute(backward);
    fftw_destroy_plan(backward);

    domain.phi = 0;
    for (int i = 0; i < ni; ++i)
    {
        domain.phi(i) = phi[i] * norm;
    }

    // Cleanup
    fftw_free(rho_k);
    fftw_free(phi_k);
    fftw_free(rho);
    fftw_cleanup();
    
}

*/
void FieldSolve::GaussElim()
{
    double L2;
    int n = domain.ni;
    double dx2 = domain.dx*domain.dx;

    vec<double> b(n);
    Matrix<double> A(n,n);

   
    A(0, 0) = 1.0; //
    b(0) = 0;   //left fixed potential  bc
    
    A(n - 1, n - 1) = 1.0; 
    b(n - 1) = 0; // Right boundary (fixed potential)

    // Populate internal nodes
    for (int i = 1; i < n-1; i++) 
    {
        A(i, i - 1) = 1.0; // A(n, n-1)
        A(i, i) = -2.0;   // A(n, n)
        A(i, i + 1) = 1.0; // A(n, n+1)
        b(i) = -domain.rho(i)*dx2*((domain.L*domain.L)/(domain.LDe*domain.LDe));
    }
    
    vec<double> sol = gausselimination(A,b);

    domain.phi = 0;
    
    domain.phi = sol;
}


void FieldSolve::pcgsolver()
{
    double L2;
    int n = domain.ni;
    double dx2 = domain.dx*domain.dx;

    vec<double> b(n);
    vec<double> x(n);
    Matrix<double> A(n,n);

    std::vector<bool> fixed_node(n, false); 
    // Set boundary conditions
   
    A(0, 0) = 1.0; //
    b(0) = 0;   //left fixed potential  bc
    
    A(n - 1, n - 1) = 1.0; 
    b(n - 1) = 0; // Right boundary (fixed potential)

    // Populate internal nodes
    for (int i = 1; i < n-1; i++) 
    {
        A(i, i - 1) = 1.0; // A(n, n-1)
        A(i, i) = -2.0;   // A(n, n)
        A(i, i + 1) = 1.0; // A(n, n+1)
        b(i) = -domain.rho(i)*dx2*((domain.L*domain.L)/(domain.LDe*domain.LDe));
        x(i) = domain.phi(i);
    }

    Matrix<double> A0 = slice(A,1,n-1,1,n-1);
    vec<double> b0 = slice(b,1,n-1);
    vec<double> x0 = slice(x,1,n-1);

    //symmetrycheck(A0);

    vec<double> x_sol = cg(A0,x0,b0,domain.max_iteration, domain.tolerance);
   
    vec<double> v_norm = A0*x_sol - b0;

    //reset phi
    domain.phi = 0;

    domain.phi(0) = b(0);
    domain.phi(n-1) = b(n-1);
    
    domain.phi = x_sol;

    for(int i = 1; i< n-1; i++)
    {
        domain.phi(i) = x_sol(i-1);
    }
}


void FieldSolve::cgsolver()
{
    double L2;

    int n = domain.ni;
    double dx2 = domain.dx*domain.dx;

    vec<double> b(n);
    vec<double> x(n);
    Matrix<double> A(n,n);

    std::vector<bool> fixed_node(n, false); 
    // Set boundary conditions
   
    A(0, 0) = 1.0; //
    b(0) = 0;   //left fixed potential  bc
    
    A(n - 1, n - 1) = 1.0; 
    b(n - 1) = 0; // Right boundary (fixed potential)

    // Populate internal nodes
    for (int i = 1; i < n-1; i++) 
    {
        A(i, i - 1) = 1.0; // A(n, n-1)
        A(i, i) = -2.0;   // A(n, n)
        A(i, i + 1) = 1.0; // A(n, n+1)
        b(i) = -domain.rho(i)*dx2*((domain.L*domain.L)/(domain.LDe*domain.LDe));
        x(i) = domain.phi(i);
    }

    Matrix<double> A0 = slice(A,1,n-1,1,n-1);
    vec<double> b0 = slice(b,1,n-1);
    vec<double> x0 = slice(x,1,n-1);

    //symmetrycheck(A0);

    vec<double> x_sol = cg(A0,x0,b0,domain.max_iteration, domain.tolerance);
   
    vec<double> v_norm = A0*x_sol - b0;

    //display::print(x_sol.getSize());

    domain.norm = v_norm.norm();

    domain.phi(0) = b(0);
    domain.phi(n-1) = b(n-1);
    
    for(int i = 1; i< n-1; i++)
    {
        domain.phi(i) = x_sol(i-1);
    }
}

//void FieldSolve::CalculateEfield(double *phi, double *ef)
void FieldSolve::CalculateEfield(Domain::BoundaryType bc_flag)
{
    
        domain.ef = 0;
        for(int i=1; i<domain.ni-1; i++)
        {
            domain.ef(i) = -((domain.LDe*domain.LDe)/(domain.L*domain.L))*(domain.phi(i+1)-domain.phi(i-1))/(2*domain.dx);
        }
        
        /*for continous bounndary
        the point 0 and ni-1 is same */
        switch (bc_flag)
        {   
            case Domain::BoundaryType::PBC:
            {
                domain.ef(0) = -((domain.LDe*domain.LDe)/(domain.L*domain.L))*(domain.phi(1)-domain.phi(domain.ni-2))/(2*domain.dx);
                domain.ef(domain.ni-1) = domain.ef(0);
                break;
            }

            case Domain::BoundaryType::OPEN:
            {
                domain.ef(0) = -((domain.LDe*domain.LDe)/(domain.L*domain.L))*(domain.phi(1)- domain.phi(0))/(domain.dx);
                domain.ef(domain.ni-1) = -((domain.LDe*domain.LDe)/(domain.L*domain.L))*(domain.phi(domain.ni-1)-domain.phi(domain.ni-2))/(domain.dx);
                break;
            }
        
        }
        
}


void FieldSolve::PotentialSolver(int ts, Domain::BoundaryType bc_flag)
{
    if      (domain.SolverType == "direct")   Direct(ts, bc_flag);
    else if (domain.SolverType == "cg")       cgsolver();
    else if (domain.SolverType == "pcg")      pcgsolver();
    else if (domain.SolverType == "spectral") Spectral();
    else if (domain.SolverType == "nlp") NewtonRaphsonBoltzmann(ts,bc_flag);
}
