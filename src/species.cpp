#include "species.h"


/* note :
"-> " symbol means dereferencing a pointer , here we created pointers to instances of class named domain.
so to dereference the instances we have to use the symbol "->"*/

Species::Species(string name, double defaultmass, double charge, double defaultspwt, double temp, int numparticle, double vs, 
    double fract_den , double bulkden, bool isfluid, std:: string initialization_pos, string initialization_vel, Domain &domain):domain(domain)
{
    this->name = name;
    this->defaultmass = defaultmass;
    this->charge = charge;
    this->defaultspwt = defaultspwt;
    this->temp = temp;
    this->numparticle = numparticle;
    this->vs = vs;
    this->fract_den = fract_den;
    this->bulkden = bulkden;
    this->isfluid = isfluid;
    this->initialization_pos = initialization_pos;
    this->initialization_vel = initialization_vel;

    den = vec<double>(domain.ni);
    coll_rate = vec<double>(domain.ni);

    velmesh = vec<double>(domain.ni);
    power_rate = vec<double>(domain.ni);
    current_density = vec<double>(domain.ni);  
    
    //velprev = vec<double>(numparticle);

    //for(int i = 0; i < domain.num_threads; i++)
    //{
    //    domain.buffers.emplace_back(domain.ni);
    //}
    
}

void Species::AddParticle(Particle part)
{
    if (std::isnan(part.mass))
    {
        part.mass = this->defaultmass;
    }
    if (std::isnan(part.spwt))
    {
        part.spwt = this->defaultspwt;
    }

    this->part_list.push_back(part);
}


void Species::Push_species_serial(vector<Species> &species_list, int sub_cycle)
{
    
    ////////
    double EN_Td = domain.ext_E_field;
    double E_actual = EN_Td ;//* 1e-21 *  3.2188e22 ;   // V/m

    double ef_test = (E_actual * Const::EPS_0 )/ (domain.density * Const::QE * domain.L);
    
    ///////////////
    
    //Precompute constants for push
    double dt = domain.DT * sub_cycle;
    double Vel_Const = ((domain.density*Const::QE*domain.L) / (Const::EPS_0*domain.W*domain.vel_norm));
    double Pos_Const = ((domain.vel_norm)/(domain.L*domain.W));
    double Bx, By, Bz, tx, ty, tz, s;
    double qm = charge / defaultmass;
    const Domain::BoundaryType bc_flag = domain.bc_type;

    if(domain.B != 0)
    {
        Bx = domain.B * sin(domain.theta) * cos(domain.azimuth);
        By = domain.B * sin(domain.theta) * sin(domain.azimuth);
        Bz = domain.B * cos(domain.theta);

        // t = (q/m) * B * dt / 2
        tx = 0.5 * qm * Bx * (dt / domain.W);
        ty = 0.5 * qm * By * (dt / domain.W);
        tz = 0.5 * qm * Bz * (dt / domain.W);
        s = 2.0 / (1.0 + tx*tx + ty*ty + tz*tz);
    }

    const double x_src = domain.x0 + domain.xL;

    for (Particle &part : part_list)
    {
        double lc = part.x;
        //double qm = charge / part.mass;

        double part_ef = ef_test + domain.Interpolate(lc, domain.ef);
        //display::print(domain.Interpolate(lc, domain.ef))

        // half step E
        double accel = qm * Vel_Const * part_ef;

        double vx_minus = part.vx + 0.5 * accel * dt;
        double vy_minus = part.vy;
        double vz_minus = part.vz;

        double vx_plus, vy_plus, vz_plus;

        // Boris push magnetic field       
        if (domain.B != 0.0)
        {
            vx_minus *= domain.vel_norm;
            vy_minus *= domain.vel_norm;
            vz_minus *= domain.vel_norm;

            // v_prime = v_minus + v_minus x t
            double vpx = vx_minus + (vy_minus*tz - vz_minus*ty);
            double vpy = vy_minus + (vz_minus*tx - vx_minus*tz);
            double vpz = vz_minus + (vx_minus*ty - vy_minus*tx);

            // s = 2 / (1 + |t|^2)
            //double s = 2.0 / (1.0 + tx*tx + ty*ty + tz*tz);

            // v_plus = v_minus + v_prime x (s * t)
            // sx = s * tx  | sy = s * ty |  sz = s * tz
            vx_plus = vx_minus + (vpy*(s*tz) - vpz*(s*ty));
            vy_plus = vy_minus + (vpz*(s*tx) - vpx*(s*tz));
            vz_plus = vz_minus + (vpx*(s*ty) - vpy*(s*tx));

            vx_plus /= domain.vel_norm;
            vy_plus /= domain.vel_norm;
            vz_plus /= domain.vel_norm;
        }
        else
        {
            vx_plus = vx_minus;
            vy_plus = vy_minus;
            vz_plus = vz_minus;
        }

        //Half-step acceleration by E field 
        part.vx = vx_plus + 0.5 * accel * dt;
        part.vy = vy_plus;
        part.vz = vz_plus;

        double old_pos = part.x;
        part.x += Pos_Const * part.vx * dt;

        double new_pos = part.x;

        
        if (fabs(new_pos - old_pos) >= 1)
        {

        }

        switch(bc_flag)
        {
            case Domain::BoundaryType::PBC:
            {
                if(part.x < domain.x0)
                {
                    part.x += domain.xL;
                }
                else if (part.x >= domain.x0 + domain.xL)
                {
                    part.x -= domain.xL;
                }

                break;

            }

            
            case Domain::BoundaryType::OPEN:
            {
                if (part.x < domain.x0 || part.x >= domain.x0 + domain.xL)
                {
                    part.alive = false; 
                }

                break;

            }

            //we will removedirneu as it could be wrong for now keep it
            case Domain::BoundaryType::DIRNEU:
            {
                if (part.x < domain.x0)
                //if (part.x < domain.x0 || part.x >= domain.x0 + domain.xL)
                {
                    part.alive = false; 
                }
                else if (charge < 0 && name != "electron" && part.x >= domain.x0 + 0.9 * domain.xL)
                {
                    part.alive = false;
                }
        
                break;
            }

        }//switch brack
    }


    if (bc_flag == Domain::BoundaryType::DIRNEU)
    {
        for (Particle &p : part_list)
        {
            if (p.alive && p.x >= x_src)
            {
                source_reemit(p, dt, x_src);   // electrons and ions both
            }
                
        }
    }

    if (bc_flag == Domain::BoundaryType::OPEN || bc_flag == Domain::BoundaryType::DIRNEU)
    {
        
        /*
            remove_if function doesnot actually erase elements from the vector, it just moves the "alive" particles to the front and returns an iterator to the new end of the vector. 
            Then the erase function is called to remove the "dead" particles from the vector based on the new end iterator returned by remove_if. This approach is efficient and
            avoids issues with modifying the vector while iterating over it in a multithreaded context.
            Argument of remove_if is iterator to the beginning of the vector, iterator to the end of the vector and a condition defined by a
            lambda function that returns true for particles that are not alive (i.e., should be removed).
        */
        
        part_list.erase(std::remove_if(part_list.begin(), part_list.end(),[](const Particle& p){ return !p.alive; }),part_list.end());

    }

    if (bc_flag == Domain::BoundaryType::DIRNEU)
    {
        //source_boundary_reemit(*this, domain, dt, int(domain.xL*0.75));
    }
        
}

void Species::Push_species(vector<Species> &species_list, int sub_cycle)
{
    
    if (domain.num_threads <= 1)
    {
        Push_species_serial(species_list, sub_cycle);
        return;
    }

    double EN_Td = domain.ext_E_field;
    double E_actual = EN_Td * 1e-21 *  3.2188e22 ;   // V/m

    double ef_test = (E_actual * Const::EPS_0 )/ (domain.density * Const::QE * domain.L);

    const int particle_count = static_cast<int>(part_list.size());
    //Precompute constants for push
    double dt = domain.DT * sub_cycle;
    double Vel_Const = ((domain.density*Const::QE*domain.L) / (Const::EPS_0*domain.W*domain.vel_norm));
    double Pos_Const = ((domain.vel_norm)/(domain.L*domain.W));
    double Bx, By, Bz, tx, ty, tz, s;
    double qm = charge / defaultmass;
    const Domain::BoundaryType bc_flag = domain.bc_type;

    if(domain.B != 0)
    {
        Bx = domain.B * sin(domain.theta) * cos(domain.azimuth);
        By = domain.B * sin(domain.theta) * sin(domain.azimuth);
        Bz = domain.B * cos(domain.theta);

        // t = (q/m) * B * dt / 2
        tx = 0.5 * qm * Bx * (dt / domain.W);
        ty = 0.5 * qm * By * (dt / domain.W);
        tz = 0.5 * qm * Bz * (dt / domain.W);
        s = 2.0 / (1.0 + tx*tx + ty*ty + tz*tz);
    }

    int ion_hits_left = 0;
    int ion_hits_right = 0;

    const double x_src = domain.x0 + domain.xL;

    #pragma omp parallel for schedule(static) num_threads(domain.num_threads) reduction(+:ion_hits_left, ion_hits_right)
    for (int i = 0; i < particle_count; ++i)
    {
        
        //double qm = charge / part.mass;
        Particle &part = part_list[i];
        double lc = part.x;
        double part_ef = ef_test + domain.Interpolate(lc, domain.ef);

        // half step E
        double accel = qm * Vel_Const * part_ef;

        double vx_minus = part.vx + 0.5 * accel * dt;
        double vy_minus = part.vy;
        double vz_minus = part.vz;

        double vx_plus, vy_plus, vz_plus;

        // Boris push magnetic field       
        if (domain.B != 0.0)
        {
            vx_minus *= domain.vel_norm;
            vy_minus *= domain.vel_norm;
            vz_minus *= domain.vel_norm;

            // v_prime = v_minus + v_minus x t
            double vpx = vx_minus + (vy_minus*tz - vz_minus*ty);
            double vpy = vy_minus + (vz_minus*tx - vx_minus*tz);
            double vpz = vz_minus + (vx_minus*ty - vy_minus*tx);

            // s = 2 / (1 + |t|^2)
            //double s = 2.0 / (1.0 + tx*tx + ty*ty + tz*tz);

            // v_plus = v_minus + v_prime x (s * t)
            // sx = s * tx  | sy = s * ty |  sz = s * tz
            vx_plus = vx_minus + (vpy*(s*tz) - vpz*(s*ty));
            vy_plus = vy_minus + (vpz*(s*tx) - vpx*(s*tz));
            vz_plus = vz_minus + (vpx*(s*ty) - vpy*(s*tx));

            vx_plus /= domain.vel_norm;
            vy_plus /= domain.vel_norm;
            vz_plus /= domain.vel_norm;
        }
        else
        {
            vx_plus = vx_minus;
            vy_plus = vy_minus;
            vz_plus = vz_minus;
        }

        //Half-step acceleration by E field 
        part.vx = vx_plus + 0.5 * accel * dt;
        part.vy = vy_plus;
        part.vz = vz_plus;

        double old_pos = part.x;
        part.x += Pos_Const * part.vx * dt;

        double new_pos = part.x;

        
        if (fabs(new_pos - old_pos) >= 1)
        {

        }

        switch(bc_flag)
        {
            case Domain::BoundaryType::PBC:
            {
                if(part.x < domain.x0)
                {
                    part.x += domain.xL;
                }
                else if (part.x >= domain.x0 + domain.xL)
                {
                    part.x -= domain.xL;
                }

                break;

            }
            case Domain::BoundaryType::OPEN:
            {
                if (part.x < domain.x0 || part.x >= domain.x0 + domain.xL)
                {
                    part.alive = false; 
                }
                break;
            }

            case Domain::BoundaryType::DIRNEU:
            {
                if (part.x < domain.x0)
                //if (part.x < domain.x0 || part.x >= domain.x0 + domain.xL)
                {
                    part.alive = false; 
                }
                else if (charge < 0 && name != "electron" && part.x >= domain.x0 + 1 * domain.xL)
                {
                    part.alive = false;
                }
                break;
            }
        }//switch bracket
    }


    if (bc_flag == Domain::BoundaryType::DIRNEU)
    {
        for (Particle &p : part_list)
        {
            if (p.alive && p.x >= x_src)
            {
                source_reemit(p, dt, x_src);   // electrons and ions both
            }
                
        }
    }

    if (bc_flag == Domain::BoundaryType::OPEN || bc_flag == Domain::BoundaryType::DIRNEU)
    {
        
        /*
            remove_if function doesnot actually erase elements from the vector, it just moves the "alive" particles to the front and returns an iterator to the new end of the vector. 
            Then the erase function is called to remove the "dead" particles from the vector based on the new end iterator returned by remove_if. This approach is efficient and
            avoids issues with modifying the vector while iterating over it in a multithreaded context.
            Argument of remove_if is iterator to the beginning of the vector, iterator to the end of the vector and a condition defined by a
            lambda function that returns true for particles that are not alive (i.e., should be removed).
        */
        
        part_list.erase(std::remove_if(part_list.begin(), part_list.end(),[](const Particle& p){ return !p.alive; }),part_list.end());

    }

    if (bc_flag == Domain::BoundaryType::DIRNEU)
    {
        //source_boundary_reemit(*this, domain, dt, int(domain.xL*0.75));
    }
    
    
}

void Species::ScatterSpecies_serial()
{   
    const Domain::BoundaryType bc_flag = domain.bc_type;
    //reset density
    den = 0;
    for(Particle &part : part_list)
    {
        double lc = domain.XtoL(part.x);
        //double lc = part.pos;
        domain.Scatter(lc,part.spwt,den);
    }

    den /= (domain.dx*domain.L);
    den /= domain.density;

    
    switch (bc_flag)
    {
        case Domain::BoundaryType::PBC:
        {
            den(0) += den(domain.ni-1);
            den(domain.ni-1) =  den(0);
            break;
        }
        case Domain::BoundaryType::OPEN:
        {
            den(0) *= 2;
            den(domain.ni-1) *= 2;
            break;
        }  
    }

}


void Species::ScatterSpecies()
{
    const Domain::BoundaryType bc_flag = domain.bc_type;
    if (domain.num_threads <= 1)
    {
        ScatterSpecies_serial();
        return;
    }

    //ScatterSpecies_serial();

    //
    const int particle_count = static_cast<int>(part_list.size());
    const int nthreads = std::max(1, domain.num_threads);
    std::vector<vec<double>> local_buffers(nthreads, vec<double>(domain.ni));
    
    //local_buffers.clear();

    #pragma omp parallel num_threads(nthreads)
    {
        //get threads id
        const int tid = omp_get_thread_num();
        vec<double> &local = local_buffers[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < particle_count; ++i)
        {
            Particle &part = part_list[i];
            const double lc = domain.XtoL(part.x);
            domain.Scatter(lc, part.spwt, local);
        }
    }

    den = 0;
    for (int i = 0; i < nthreads; ++i)
    {
        den += local_buffers[i];
    }

    den /= (domain.dx * domain.L);
    den /= domain.density;

    switch (bc_flag)
    {
        case Domain::BoundaryType::PBC:
        {
            den(0) += den(domain.ni-1);
            den(domain.ni-1) =  den(0);
            break;
        }
        case Domain::BoundaryType::OPEN:
        {
            den(0) *= 2;
            den(domain.ni-1) *= 2;
            break;
        }  
    }
   // 
}

///mesh averaged velocity or bulk fluid velocity
/// we are calculating u(x) = ∫ v * f(x,v) dv / ∫ f(x,v) dv
/// u(x) = (1/n)∫v * f(x,v) dv
void Species::ScatterVel_serial()
{   

    const Domain::BoundaryType bc_flag = domain.bc_type;
    //reset density
    velmesh = 0;
    for(Particle &part : part_list)
    {
        double lc = domain.XtoL(part.x);
        domain.Scatter(lc,part.spwt*part.vx,velmesh);
    }

    //divide bye cell voloume ( in 1D it is just cell lenght)
    velmesh /= (domain.dx*domain.L);

    for(int i = 0 ; i < domain.ni ; i++)
    {
        //bulk fluid velocity 
        //divide by den(i)*domain.density because actual density
        // is den(i)*domain.density and den(i) is just normalized qunatity
        // so we divide by actual density
        velmesh(i) /= den(i)*domain.density;
    }
    //velmesh /= den*domain.density;

    switch (bc_flag)
    {
        case Domain::BoundaryType::PBC:
        {
            velmesh(0) += velmesh(domain.ni-1);
            velmesh(domain.ni-1) =  velmesh(0);
            break;
        }
        case Domain::BoundaryType::OPEN:
        {
            velmesh(0) *= 2;
            velmesh(domain.ni-1) *= 2;
            break;
        }  
    }
}

void Species::Rewind_species()
{
    for (Particle &part: part_list)
    {
        double lc = domain.XtoL(part.x);
        //double lc = part.pos;
        // compute charge to mass ratio
        double qm = charge/part.mass;
        //cout<<qm<<endl;
        double part_ef = domain.Interpolate(lc,domain.ef);
        double wl = domain.LDe*domain.LDe*domain.wpe*domain.wpe;

        part.vx -= 0.5*qm*((domain.density*Const::QE*domain.L)/(Const::EPS_0*domain.W*domain.vel_norm))*part_ef*domain.DT;
                
    }
}

// paralla is checked for correcness
vec<double> Species::Compute_KE(Species &species)
{
    vec<double> ke(3);
    const int N = static_cast<int>(part_list.size());

    double ke0 = 0;
    double ke1 = 0;
    double ke2 = 0;

    #pragma omp parallel for reduction(+:ke0,ke1,ke2) schedule(static)
    for (int i = 0; i < N; i++)
    //for (Particle &part:part_list)
    {
        const Particle &part = part_list[i];
        // un-normalize the velocity by multiplying with the cold thermal velocity
        double massfactor = part.mass*part.spwt;
        ke0 += (part.vx*part.vx) * massfactor;
        ke1 += (part.vy*part.vy) * massfactor;
        ke2 += (part.vz*part.vz) * massfactor;
        //ke(0) += (part.vx*part.vx) * massfactor;
        //ke(1) += (part.vy*part.vy) * massfactor;
        //ke(2) += (part.vz*part.vz) * massfactor ;
    }
    /*Multiply 0.5*mass for all particles*/
    ke(0) = 0.5 * ke0 * (domain.vel_norm)*(domain.vel_norm);
    ke(1) = 0.5 * ke1 * (domain.vel_norm)*(domain.vel_norm);
    ke(2) = 0.5 * ke2 * (domain.vel_norm)*(domain.vel_norm);
    //ke(0) *= 0.5 * (domain.vel_norm)*(domain.vel_norm);
    //ke(1) *= 0.5 * (domain.vel_norm)*(domain.vel_norm);
    //ke(2) *= 0.5 * (domain.vel_norm)*(domain.vel_norm);
    
    // Calculate the total thermal energy of all the cold electrons
    double Th = (species.temp*Const::eV)*(species.defaultspwt)*species.numparticle;
    if(domain.normscheme == 5)
    {Th = 1;}

    // Normalize the kinetic energy by the total thermal energy of cold electrons    
    ke(0) = ke(0)/Th;
    ke(1) = ke(1)/Th;
    ke(2) = ke(2)/Th;
    return ke;
}

vec<double> Species::Compute_Momentum(Species &species)
{
    vec<double> pe(3);
    const int N = static_cast<int>(part_list.size());

    double pe0 = 0;
    double pe1 = 0;
    double pe2 = 0;


    #pragma omp parallel for reduction(+:pe0,pe1,pe2) schedule(static)
    for (int i = 0; i < N; i++)
    //for (Particle &part : part_list)
    {
        const Particle &part = part_list[i];
        // Compute total momentum in each direction (average velocity)
        pe0 += (part.vx) * part.mass * part.spwt;
        pe1 += (part.vy) * part.mass * part.spwt;
        pe2 += (part.vz) * part.mass * part.spwt;
    }

    pe(0) = pe0 * domain.vel_norm;
    pe(1) = pe1 * domain.vel_norm;
    pe(2) = pe2 * domain.vel_norm;

    // Normalize momentum
    double Thp;
    if (domain.normscheme == 5)
    {
        Thp = sqrt(domain.energy_scale * species.defaultspwt * species.numparticle * Const::ME);
    }
    else
    {
        Thp = sqrt((species.temp * Const::eV) / Const::ME) * species.defaultspwt * species.numparticle * Const::ME;
    }

    pe(0) /= Thp;
    pe(1) /= Thp;
    pe(2) /= Thp;

    return pe;
}

bool Species::IsIon()
{
    if(charge > 0)
    {return true;}
    else
    {return false;}
}


void Species::ComputeCurrentDensity()
{
    //Reset current density to zero
    current_density = 0;
    
    //Calculate current density at each grid point: j = q * n * v
    //where:
    //q = charge of the species
    //n = number density at grid point (den)
    //v = mean velocity at grid point (velmesh)
    
    for (int i = 0; i < domain.ni; i++)
    {
        // Current density = charge * density * velocity * unnorm 
        current_density(i) = charge * den(i) * velmesh(i) * domain.density * domain.vel_norm;
    }
}

void Species::ComputeHeatingRate()
{
    // Reset heating rate to zero
    power_rate = 0;
    
    //Calculate heating rate at each grid point: P = j * E
    //where:
    //j = current density at grid point (current_density)
    //E = electric field at grid point (domain.ef)
    
    for (int i = 0; i < domain.ni; i++)
    {
        // Heating rate (power absorption) = current density * electric field
        power_rate(i) = current_density(i) * domain.ef(i) * ((domain.density * Const::QE * domain.L) / Const::EPS_0);
    }
}



// re-thermalize one particle that crossed the source plane

// Called for ANY particle (ion or electron) that crosses the reflux plane
// going out of the plasma. Plasma lies to the LEFT of x_src, so "into plasma" = -x.
void Species::source_reemit(Particle &p, double dt, double x_src)
{
    const double Pos_Const = domain.vel_norm / (domain.L * domain.W); // keep your unit conversion
    const double vt        = std::sqrt(temp * Const::eV / defaultmass); // source temp kT_s0

    // New parallel speed: untruncated half-Maxwellian (no upper cutoff!)
    // Option B (flux-weighted, particles leaving a thermal reservoir):
    //double speed = std::sqrt(-2.0 * std::log(1.0 - rnd())) * vt;
    // Option A (literal half-Maxwellian), use instead if you want to compare:
    // double speed = std::fabs(Init::SampleVel(*this, temp));

    //p.vx = -speed / domain.vel_norm;          // directed into the plasma (-x)
    p.vx = -vt * std::sqrt(-2.0 * std::log(1.0 - rnd())) / domain.vel_norm;

    // Perpendicular components re-randomized (Maxwellian at source temperature)
    p.vy = Init::SampleVel(*this, temp) / domain.vel_norm;
    p.vz = Init::SampleVel(*this, temp) / domain.vel_norm;

    // Paper Eq. (1): z = v* dt R_N, measured from the plane into the plasma.
    // p.vx is negative, so adding it moves the particle left of x_src.
    p.x = x_src + Pos_Const * p.vx * dt * rnd();

    p.alive = true;
}
/*

void Species::source_reemit(Particle &p, double dt, double x_src)
{
    const double sigma     = std::sqrt(temp * Const::eV / defaultmass);   // [m/s]
    const double Pos_Const = domain.vel_norm / (domain.L * domain.W);


    ///
    //const double x_src    = domain.x0 + domain.xL;
    const double Ls       = 0.10 * domain.xL;      // source band width 
    const double x_lo     = x_src - Ls;
    ///

    // flux-weighted speed at the source temperature, always inward (-x)
    p.vx = -sigma * std::sqrt(-2.0 * std::log(1.0 - rnd())) / domain.vel_norm;
    p.vy = Init::SampleVel(*this, temp) / domain.vel_norm;
    p.vz = Init::SampleVel(*this, temp) / domain.vel_norm;

    // random fraction of a step inward (paper's eq. 1)
    p.x = x_lo + rnd() * Ls;
    //p.x = x_src + rnd() * Pos_Const * p.vx * dt;
    if (p.x >= x_src) p.x = x_src - 1e-9;   // guard against rnd() == 0
    p.alive = true;
}

*/
