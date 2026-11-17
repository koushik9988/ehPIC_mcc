#include "init.h"

#include "domain.h"
#include "iniparser.h"
#include "species.h"



Init::Init(Species &species, Domain &domain, const std::string& restart_file) : species(species),domain(domain)
{

    auto initilization1 = INIParser::loadtypeextract(species.initialization_pos);
    auto initilization2 = INIParser::loadtypeextract(species.initialization_vel);

    auto [init_type1, n1, amplitude1] = initilization1;
    auto [init_type2, n2, amplitude2] = initilization2;

    //display::print(init_type1);
    //display::print(init_type2, n2, amplitude2);
    //double k_launch = 0.0;


    ////test code
    // -----------------------------------------
    bool dump_mode = (init_type1 == "dump" || init_type2 == "dump" || init_type1 == "dump+sin" || init_type1 == "dump+cos");

    std::vector<double> dump_phase;  // x vx vy vz id
    hsize_t dump_np = 0;

    if (dump_mode)
    {
        H5File file(restart_file, H5F_ACC_RDONLY);

        std::string dset_path = "/" + species.name + "/phase";//"/dump/particles/" + species.name + "/phase";
        DataSet dset = file.openDataSet(dset_path);

        ////
        //domain.restart_ts;
        // Read timestep from file
        file.openAttribute("ts").read(PredType::NATIVE_INT, &(domain.restart_ts));
        //file.openGroup("/dump").openAttribute("ts").read(H5::PredType::NATIVE_INT, &(domain.restart_ts));
        ////

        DataSpace space = dset.getSpace();
        hsize_t dims[2];
        space.getSimpleExtentDims(dims);

        dump_np = dims[0];

        if (dump_np != static_cast<hsize_t>(species.numparticle))
        {
            //std::cerr << "ERROR: dump particle count mismatch for species "<< species.name << std::endl;
            //exit(-1);
        }

        dump_phase.resize(dump_np * 5);
        dset.read(dump_phase.data(), PredType::NATIVE_DOUBLE);

        file.close();

        //display::print("Restarting sim species:", species.name," with particles:", dump_np);
        // Overwrite particle count from dump
        species.numparticle = dump_np;
        display::print("Loaded particles from dump file.",dump_np);

    }

    //////

    for (int p = 0; p < species.numparticle; p++) 
    {
        double x = 0;
        double vx = 0, vy = 0, vz = 0;

        int idx = 5 * p;

        // ----- Position initialization -----
        if (init_type1 == "random") 
        {
            x = domain.x0 + domain.xL * rnd();
        }
        
        else if (init_type1 == "uniform") 
        {
            x = domain.x0 + p * (domain.xL / (species.numparticle - 0));
            // earlier verison line
            // x = domain.x0 + p * (domain.xL / (species.numparticle - 0)); 
            //was causing a random bug when 3 species was negative and its density was 0.4 
            // dodnow know why but (species.numparticle - 1)=> (species.numparticle ) foxed the issue.
        }
        else if (init_type1 == "sin" || init_type1 == "cos") 
        {
            ///*
            //acceptance-rejection sampling (may better reproduce density)
            double k = 2 * Const::PI * n1 / domain.xL;
            double A = amplitude1;

            //display::print("k value:", k);

            double pdf_env = 1/(1 + A);

            while(true)
            {
                double x_try = domain.x0 + domain.xL * rnd();

                double s = (init_type1 == "sin") ? sin(k * x_try) : cos(k * x_try);

                double pdf = (1.0 + A * s) * pdf_env;

                if (rnd() < pdf)
                {
                    x = x_try;
                    break;
                }
            }
            
            /*

            double x0 = domain.xL * rnd();;///domain.x0 + p * (domain.xL / (species.numparticle - 1));
            double k = 2 * Const::PI * n1 / domain.xL;
            double A = amplitude1;

            if (init_type1 == "sin")
            {
                x = x0 + A * sin(k * x0);
            } 
            else
            {
                x = x0 + A * cos(k * x0);
            }
            
            */
            

        }
        /*
        else if (init_type1 == "extend")
        {
            int start = static_cast<int>(n1);
            int end = static_cast<int>(amplitude1);
            x = start + (end - start) * rnd();  // uniform random in [start, end]
        }*/

        else if (init_type1 == "extend")
        {
            double start = n1;
            double end   = amplitude1;

            x = start + (end - start) * static_cast<double>(p) / (species.numparticle - 1);
        }

        //test code
        ///
        else if (init_type1 == "dump")
        {
            x  = dump_phase[idx + 0];
            //display::print("Loaded particle position:", x);
        }
        else if (init_type1 == "dump+sin" || init_type1 == "dump+cos")
        {
            x  = dump_phase[idx + 0];

            //double k = 2 * Const::PI * n1 / domain.xL;
            double k = 2 * Const::PI * n1 / (domain.xL);
            //k_launch = k;

            double A = amplitude1;

            if (init_type1 == "dump+sin")
            {
                x += A * sin(k * x);
            }   
            else
            {
                x += A * cos(k * x);
            }    
        }

        else if (init_type1 == "rs_tanh+sin")
        {
            
            double x_random;
            //construct density profile
            double RB = domain.x0 + 0.25 * domain.xL;  // left boundary/edge
            double LB = domain.x0 + 0.75 * domain.xL;  // right boundary/edge
            double d  = 0.05 * domain.xL;              // interface width

            while (true)
            {
                // pick a random x and check whether it fit in the profile we need or not /comapre againt a random number 
                x_random = domain.x0 + rnd()*domain.xL;
                double rho = 0.5 * (tanh((x_random - RB)/d)-tanh((x_random - LB)/d)); //multiply by half such that value varies between 0 to 1
                if (rnd() < rho)
                {
                    break;
                }
            }
            ///*
            double A = amplitude1;                 // perturbation amplitude
            //double k = (2.0 * M_PI * 1) / (LB - RB);
            double k =  2 * Const::PI * n1 / (domain.xL);   

            x = x_random; //uncomment#when removed test

            //@if (x > RB && x < LB)
            //@{
                //@x += A * std::sin(k * (x - RB));
                x += A * sin(k*x); //uncomment# when removed test
            //@}

        
            
            //*/
            //x = x_random;
        }

        else if (init_type1 == "rs_tanh")
        {
            
            double x_random;
            //construct density profile
            double RB = domain.x0 + 0.25 * domain.xL;  // left boundary/edge
            double LB = domain.x0 + 0.75 * domain.xL;  // right boundary/edge
            double d  = 0.05 * domain.xL;              // interface width

            while (true)
            {
                // pick a random x and check whether it fit in the profile we need or not /comapre againt a random number 
                x_random = domain.x0 + rnd()*domain.xL;
                double rho = 0.5 * (tanh((x_random - RB)/d)-tanh((x_random - LB)/d)); //multiply by half such that value varies between 0 to 1
                if (rnd() < rho)
                {
                    break;
                }
            }
            x = x_random;
        }


        else if (init_type1 == "tanh")
        {
            // Here we have use inverse CDF method. We cannot map directly from PDF as PDF itself doesnot give probababilty value.
            // CDF(x) = u means at what x does the distribution reach probability u
            // So we can invert take a probabilty u to find x foe which we can simple use root finding method CDF(x) - u = 0
            // So simply we map CDF value to x to find our desired distribution
            // Prevously we have used direct pdf to x map using monte carlo method but it introduces soem initial non zero charge density
            // So in order to have a quite start we have to use CDF inverting method

            double LB = domain.x0 + 0.25 * domain.xL;  // left boundary
            double RB = domain.x0 + 0.75 * domain.xL;  // right boundary
            double d  = 0.05 * domain.xL;

            // CDF of the tanh density profile
            //integral of 0.5*(tanh((x-RB)/d) - tanh((x-LB)/d)) dx
            auto cdf = [&](double xs) { return 0.5 * (d * log(cosh((xs - LB) / d)) - d * log(cosh((xs - RB) / d)));};
            

            double cdf0 = cdf(domain.x0);
            double cdf1 = cdf(domain.x0 + domain.xL);
              
            //Generate uniform probability value
            double target_pdf = cdf0 + (cdf1 - cdf0) * (double(p) / (species.numparticle-1));

            //display::print(target_pdf);
            //In order to inverse cdf we have to solve cdf(x)-target_pdf = 0
            //Because cdf maynot always have easy invertable funcional form 
            //So better to use root finding method
            auto root_finding_expre = [&](double xs) {return cdf(xs) - target_pdf;};

            x = bisection(domain.x0, domain.x0 + domain.xL, root_finding_expre);

            //x = bisection(domain.x0, domain.x0 + domain.xL, [&](double xval) {return cdf(xval) - target_pdf;});
            
        }


        else if (init_type1 == "tanh+sin")
        {
            // Here we have use inverse CDF method. We cannot map directly from PDF as PDF itself doesnot give probababilty value.
            // CDF(x) = u means at what x does the distribution reach probability u
            // So we can invert take a probabilty u to find x foe which we can simple use root finding method CDF(x) - u = 0
            // So simply we map CDF value to x to find our desired distribution
            // Prevously we have used direct pdf to x map using monte carlo method but it introduces soem initial non zero charge density
            // So in order to have a quite start we have to use CDF inverting method

            double LB = domain.x0 + 0.25 * domain.xL;  // left boundary
            double RB = domain.x0 + 0.75 * domain.xL;  // right boundary
            double d  = 0.04 * domain.xL;

            // CDF of the tanh density profile
            //integral of 0.5*(tanh((x-RB)/d) - tanh((x-LB)/d)) dx
            auto cdf = [&](double xs) { return 0.5 * (d * log(cosh((xs - LB) / d)) - d * log(cosh((xs - RB) / d)));};

            double cdf0 = cdf(domain.x0);
            double cdf1 = cdf(domain.x0 + domain.xL);
              
            //Generate uniform probability value
            double target_pdf = cdf0 + (cdf1 - cdf0) * (double(p) / (species.numparticle-1));

            //display::print(target_pdf);
            //In order to inverse cdf we have to solve cdf(x)-target_pdf = 0
            //Because cdf maynot always have easy invertable funcional form 
            //So better to use root finding method
            auto root_finding_expre = [&](double xs) {return cdf(xs) - target_pdf;};

            double x0 = bisection(domain.x0, domain.x0 + domain.xL, root_finding_expre);


            // Apply sinusoidal perturbation on top of the quiet-start base position
            double k = 2 * Const::PI * n1 / (std::abs((RB- LB)));
            double A = amplitude1;

            //x = x0 + A * sin(k * x0);
            x = x0;
            //Apply perturbation ONLY inside region
            if (x0 < RB && x0 > LB)
            {
                // shifted so wave starts at LB
                x += A * sin(k * (x0-LB));
            }
        }

        else if (init_type1 == "half_tanh")
        {
            // Right half of a symmetric tanh bump:
            // density is ~1 at x=0, decays smoothly around x = 0.5*xL
            // rho(x) = 0.5 * (1 - tanh((x - mid) / d))

            double mid = domain.x0 + 0.6 * domain.xL;
            double d   = 0.1 * domain.xL;

            auto cdf = [&](double xs) {return 0.5 * (xs - d * log(cosh((xs - mid) / d)));};

            double cdf0 = cdf(domain.x0);
            double cdf1 = cdf(domain.x0 + domain.xL);

            double target_pdf = cdf0 + (cdf1 - cdf0) * (double(p) / (species.numparticle - 1));

            auto root_finding_expre = [&](double xs) {return cdf(xs) - target_pdf;};

            x = bisection(domain.x0, domain.x0 + domain.xL, root_finding_expre);
        }

        //need fixning later
        else if (init_type1 == "half_tanh+sin")
        {
            double mid = domain.x0 + 0.5 * domain.xL;
            double d   = 0.1 * domain.xL;

            auto cdf = [&](double xs) {return 0.5 * (xs - d * log(cosh((xs - mid) / d)));};

            double cdf0 = cdf(domain.x0);
            double cdf1 = cdf(domain.x0 + domain.xL);

            double target_pdf = cdf0 + (cdf1 - cdf0) * ((double(p)) / (species.numparticle - 1));

            auto root_finding_expre = [&](double xs) {return cdf(xs) - target_pdf;};

            double x0 = bisection(domain.x0, domain.x0 + domain.xL, root_finding_expre);

            double k = 2.0 * Const::PI * n1 / mid;
            double A = amplitude1;

            x = x0;

            if (x0 < mid && x0 > 0.3*mid)
            {
                x += A * sin(k * x0);
            }
    
        }


        ////
        else if (init_type1 == "gaussian")
        {
            double x0_gauss = 0.25 * domain.xL;
            double W        = 5.0;
            double A_gauss  = 0.5;

            // CDF function
            auto cdf = [&](double xs)
            {
                double term1 = xs;
                double term2 = A_gauss * W * sqrt(M_PI / 2.0) * erf((xs - x0_gauss) / (sqrt(2.0) * W));
                return term1 + term2;
            };

            double cdf0 = cdf(domain.x0);
            double cdf1 = cdf(domain.x0 + domain.xL);

            // quiet start (same as tanh)
            double u = cdf0 + (cdf1 - cdf0) * (double(p) / (species.numparticle - 1));

            auto root = [&](double xs)
            {
                return cdf(xs) - u;
            };

            x = bisection(domain.x0, domain.x0 + domain.xL, root);
        }

        else if (init_type1 == "sech")
        {
            // Soliton-like density profile: rho(x) = 1 + A * sech^2((x - x0)/W)
            //   A > 0  -> compressive soliton (density bump)
            //   A < 0  -> rarefactive soliton (density dip); keep |A| < 1 so rho stays positive
            double x0_sech = domain.x0 + 0.25 * domain.xL;   // launch location
            double W       = n1;           // pulse width
            double A_sech  = amplitude1;   // amplitude

            auto cdf = [&](double xs)
            {
                double term1 = xs;
                double term2 = A_sech * W * tanh((xs - x0_sech) / W);
                return term1 + term2;
            };

            double cdf0 = cdf(domain.x0);
            double cdf1 = cdf(domain.x0 + domain.xL);

            double u = cdf0 + (cdf1 - cdf0) * (double(p) / (species.numparticle - 1));

            auto root = [&](double xs) { return cdf(xs) - u; };

            x = bisection(domain.x0, domain.x0 + domain.xL, root);
        }


        ////



        //out of bound case
        if (x >= domain.xL)
        {
            x -= domain.xL;
        }
        else if (x < 0)
        {
            x += domain.xL;
        }
        // ----- Velocity initialization -----

        vx = SampleVel(species) + species.vs * domain.vel_norm;

        if (init_type2 == "sin" || init_type2 == "cos") 
        {

            double k = 2 * Const::PI * n2 / domain.xL;
            //k_launch = k;
            
            double A = amplitude2 * domain.vel_norm;

            if (init_type2 == "sin")
            {
               vx += A * sin(k * x);  // NOTE: use x, not x0
            }   
            else
            {
                vx += A * cos(k * x);
            }    
        }



        else if (init_type2 == "sech")
        {
            double x0_sech = domain.x0 + 0.25 * domain.xL;
            double W       = n1;
            double A_sech  = amplitude1;

            double M = 1.6;       // soliton Mach number
            double Cs = domain.IAW_vel/domain.vel_norm;      // normalized ion-acoustic speed

            double sech2 = 1.0 / pow(cosh((x - x0_sech) / W), 2);

            double density_ratio = 1.0 + A_sech * sech2;

            double delta_v = M * Cs * (1.0 - 1.0 / density_ratio);

            vx += delta_v * domain.vel_norm;
        }

        /*
        else if (init_type2 == "sech")
        {
            // Correlated velocity kick so the pulse launches as a single right-moving
            // disturbance instead of splitting into two counter-propagating pulses.
            // Linear relation: delta_v = Cs * (delta_n / n0)
            double x0_sech = domain.x0 + 0.25 * domain.xL;  // MUST match the position branch
            double W       = n1;                            // MUST match the position branch
            double A_sech  = amplitude1;                    // MUST match the position branch

            double Cs = 1; 

            double dn_over_n0 = A_sech / pow(cosh((x - x0_sech) / W), 2);

            vx += Cs * dn_over_n0 * domain.vel_norm;
        }
        */
        else if (init_type2 == "dump")
        {
            vx = dump_phase[idx + 1]*domain.vel_norm;
            vy = dump_phase[idx + 2]*domain.vel_norm;
            vz = dump_phase[idx + 3]*domain.vel_norm;
            //display::print("Loaded particle velocity:", vx, vy, vz);
        }

        
            
        else
        {
            display::print("ERROR! Wrong velcoity init");
            exit(-1);
        }
        // Normalize velocity
        vx /= domain.vel_norm;
        vy /= domain.vel_norm;
        vz /= domain.vel_norm;

        species.AddParticle(Particle(x, vx, vy, vz, 0));
    }
    //display::print("launched wave k:", k_launch);
}

double Init::SampleVel(Species &species)
{
    //double v_th = sqrt(2 * Const::K_b * domain.tempE * Const::EV_to_K / Const::ME);
    double v_th =  sqrt(2*Const::K_b*species.temp*Const::EV_to_K/species.defaultmass);
    double vt = v_th * sqrt(2) * (rnd() + rnd() + rnd() - 1.5) ;//+ domain.v_i*domain.wp * domain.LD;
    //double vt = v_th * sqrt(2) * (rnd()*rnd()*rnd() - 1.5) ;
    return vt;
}

double Init::SampleVel(Species &species, double temp)
{
    //double v_th = sqrt(2 * Const::K_b * domain.tempE * Const::EV_to_K / Const::ME);
    double v_th =  sqrt(2*Const::K_b*temp*Const::EV_to_K/species.defaultmass);
    double vt = v_th * sqrt(2) * (rnd() + rnd() + rnd() - 1.5); //+ domain.v_i*domain.wp * domain.LD;
    return vt;
}