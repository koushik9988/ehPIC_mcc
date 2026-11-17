/* */
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include "collision.h"
#include "domain.h"
#include "emitter.h"
#include "extrafun.h"
#include "field.h"
#include "init.h"
#include "iniparser.h"
#include "output.h"
#include "setup.h"
#include "species.h"


using namespace std; 
using namespace display;

int main( int argc , char *argv[]) 
{     
    ParsedArgs args = parse_cli_args(argc, argv);
    const std::string filename = args.inputfile;
    int num_threads = args.num_threads;
    //globally declared here once but have to check if it is enough?
    omp_set_num_threads(num_threads);
    bool restart_from_dump = args.restart_from_dump;
    std::string restartfile = args.restartfile;

    auto iniData = INIParser::parse(filename);

    //output folder
    std::string outputfolder = INIParser::getString(iniData["file"],"output");

    //grid/domain
    int ni = INIParser::getInt(iniData["domain"], "ni");//grid no
    //int ni = NC+1; // no of grid points is one more than cell no
    double x0 = INIParser::getInt(iniData["domain"],"x0");

    //diagnostic
    int save_fig = INIParser::getDouble(iniData["diagnostics"],"save_fig");
    int write_interval = INIParser::getInt(iniData["diagnostics"],"write_interval");
    int write_interval_phase = INIParser::getInt(iniData["diagnostics"],"write_interval_phase");
    int write_diagnostics = INIParser::getInt(iniData["diagnostics"],"write_diagnostics");
    int sub_cycle_interval = INIParser::getInt(iniData["diagnostics"],"sub_cycle_interval");
    int precision = INIParser::getInt(iniData["diagnostics"],"precision");
    int write_flag = INIParser::getInt(iniData["diagnostics"],"write_flag");
    std::string diagtype = INIParser::getString(iniData["diagnostics"],"diagtype");

    //time
    int NUM_TS = INIParser::getInt(iniData["time"], "NUM_TS");
    double DT_coeff = INIParser::getDouble(iniData["time"],"DT_coeff");
    
    //simulation parameter
    //int see_rate = INIParser::getInt(iniData["simulation"],"see_rate");
    std:: string bc = INIParser::getString(iniData["simulation"],"bc");
    std:: string shapefunction = INIParser::getString(iniData["simulation"],"shapefunction");
    std::string push_parallal = INIParser::getString(iniData["simulation"],"push_parallal");
    std::string deposit_parallal = INIParser::getString(iniData["simulation"],"deposit_parallal");
    double den = INIParser::getDouble(iniData["simulation"],"density");
    double dphi_dx  = INIParser::getInt(iniData["simulation"],"dphi_dx");
    int ionfixed = INIParser::getInt(iniData["simulation"],"ionfixed");
    //double voltage_left = INIParser::getDouble(iniData["simulation"],"voltage_left");
    //double voltage_right = INIParser::getDouble(iniData["simulation"],"voltage_right");



    // boundary potential drive (RF)
    double bc_left_amp = INIParser::getDouble(iniData["boundary_potential"],"voltage_left");
    double bc_left_freq = INIParser::getDouble(iniData["boundary_potential"],"frequency_left");
    double bc_left_phase = INIParser::getDouble(iniData["boundary_potential"],"phase_left");
    double bc_right_amp = INIParser::getDouble(iniData["boundary_potential"],"voltage_right");
    double bc_right_freq = INIParser::getDouble(iniData["boundary_potential"],"frequency_right");
    double bc_right_phase = INIParser::getDouble(iniData["boundary_potential"],"phase_right");
    


    //collision
    std::string elastic_flag = INIParser::getString(iniData["collision"],"elastic");
    std::string excitation_flag = INIParser::getString(iniData["collision"],"excitation");
    std::string ionization_flag = INIParser::getString(iniData["collision"],"ionization");
    std:: string pion_elastic_flag = INIParser::getString(iniData["collision"],"pion_elastic");
    std:: string nion_elastic_flag = INIParser::getString(iniData["collision"],"nion_elastic");
    std:: string e_detach_collision_flag = INIParser::getString(iniData["collision"],"e_detach_collision");
    double GAS_DENSITY = INIParser::getDouble(iniData["collision"],"GAS_DENSITY");
    std::string collgroup = INIParser::getString(iniData["collision"],"collgroup");
    auto collisionPairs = INIParser::parseCollGroup(collgroup);
    std::string GAS_TYPE = INIParser::getString(iniData["collision"],"GAS_TYPE");
    // newww: ion-ion recombination rate [m^3/s], optional key in [collision], 0 = off
    double k_recomb = 0;
    try { k_recomb = INIParser::getDouble(iniData["collision"],"k_recomb"); } catch (...) { k_recomb = 0; }

    
    //normalization
    int norm_scheme = INIParser::getInt(iniData["normalization"],"norm_scheme");
    int vel_normscheme = INIParser::getInt(iniData["normalization"],"vel_norm_scheme");
    double L_scale = INIParser::getDouble(iniData["normalization"],"lenght_scale");
    std::string T_scale = INIParser::getString(iniData["normalization"],"time_scale");

    //potential solver
    double tolerance = INIParser::getDouble(iniData["solver"],"tolerance");
    double solver_state_flag = INIParser::getInt(iniData["solver"],"solver_state");
    double max_iteration = INIParser::getInt(iniData["solver"],"max_iteration");
    std::string SolverType = INIParser::getString(iniData["solver"],"solvertype");


    //external field
    double B = INIParser::getDouble(iniData["ExternalField"],"B");
    double theta = (M_PI / 180.0) * INIParser::getDouble(iniData["ExternalField"],"theta");
    double azimuth = (M_PI / 180.0) * INIParser::getDouble(iniData["ExternalField"],"azimuth");

    double ext_E_field = INIParser::getDouble(iniData["ExternalField"],"E_ext");
    
    //visual plot flag(using matplotlibcpp.h) 
    int Energyplot_flag = INIParser::getInt(iniData["visualplot"],"Energy_plot");
    int chargeplot_flag = INIParser::getInt(iniData["visualplot"],"Chargedensity_plot");
    int Potentialfieldplot_flag = INIParser::getInt(iniData["visualplot"],"Potentialfield_plot");
    int keflag = INIParser::getInt(iniData["visualplot"],"keflag");
    int peflag = INIParser::getInt(iniData["visualplot"],"peflag");
    int teflag = INIParser::getInt(iniData["visualplot"],"teflag");
    int phaseplot_flag = INIParser::getInt(iniData["visualplot"],"phase_plot");
    int species_index = INIParser::getInt(iniData["visualplot"],"species_index");
    int dft_flag = INIParser::getInt(iniData["visualplot"],"dft_rho");
    int ke_components = INIParser::getInt(iniData["visualplot"],"ke_components");
    int coll_freq_plot = INIParser::getInt(iniData["visualplot"],"coll_freq_plot");

    SpeciesConfig species_cfg = load_species_config(iniData, den);
    int species_no = species_cfg.count;

    //noramlizing quantity(electron)
    double LDe = sqrt((Const::EPS_0*Const::K_b*species_cfg.temps[0]*Const::EV_to_K)/(species_cfg.normden[0]*Const::QE*Const::QE)); // Electron Debye Length   
    double wpe = sqrt((species_cfg.normden[0]*Const::QE*Const::QE)/(species_cfg.mass[0]*Const::EPS_0)); // Total Electron Plasma Frequency
    double wpi = sqrt((species_cfg.normden[1]*Const::QE*Const::QE)/(species_cfg.mass[1]*Const::EPS_0)); //ion timescale
    double LDi = sqrt((Const::EPS_0*Const::K_b*species_cfg.temps[1]*Const::EV_to_K)/(species_cfg.normden[1]*Const::QE*Const::QE)); // ion Debye Length
    double CS = sqrt(species_cfg.temps[0]*Const::K_b*Const::EV_to_K/species_cfg.mass[1]); // Ion acoustic speed


    double vthe = LDe*wpe;
    double vthi = LDi*wpi;

    //user defined scales
    double energy_scale = 1;
    double time_scale;
    double lenght_scale = L_scale;
    if(T_scale == "omegape")
    {
        time_scale = wpe; 
    }
    else if(T_scale == "omegapi")
    {
        time_scale = wpi; 
    }
    else if(T_scale != "omegape" || T_scale != "omegapi")
    {
        time_scale = std::stod(T_scale);
    }
 
    double dx;
    double DT;
    double stepSize;
    if(norm_scheme == 2 || norm_scheme == 4)
    {
        stepSize = LDi;
        dx = stepSize/LDi;
        DT = DT_coeff*(1.0/wpi);
        DT = wpi*DT;
    }
    if(norm_scheme == 1 || norm_scheme == 3)
    {
        stepSize = LDe;
        dx = stepSize/LDe;
        DT = DT_coeff*(1.0/wpe);
        DT = wpe*DT;
    }
    
    if(norm_scheme == 5)
    {
        stepSize = lenght_scale;
        dx = stepSize/lenght_scale;
        DT = DT_coeff*(1.0/time_scale);
        DT = time_scale*DT;
    }
    

    Domain domain(x0,dx,ni);

    domain.bc  = bc;
    domain.vel_ratio = vthi/vthe;
    //domain.see_rate = see_rate;
    //domain.tempwall = tempwall;
    domain.shape = shapefunction;
    domain.SolverType = SolverType;
    domain.solver_state = solver_state_flag;
    domain.tolerance = tolerance;
    domain.num_threads = num_threads;
    domain.push_parallal = string_to_bool(push_parallal);
    domain.deposit_parallal = string_to_bool(deposit_parallal);
    domain.species_no = species_no;
    domain.density = den;
    domain.ne0_ref = species_cfg.normden[0]/domain.density;
    domain.IAW_vel = CS;
    domain.restart_ts = 0;
    domain.restart = false;
    if(restart_from_dump)
    {
        domain.restart = true;
        //domain.restart_ts = extract_ts_from_filename(restartfile);

    }
    if(domain.num_threads == 1)
    {
        domain.push_parallal = false;
        domain.deposit_parallal = false;
    }
    domain.max_iteration = max_iteration;
    domain.normscheme = norm_scheme;
    domain.vel_normscheme = vel_normscheme;
    domain.diagtype = diagtype;
    domain.sub_cycle_interval = sub_cycle_interval;
    domain.set_normparam(LDe,wpe,LDi,wpi);
    domain.set_userdefined_normscheme(time_scale,lenght_scale,energy_scale);
    domain.set_time(DT,NUM_TS,write_interval);
    domain.set_normscheme();
    domain.read_types();
    //domain.vL = voltage_left;
    //domain.vR = voltage_right;
    // Normalization factor for potential
    if(species_cfg.temps[0] != 0)
    {
        domain.phi_norm = Const::QE/(Const::K_b * Const::EV_to_K * species_cfg.temps[0]);
    }
    else
    {
        domain.phi_norm = 1;
    }
     
    domain.efieldnorm = (domain.density * Const::QE * domain.L) / Const::EPS_0;

    domain.vL = bc_left_amp;
    domain.fL = bc_left_freq;
    domain.phaseL = bc_left_phase;
    domain.vR = bc_right_amp;
    domain.fR = bc_right_freq;
    domain.phaseR = bc_right_phase;
    domain.grad_phi = dphi_dx;
    
    //collision
    domain.GAS_DENSITY = GAS_DENSITY;
    domain.enable_elastic_collision = string_to_bool(elastic_flag);    // Flag for elastic collisions
    domain.enable_excitation_collision = string_to_bool(excitation_flag); // Flag for excitation collisions
    domain.enable_ionization_collision = string_to_bool(ionization_flag); // Flag for ionization collisions
    domain.enable_pion_elastic = string_to_bool(pion_elastic_flag);
    domain.enable_nion_elastic = string_to_bool(nion_elastic_flag);
    domain.enable_e_detach_collision = string_to_bool(e_detach_collision_flag);
    domain.GAS_TYPE = GAS_TYPE;
    domain.delta_g = 0;
    domain.N_ecoll = 0;
    domain.N_ioncoll = 0;
    domain.N_negioncoll = 0;

    domain.electronegativity = 0;
    
    domain.ionfixed = ionfixed;


    domain.B = B;
    domain.theta = theta;
    domain.azimuth = azimuth;

    domain.ext_E_field = ext_E_field;
    
    std::vector<Species> species_list = create_species(species_cfg, domain, restart_from_dump, restartfile);

    ///
    //double beam_temp = 0.5 * species_list[2].mass * pow(species_list[2].vs * domain.vel_norm, 2) / Const::QE;

    //domain.tau = (sqrt(normden[0]*species_list[2].mass))/(sqrt(normden[2]*species_list[0].mass)) * (species_list[2].temp/beam_temp);
    ///
    
    auto ElectronNeutralCollision = create_collision_handler(iniData, domain);

    std::vector<Emitter> emitters = create_emitters(iniData, domain);

    if(emitters.size() > 0)
    {
        domain.current_density = emitters[0].get_current_density();
    }
    

    domain.display(species_list);

    print("Press Enter to continue...");
    //std::cin.get();

   // In main.cpp, within the main function, modify the Output initialization
    Domain::BoundaryType bc_flag = domain.bc_type;
    Output output(outputfolder, domain);

    output.precision = precision;
    #ifdef ENABLE_PLOTTING
    output.Energy_plot = Energyplot_flag;
    output.Potentialfield_plot = Potentialfieldplot_flag;
    output.Chargedensity_plot = chargeplot_flag;
    output.keflag = keflag;
    output.peflag = peflag;
    output.teflag = teflag;
    output.phase_plot = phaseplot_flag;
    output.dft_flag = dft_flag;
    output.species_index = species_index;
    output.ke_components = ke_components;
    output.coll_freq_plot = coll_freq_plot;
    #else
    output.Energy_plot = 0;
    output.Potentialfield_plot = 0;
    output.Chargedensity_plot = 0;
    output.keflag = 0;
    output.peflag = 0;
    output.teflag = 0;
    output.phase_plot = 0;
    output.dft_flag = 0;
    output.species_index = 0;
    output.ke_components = 0;
    output.coll_freq_plot = 0;
    #endif
    output.write_metadata(ni, NUM_TS, write_interval, write_interval_phase, DT_coeff, den, save_fig, domain.normscheme,
    domain.sub_cycle_interval, LDe, LDi, wpe, wpi, species_no, GAS_DENSITY, domain.max_electron_coll_freq, domain.B, domain.theta, domain.azimuth);
    output.write_species_metadata(species_list);
    
    //std::vector<Emitter> emitters = create_emitters(iniData, domain);

    //domain.current_density = emitters[0].GetCurrentDensity();

    
    auto start_time = std::chrono::high_resolution_clock::now();

    //initializing the species by creating instances of Init class
    for(Species &sp : species_list)
    {
        if (sp.isfluid) continue;
        if (sp.name == "negion" && !restart_from_dump) continue;   // newww: negion starts empty, built up by the 3-species source
        Init init(sp,domain,restartfile);
    } 
    
    //FieldSolve class instance declaration
    FieldSolve fieldsolver(domain);
    
    //scatter species to mesh/grid 
    for (Species &sp:species_list)
	{
		if (sp.isfluid) continue;
        sp.ScatterSpecies();
	}

    //compute charge density on the grid
    domain.ComputeRho(species_list);

    //initial potential and field calculation
    fieldsolver.PotentialSolver(0,bc_flag);
    
    fieldsolver.CalculateEfield(bc_flag);

    //rewind species velocity by half a time step
    for (Species &sp:species_list)
	{
        if (sp.isfluid) continue;
        sp.Rewind_species();
    }

    //double nu = ElectronNeutralCollision->average_collision_frequency(species_list[0]);
    //display::print("avaerage collision frequecny : ", nu);
    output.init_data_logs(species_list);
    output.datafile_headers(species_list);
    //--------------MAIN LOOP----------------------- 
    //int ts = ts + domain.restart_ts;

    

    for(int ts = 0 + domain.restart_ts; ts < NUM_TS + domain.restart_ts + 1; ts++)
    {
        
        if(species_list.size() > 2)
        {
            domain.electronegativity = domain.Calculate_alpha(species_list[0], species_list[2]);
        }
        //else
        //{
            //domain.electronegativity = 0;
        //}
        
        domain.avg_coll_freq = ElectronNeutralCollision->average_collision_frequency(species_list[0]);

        //domain.collision_rate.display();
        
        int start = ts;
        int end = NUM_TS;
       
        //if (ts >= start && ts < end && bc_flag == Domain::BoundaryType::DIRNEU)
        if (bc_flag == Domain::BoundaryType::DIRNEU)
        {
            for (auto &emit : emitters)
            {
                emit.inject(species_list);
            }
        }
        
        for (Species &sp:species_list)
		{
			if (sp.isfluid) continue;
            //if(domain.solver_state == 0) continue;
            sp.ScatterSpecies();
            sp.ScatterVel_serial();
            sp.ComputeCurrentDensity();
		}


        if(domain.solver_state == 1)
        {
            domain.ComputeRho(species_list);
        
            fieldsolver.PotentialSolver(ts,bc_flag);

            fieldsolver.CalculateEfield(bc_flag);

            fieldsolver.fluid_den_update(species_list);
        }

    
        for (auto &sp : species_list)
        {
            if (sp.isfluid) continue;
            sp.ComputeHeatingRate();
        }

        //---------particle-mover-------------------------------------------- 
        for (Species &sp:species_list)
		{
            
            if (sp.isfluid) continue;
            if (sp.name == "ion" && domain.ionfixed == 1) continue;
            if(domain.normscheme == 1 || domain.normscheme == 2 || domain.normscheme == 4 || domain.normscheme == 5)
            {
                sp.Push_species(species_list,1);
            }
            if(domain.normscheme == 3)
            {
                if(sp.name == "electron")
                {
                    sp.Push_species(species_list,1);
                }
                else
                {
                    if(ts%domain.sub_cycle_interval == 0)
                    {
                        sp.Push_species(species_list,sub_cycle_interval);
                    }
                }
            }
        }


        // neww: 3-species source (ion + electron + negion, S+ = Se + S-)
        if (bc_flag == Domain::BoundaryType::DIRNEU && species_list.size() > 1)
        {
            Species *negp = (species_list.size() > 2) ? &species_list[2] : nullptr;
            double alpha_src = negp ? negp->bulkden / species_list[0].bulkden : 0.0;   // alpha = n-/ne from normden
            double neg_frac  = alpha_src / (1.0 + alpha_src);                          // S-/S+
            source_emitter(species_list[1], species_list[0], domain, 1.0, negp, neg_frac);   // 1.0 = emitter area
        }
        //new : volume ion-ion recombination (loss of negion), species 1 = ion, 2 = negion
        if (species_list.size() > 2 && k_recomb > 0)
        {
            ion_ion_recombination(species_list[1], species_list[2], domain, k_recomb);
        }
            
		if(domain.GAS_DENSITY > 0)
        {
            for (const auto& [first, second] : collisionPairs)
            {
                // Skip if either species in the pair is a fluid species — no particles to collide
                if (species_list[first].isfluid || species_list[second].isfluid) continue;
                ElectronNeutralCollision->handle_collisions(species_list[first], species_list[second],species_list[0]);
                ElectronNeutralCollision->coll_rate(species_list[first], species_list[second]);  
            }
            
        }

        //------------------------------------------------------------------------
        
        output.write_energy_log(ts, species_list);
        output.write_momentum_log(ts, species_list);
        //output.write_particle_number_log(ts, species_list);
        if(ts%write_interval== 0)
        {
            if(write_flag == 1 || write_flag == 2)
            {    
                output.write_field_data(ts);
                //output.storeKE_to_matrix(ts,species_list);
                //output.storem_to_matrix(ts,species_list);
                for(Species &sp:species_list)
                {
                    output.write_den_data(ts,sp);
                    if (sp.isfluid) continue;
                    output.write_collrate_data(ts,sp);
                    output.write_heating_data(ts, sp);
                    output.write_current_density_data(ts, sp);
                    output.write_vel_data(ts,sp);
                    //double temp_avg_coll_freq = ElectronNeutralCollision->average_collision_frequency(sp);///
                    output.write_collision_freq_log(ts, species_list, ElectronNeutralCollision.get());
                    //temp_avg_coll_freq = temp_avg_coll_freq/domain.W;
                    //output.store_avg_coll_freq_to_matrix(ts,sp,temp_avg_coll_freq,species_list);/////////

                }    
            }
            output.write_domain_avg_collision_freq(ts);
            output.write_alpha_to_log(ts);
        }
        
        if(ts%write_interval_phase == 0)
        {
            if(write_flag == 1 || write_flag == 3)
            {
                for(Species &sp:species_list)
                {
                    output.write_particle_data(ts,sp);
                }
            }
        }
        
        if(ts%write_diagnostics == 0)
        {
            output.diagnostics(ts,species_list);
        }

		std::filesystem::path dumpoutput = std::filesystem::current_path();
 
    }
    
    //output.write_ke();
    //output.write_m();
    //output.write_avg_coll_freq();/////////

    if(domain.diagtype != "off")
    {
        print("average no of electron crossed one or more than one cell per time step : ",domain.ele_cross/NUM_TS);
        print("average no of ion crossed one or more than  cell per time step : ",domain.ion_cross/NUM_TS,"\n");
        if(domain.ele_cross > 0 )
        {print("average no of cell crossed by electron: ",domain.crossed_cellno_ele/domain.ele_cross,"\n");}
        if(domain.ion_cross > 0 )
        {print("average no of cell crossed by ion: ",domain.crossed_cellno_ion/domain.ion_cross,"\n");}
    
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();

    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    if(domain.diagtype != "off")
    {
        std::cout << "Elapsed time: " << elapsed_time.count() << "seconds." <<"or "<< elapsed_time.count()/60<<"minutes"<<std::endl;
    }
    
    //print(domain.N_pcoll);
    //print("total sim time:",NUM_TS*(DT_coeff/wpe));
    double z1 = static_cast<double>(domain.N_ecoll)/(static_cast<double>(species_cfg.nParticles[0]) * NUM_TS * DT_coeff);
    double z2 = static_cast<double>(domain.N_ioncoll)/(static_cast<double>(species_cfg.nParticles[1]) * NUM_TS * DT_coeff);
    double z3 = 0;//static_cast<double>(domain.N_negioncoll)/((static_cast<double>(nParticles[2]) + nParticles[3]) * NUM_TS * DT_coeff);

    display::print("Average electron collision frequency: ", z1);
    display::print("Average ion collision frequency: ", z2);
    display::print("Average negative ion collision frequency: ", z3);

    //output.write_extra(z1,z2,z3);

    //print("Press Enter to close the simulation...");
    //std::cin.get();

    return 0;
}
