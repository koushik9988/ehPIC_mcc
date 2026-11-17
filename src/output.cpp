#include "output.h"
//#include "collision.h"


Output::Output(const std::filesystem::path& outputfolder, Domain& domain) : outputfolder(outputfolder), domain(domain)
{
    //Keep existing outputs/checkpoints when restarting otherwise start clean.
    std::filesystem::create_directories(outputfolder);
    if (!domain.restart)
    {
        std::filesystem::remove_all(outputfolder / "field");
        std::filesystem::remove_all(outputfolder / "phase");
        std::filesystem::remove_all(outputfolder / "logs");
    }

    field_dir = outputfolder / "field";
    phase_dir = outputfolder / "phase";
    log_dir = outputfolder / "logs";
    restartdump_dir = outputfolder / "restart_dump";

    std::filesystem::create_directories(field_dir);
    std::filesystem::create_directories(phase_dir);
    std::filesystem::create_directories(log_dir);
    std::filesystem::create_directories(restartdump_dir);

    //hdf5 throws error if path is passed directly on some system, convert to string first
    std::string metadata_filename = (outputfolder / "metadata.h5").string(); 
    metadata_file = H5File(metadata_filename, H5F_ACC_TRUNC);
    metadata_group = metadata_file.createGroup("/metadata");
    metadata_species = metadata_file.createGroup("/metadata_species");
   

    //open_field_datafile(0); 
    //open_phase_datafile(0);

    //fixed
    int start_ts = domain.restart ? domain.restart_ts : 0;
    //display::print("Starting simulation at timestep: ", start_ts);
    //open_field_datafile(start_ts);
    //open_phase_datafile(start_ts);


    //std::filesystem::path energy_log_path = field_dir / "energy_log.csv";
    //energy_log.open(energy_log_path, std::ios::out | std::ios::trunc);

    int sp_no = domain.species_no;
    int t_step = int(domain.NUM_TS/domain.write_interval) + 1;
    
    //store_ke = Matrix<double>(t_step + domain.restart_ts,3*sp_no + 2);
    //store_m = Matrix<double>(t_step,sp_no + 1);
    //store_m = Matrix<double>(t_step + domain.restart_ts, sp_no * 3 + 1); //new structure to allocate extra memeory for 3-momentum component
    //store_avg_coll_freq = Matrix<double>(t_step + domain.restart_ts, sp_no + 1);
}

void Output::write_dataset_N_2(Group& group, const std::string& name, const std::vector<double> &x_vals, const std::vector<double> &data_vals)
{
    int n = x_vals.size();
    
    // Create 2D array: n rows × 2 columns
    std::vector<double> data_2d(n * 2);
    for (int i = 0; i < n; i++)
    {
        data_2d[i * 2 + 0] = x_vals[i];      // Column 0: x position
        data_2d[i * 2 + 1] = data_vals[i];   // Column 1: value
    }
    
    // Create dataset dimensions [rows, cols]
    hsize_t dims[2] = {static_cast<hsize_t>(n), 2};
    DataSpace dataspace(2, dims);
    
    // Create and write dataset
    DataSet dataset = group.createDataSet(name, PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(data_2d.data(), PredType::NATIVE_DOUBLE);
}

/*
/////fixes theste fxeds suggested by claude but i think tey are not needed so for time being i commnetingthese and keeping my old codes
void Output::open_field_datafile(int ts)
{
    if (ts == current_field_ts)
        return;

    // Close child group BEFORE closing/replacing the parent file
    if (field_data_group.getId() >= 0)
        field_data_group.close();           // ← ADD THIS

    if (field_file.getId() >= 0)
        field_file.close();

    std::string filename = (field_dir / ("data_" + std::to_string(ts) + ".h5")).string();
    field_file = H5File(filename, H5F_ACC_TRUNC);
    field_data_group = field_file.createGroup("/fielddata");

    current_field_ts = ts;
}


void Output::open_phase_datafile(int ts)
{
    if (ts == current_phase_ts)
        return;

    // Close all child groups that belong to phase_file BEFORE closing the file
    for (auto& [key, grp] : particle_groups)
        if (grp.getId() >= 0) grp.close();

    for (auto& [key, grp] : den_subgroups)
        if (grp.getId() >= 0) grp.close();

    for (auto& [key, grp] : vel_subgroups)
        if (grp.getId() >= 0) grp.close();

    for (auto& [key, grp] : collrate_subgroups)
        if (grp.getId() >= 0) grp.close();

    for (auto& [key, grp] : heating_subgroups)
        if (grp.getId() >= 0) grp.close();

    for (auto& [key, grp] : current_subgroups)
        if (grp.getId() >= 0) grp.close();


    particle_groups.clear();
    den_subgroups.clear();
    vel_subgroups.clear();
    collrate_subgroups.clear();
    heating_subgroups.clear();
    current_subgroups.clear();
    // Now safe to close the file
    if (phase_file.getId() >= 0)
        phase_file.close();

    std::string filename = (phase_dir / ("phase_" + std::to_string(ts) + ".h5")).string();
    phase_file = H5File(filename, H5F_ACC_TRUNC);

    current_phase_ts = ts;
}

/////
*/

///*
void Output::open_field_datafile(int ts)
{
    //check if  Already open for this timestep
    if (ts == current_field_ts)
    {
        //skip function if already open for this timestep
        return;
    }
        
    // Close previous file
    if (field_file.getId() >= 0)
    {
        field_file.close();
    }
        
    // Create new file, eg: field/data_100.h5
    std::string filename = (field_dir / ("data_" + std::to_string(ts) + ".h5")).string();
    field_file = H5File(filename, H5F_ACC_TRUNC);

    // Create groups
    field_data_group = field_file.createGroup("/fielddata");
    
    current_field_ts = ts;
}



void Output::open_phase_datafile(int ts)
{
    // Already open for this timestep?
    if (ts == current_phase_ts)
        return;

    // Close previous file
    if (phase_file.getId() >= 0)
        phase_file.close();

    // Create new file: phase/phase_100.h5
    std::string filename = (phase_dir / ("phase_" + std::to_string(ts) + ".h5")).string();
    phase_file = H5File(filename, H5F_ACC_TRUNC);

    current_phase_ts = ts;
}

//*/

void Output::write_field_data(int ts)
{
    open_field_datafile(ts);
    
    int ni = domain.ni;
    
    // Prepare x positions (common for all fields)
    std::vector<double> x_positions(ni);
    for (int i = 0; i < ni; i++)
        x_positions[i] = i * domain.dx;
    
    // --- Electric Field ---
    {
        Group efield_group;
        if (!field_data_group.exists("efield"))
            efield_group = field_data_group.createGroup("efield");
        else
            efield_group = field_data_group.openGroup("efield");
        
        std::vector<double> efield(ni);
        for (int i = 0; i < ni; i++)
            efield[i] = domain.ef(i);
        
        write_dataset_N_2(efield_group, "efield", x_positions, efield);
    }
    
    // --- Potential ---
    {
        Group pot_group;
        if (!field_data_group.exists("pot"))
            pot_group = field_data_group.createGroup("pot");
        else
            pot_group = field_data_group.openGroup("pot");
        
        std::vector<double> potential(ni);
        for (int i = 0; i < ni; i++)
            potential[i] = domain.phi(i);
        
        write_dataset_N_2(pot_group, "potential", x_positions, potential);
    }
    
    // --- Charge Density ---
    {
        Group rho_group;
        if (!field_data_group.exists("rho"))
            rho_group = field_data_group.createGroup("rho");
        else
            rho_group = field_data_group.openGroup("rho");
        
        std::vector<double> charge_density(ni);
        for (int i = 0; i < ni; i++)
            charge_density[i] = domain.rho(i);
        
        write_dataset_N_2(rho_group, "rho", x_positions, charge_density);
    }
}

void Output::write_heating_data(int ts, Species& species)
{
    open_field_datafile(ts);
    
    // Create or get heating group
    std::string group_name = "heating";
    Group heating_group;
    
    if (!field_data_group.exists(group_name))
        heating_group = field_data_group.createGroup(group_name);
    else
        heating_group = field_data_group.openGroup(group_name);
    
    int ni = domain.ni;
    
    // Prepare x positions and heating rate values
    std::vector<double> x_positions(ni);
    std::vector<double> heating(ni);
    
    for (int i = 0; i < ni; i++)
    {
        x_positions[i] = i * domain.dx;
        heating[i] = species.power_rate(i);
    }
    
    // Write as 2-column dataset
    write_dataset_N_2(heating_group, species.name, x_positions, heating);
}


void Output::write_current_density_data(int ts, Species& species)
{
    open_field_datafile(ts);
    
    // Create or get current group
    std::string group_name = "current";
    Group current_group;
    
    if (!field_data_group.exists(group_name))
        current_group = field_data_group.createGroup(group_name);
    else
        current_group = field_data_group.openGroup(group_name);
    
    int ni = domain.ni;
    
    // Prepare x positions and current density values
    std::vector<double> x_positions(ni);
    std::vector<double> current(ni);
    
    for (int i = 0; i < ni; i++)
    {
        x_positions[i] = i * domain.dx;
        current[i] = species.current_density(i);
    }
    
    // Write as 2-column dataset
    write_dataset_N_2(current_group, species.name, x_positions, current);
}


//-----------------den data----------------------------------------

void Output::write_den_data(int ts, Species& species)
{
    open_field_datafile(ts);
    
    // Create or get density group
    std::string group_name = "den_" + species.name;

    Group den_group;
    
    if (!field_data_group.exists(group_name))
    {
        den_group = field_data_group.createGroup(group_name);
    }
    else
    {
        den_group = field_data_group.openGroup(group_name);
    }
    
    int ni = domain.ni;
    
    // Prepare x positions and density values
    std::vector<double> x_positions(ni);
    std::vector<double> density(ni);
    
    for (int i = 0; i < ni; i++)
    {
        x_positions[i] = i * domain.dx;  // x position
        density[i] = species.den(i);      // density value
    }
    
    // Write as 2-column dataset
    write_dataset_N_2(den_group, species.name, x_positions, density);
}


void Output::write_vel_data(int ts, Species& species)
{
    open_field_datafile(ts);
    
    // Create or get velocity group
    std::string group_name = "vel_" + species.name;
    Group vel_group;
    
    if (!field_data_group.exists(group_name))
    {
        vel_group = field_data_group.createGroup(group_name);
    }
    else
    {
        vel_group = field_data_group.openGroup(group_name);
    }
    
    int ni = domain.ni;
    
    // Prepare x positions and velocity values
    std::vector<double> x_positions(ni);
    std::vector<double> velocity(ni);
    
    for (int i = 0; i < ni; i++)
    {
        x_positions[i] = i * domain.dx;
        velocity[i] = species.velmesh(i); //mesh velocity at cell centers
    }
    
    // Write as 2-column dataset
    write_dataset_N_2(vel_group, species.name, x_positions, velocity);
}

void Output::write_collrate_data(int ts, Species& species)
{
    open_field_datafile(ts);
    
    // Create or get collision rate group
    std::string group_name = "collrate_" + species.name;
    Group collrate_group;
    
    if (!field_data_group.exists(group_name))
    {
        collrate_group = field_data_group.createGroup(group_name);
    }
    else
    {
        collrate_group = field_data_group.openGroup(group_name);
    }
    
    int ni = domain.ni;
    
    // Prepare x positions and collision rate values
    std::vector<double> x_positions(ni);
    std::vector<double> collrate(ni);
    
    for (int i = 0; i < ni; i++)
    {
        x_positions[i] = i * domain.dx;
        collrate[i] = species.coll_rate(i);
    }
    
    // Write as 2-column dataset
    write_dataset_N_2(collrate_group, species.name, x_positions, collrate);
}



void Output::write_particle_data(int ts, Species& species)
{
    open_phase_datafile(ts);


    if (!phase_file.attrExists("ts"))
    {
        phase_file.createAttribute("ts", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &ts);
    }
    
    // Create group for this species
    Group species_group = phase_file.createGroup(species.name);
    
    // Prepare phase space data [x, vx, vy, vz, id]
    int num_particles = species.part_list.size();
    std::vector<double> phase_data;
    phase_data.reserve(num_particles * 5);
    
    for (Particle& p : species.part_list)
    {
        phase_data.push_back(p.x);
        phase_data.push_back(p.vx);
        phase_data.push_back(p.vy);
        phase_data.push_back(p.vz);
        phase_data.push_back(p.id);
    }
    
    // Create dataset dimensions [num_particles, 5]
    hsize_t dims[2] = {static_cast<hsize_t>(num_particles), 5};
    DataSpace dataspace(2, dims);
    
    // Write dataset
    DataSet dataset = species_group.createDataSet("phase", PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(phase_data.data(), PredType::NATIVE_DOUBLE);
}

// ============================================================================
// WRITE METADATA
// ============================================================================
void Output::write_metadata(int NC, int NUM_TS, int write_int, int write_int_phase, 
                           double DT, double density, int save_fig, int normscheme, 
                           int subcycleint, double LDe, double LDi, double wpe, double wpi,
                           int spno, double GAS_DENSITY, double max_electron_collision_freq, 
                           double B, double theta, double azimuth)
{
    metadata_group.createAttribute("NC", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &NC);
    metadata_group.createAttribute("NUM_TS", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &NUM_TS);
    metadata_group.createAttribute("write_int", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &write_int);
    metadata_group.createAttribute("write_int_phase", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &write_int_phase);
    metadata_group.createAttribute("DT_coeff", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &DT);
    metadata_group.createAttribute("density", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &density);
    metadata_group.createAttribute("save_fig", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &save_fig);
    metadata_group.createAttribute("norm_scheme", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &normscheme);
    metadata_group.createAttribute("sub_cycle_interval", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &subcycleint);
    metadata_group.createAttribute("LDe", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &LDe);
    metadata_group.createAttribute("LDi", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &LDi);
    metadata_group.createAttribute("wpe", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &wpe);
    metadata_group.createAttribute("wpi", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &wpi);
    metadata_group.createAttribute("spno", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &spno);
    metadata_group.createAttribute("GAS_DENSITY", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &GAS_DENSITY);
    metadata_group.createAttribute("max_ele_coll_freq", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &max_electron_collision_freq);
    metadata_group.createAttribute("B", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &B);
    metadata_group.createAttribute("theta", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &theta);
    metadata_group.createAttribute("azimuth", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &azimuth);

    metadata_group.close();
}

// ============================================================================
// WRITE SPECIES METADATA
// ============================================================================
void Output::write_species_metadata(std::vector<Species>& species_list)
{
    std::vector<std::string> species_order;
    StrType strdatatype(PredType::C_S1, H5T_VARIABLE);
    
    for (Species& sp : species_list)
    {
        Group species_group = metadata_species.createGroup(sp.name);
        
        species_group.createAttribute("name", strdatatype, DataSpace(H5S_SCALAR)).write(strdatatype, sp.name);
        species_group.createAttribute("mass", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &sp.defaultmass);
        species_group.createAttribute("charge", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &sp.charge);
        species_group.createAttribute("spwt", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &sp.defaultspwt);
        species_group.createAttribute("temperature", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &sp.temp);
        species_group.createAttribute("density", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &sp.fract_den);
        species_group.createAttribute("num_particles", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &sp.numparticle);
        species_group.createAttribute("streaming_velocity", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &sp.vs);
        species_group.createAttribute("pos_init", strdatatype, DataSpace(H5S_SCALAR)).write(strdatatype, sp.initialization_pos);
        species_group.createAttribute("vel_init", strdatatype, DataSpace(H5S_SCALAR)).write(strdatatype, sp.initialization_vel);
        
        species_order.push_back(sp.name);
        species_group.close();
    }
    
    // Store species order
    hsize_t dims[1] = {species_order.size()};
    DataSpace order_space(H5S_SIMPLE, dims);
    Attribute order_attr = metadata_species.createAttribute("species_order", strdatatype, order_space);
    
    std::vector<const char*> species_name_pointers;
    for (const auto& name : species_order)
        species_name_pointers.push_back(name.c_str());
    
    order_attr.write(strdatatype, species_name_pointers.data());


    // Flush and close the metadata file immediately so it is safely on disk
    // before the simulation loop starts. This protects against data loss
    // from power outages or crashes during the run.
    metadata_species.close();
    metadata_file.flush(H5F_SCOPE_GLOBAL);
    metadata_file.close();
}


void Output::printmatrix(int row, int col, double **matrix)
{
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            std::cout << matrix[i][j] <<"\t\t";
        }
        std::cout << std::endl;
    }
}

/*
void Output::write_extra(double z1, double z2, double z3)
{
    // Create a subgroup under metadata or fielddata
    Group extra_group = file.createGroup("/extra_results");

    // Write attributes
    extra_group.createAttribute("z1", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &z1);

    extra_group.createAttribute("z2", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &z2);

    extra_group.createAttribute("z3", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &z3);

    extra_group.close();
}
*/

void Output::diagnostics(int ts, std::vector<Species> &species_list)
{
    if (domain.diagtype == "off")
    {
        std::cout << "TS: " << ts << std::endl;
        return;
    }

    Species &norm_species = (domain.normscheme == 2 || domain.normscheme == 4) ? species_list[1] : species_list[0];
    
    if (domain.diagtype == "basic")
    {
        double max_phi = domain.phi(0);
        for (int i = 0; i < domain.ni; i++)
        {
            if (domain.phi(i) > max_phi)
            {
                max_phi = domain.phi(i);
            }
        }

        if (domain.SolverType == "direct" || domain.SolverType == "spectral")
        {
            std::cout << "TS: " << ts << " \t max_phi: " << std::fixed << std::setprecision(2) << (max_phi - domain.phi(0));
        }
        else
        {
            std::cout << "TS: " << ts << "\t" << "norm:" << domain.norm << " delta_phi: " << std::fixed << std::setprecision(2) << (max_phi - domain.phi(0));
        }

        if (domain.bc == "open" || domain.enable_ionization_collision == true)
        {
            for (Species& sp : species_list)
            {
                std::cout << " n_" << std::setw(4) << sp.name << ":" << sp.part_list.size();
            }
        }

        if (domain.normscheme == 5)
        {
            std::cout << std::scientific << std::setprecision(precision);
        }
        else
        {
            std::cout << std::fixed << std::setprecision(precision);
        }

        std::cout << std::endl;
    }

    if (domain.diagtype == "full")
    {
        double max_phi = domain.phi(0);
        for (int i = 0; i < domain.ni; i++)
        {
            if (domain.phi(i) > max_phi)
            {
                max_phi = domain.phi(i);
            }
        }

        if (domain.SolverType == "direct" || domain.SolverType == "spectral")
        {
            std::cout << "TS: " << ts << " \t max_phi: " << std::fixed << std::setprecision(2) << std::setw(6)<<(max_phi - domain.phi(0));
        }
        else
        {
            std::cout << "TS: " << ts << "\t" << "norm:" << domain.norm << " delta_phi: " << std::fixed << std::setprecision(2) << std::setw(6)<< (max_phi - domain.phi(0));
        }

        if (domain.bc == "open" || domain.bc == "dirneu" || domain.GAS_DENSITY > 0)
        {
            for (Species& sp : species_list)
            {
                if(sp.charge == 0)
                    continue;
                std::cout << " n_" << std::setw(6) << sp.name << ":" << sp.part_list.size();
            }

            double average_coll_freq = domain.avg_coll_freq / domain.W;
            std::cout << " \u03BD: " << average_coll_freq;
        }
        
        if (domain.enable_e_detach_collision == true || domain.bc == "dirneu")
        {
            std::cout << " alpha:" << std::setw(6)<<domain.electronegativity;
        }

        if (domain.normscheme == 5)
        {
            std::cout << std::scientific << std::setprecision(precision);
        }
        else
        {
            std::cout << std::fixed << std::setprecision(precision);
        }

        double total_kinetic_energy = 0.0;
        vec<double> total_ke_components(3);
        double potential_energy_val;

        if ((domain.bc == "pbc" || domain.bc == "rbc") || Energy_plot == 1)
        {
            potential_energy_val = domain.ComputePE(norm_species);
            for (Species& sp : species_list)
            {
                vec<double> ke = sp.Compute_KE(norm_species);
                total_ke_components += ke;
                total_kinetic_energy += ke(0) + ke(1) + ke(2);
            }

            std::cout << " KE_: " <<std::setw(10)<< total_kinetic_energy;
            std::cout << " PE_: " << std::setw(10)<<potential_energy_val;
            std::cout << " TE_: " << std::setw(10)<<total_kinetic_energy + potential_energy_val;
            //std::cout << " delta_g:" << std::setw(10)<<domain.delta_g;
            //double average_coll_freq = domain.avg_coll_freq / domain.W;
            //std::cout << " \u03BD: " << average_coll_freq;

#ifdef ENABLE_PLOTTING
            nu_avg.push_back(average_coll_freq);
            time_steps.push_back(static_cast<double>(ts));
            kinetic_energy.push_back(total_kinetic_energy);
            potential_energy.push_back(potential_energy_val);
            total_energy.push_back(total_kinetic_energy + potential_energy_val);
            Ke_x.push_back(total_ke_components(0));
            Ke_y.push_back(total_ke_components(1));
            Ke_z.push_back(total_ke_components(2));
#endif
        }
        std::cout << std::endl;

#ifdef ENABLE_PLOTTING
        // Plotting with matplotlibcpp
        static std::vector<double> lenght;
        static std::vector<double> ke;
        static std::vector<double> pe;
        static std::vector<double> total_energie;
        static std::vector<double> phi;
        static std::vector<double> charge_density;
        static std::vector<double> efield;
        static std::vector<double> pot;
        static std::vector<int> num1;
        static std::vector<int> num2;
        static std::vector<double> k;
        static std::vector<double> rho_k;

        std::vector<double> x;
        std::vector<double> vx;

        if (Energy_plot == 1)
        {
            ke.push_back(total_kinetic_energy);
            pe.push_back(potential_energy_val);
            total_energie.push_back(total_kinetic_energy + potential_energy_val);
        }

        if (phase_plot == 1)
        {
            for (auto &part : species_list[species_index].part_list)
            {
                x.push_back(part.x);
                vx.push_back(part.vx);
            }
        }

        if (Chargedensity_plot == 1)
        {
            lenght.clear();
            charge_density.clear(); 
            for (int i = 0; i < domain.ni; i++)
            {
                lenght.push_back(i * domain.dx);
                charge_density.push_back(domain.rho(i));
            }   
        }

        if (Potentialfield_plot == 1 || domain.bc == "open")
        {
            lenght.clear();
            efield.clear();
            pot.clear();
            for (int i = 0; i < domain.ni; i++)
            {
                lenght.push_back(i * domain.dx);
                efield.push_back(domain.ef(i));
                pot.push_back(domain.phi(i));
            }   
        }

        if (dft_flag == 1 && domain.SolverType == "spectral")
        {
            k.clear();
            rho_k.clear();
            for (int i = 0; i < (domain.ni / 2) + 1; i++)
            {
                k.push_back(domain.dft_k(i));
                rho_k.push_back(domain.dft_value(i));
            }
        }

        plt::ion(); 

        if (Energy_plot == 1 && (domain.bc == "pbc" || domain.bc == "rbc"))
        {
            plt::figure(1);
            plt::clf();
            if (keflag == 1 && (domain.bc == "pbc" || domain.bc == "rbc") && domain.diagtype == "full")
            {
                plt::named_plot("kinetic energy", time_steps, kinetic_energy, "r-");
            }
            if (peflag == 1 && (domain.bc == "pbc" || domain.bc == "rbc") && domain.diagtype == "full")
            {
                plt::named_plot("potential energy", time_steps, potential_energy, "b-");
            }
            if (teflag == 1 && (domain.bc == "pbc" || domain.bc == "rbc") && domain.diagtype == "full")
            {
                plt::named_plot("total energy", time_steps, total_energy, "g-");
            }
            plt::xlabel("Time");
            plt::ylabel("Energy");
            plt::legend();
        }

        double marker_size = 1.0;
        std::map<std::string, std::string> style_options1 = {{"color", "blue"}, {"marker", "o"}};   
        std::map<std::string, std::string> style_options2 = {{"color", "red"}, {"marker", "o"}};

        if (phase_plot == 1)
        {   
            std::string label1 = species_list[species_index].name;
            std::map<std::string, std::string> scatter_keywords1;
            scatter_keywords1["label"] = label1;
            scatter_keywords1["color"] = "black";

            plt::figure(2);
            plt::clf();
            plt::scatter(x, vx, marker_size, scatter_keywords1);
            plt::xlim(0, domain.ni);
            plt::xlabel("x");
            plt::ylabel("v");
            plt::legend({{"loc", "upper right"}}); 
            plt::show(); 
        }

        if (Chargedensity_plot == 1)
        {   
            plt::figure(3);
            plt::clf();
            plt::named_plot("charge-density", lenght, charge_density, "r-");
            plt::xlabel("x");
            plt::ylabel("rho");
            plt::legend(); 
        }

        if (Potentialfield_plot == 1 || domain.bc == "open")
        {
            plt::figure(4);
            plt::clf();
            plt::named_plot("potential", lenght, pot, "r-");
            plt::xlabel("x");
            plt::ylabel("phi/Efield");
            plt::legend(); 
        }

        if (dft_flag == 1 && domain.SolverType == "spectral")
        {
            plt::figure(6);
            plt::clf();
            plt::named_plot("fourier transformed charge density", k, rho_k, "r-");
            plt::xlabel("k");
            plt::ylabel("rho_k");
            plt::legend(); 
        }

        if (ke_components == 1)
        {
            plt::figure(7);
            plt::clf();
            plt::plot(time_steps, Ke_x, {{"label", "KE_x"}, {"color", "blue"}});
            plt::plot(time_steps, Ke_y, {{"label", "KE_y"}, {"color", "red"}});
            plt::plot(time_steps, Ke_z, {{"label", "KE_z"}, {"color", "green"}});
            plt::title("Kinetic Energy Components vs Time");
            plt::xlabel("Time Step");
            plt::ylabel("Kinetic Energy component");
            plt::legend({{"loc", "upper right"}});
        }
        
        if (coll_freq_plot == 1)
        {
            plt::figure(8);
            plt::clf();
            plt::named_plot("average collision frequency", time_steps, nu_avg, "r-");
            plt::title("Average Collision Frequency vs Time");
            plt::xlabel("Time Step");
            plt::ylabel("\u03BD");
            plt::legend({{"loc", "upper right"}});
        }
        
        plt::pause(0.1);
        plt::show();
#endif
    }
}

void Output::init_data_logs(std::vector<Species>& species_list)
{
    // Determine open mode based on restart flag
    std::ios_base::openmode mode = domain.restart ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc);
    
    // Energy log (ts, time, KE_total, PE, TE)
    std::filesystem::path energy_log_path = log_dir / "energy_log.csv";

    energy_log.open(energy_log_path, mode);
    if (!domain.restart)
    {
        energy_log << "#ts\ttime\tKE_total\tPE\tTE\n";
    }
    energy_log.flush();
    
    //KE components log (ts, time, KE_x, KE_y, KE_z per species, PE, KE_total, TE)
    std::filesystem::path ke_log_path = log_dir / "ke_components_log.csv";
    ke_log.open(ke_log_path, mode);
    if (!domain.restart)
    {
        ke_log << "#ts\ttime";
        // Headers will be completed when species info is available
    }
    ke_log.flush();
    
    // Momentum log (ts, time, px, py, pz per species)
    std::filesystem::path momentum_log_path = log_dir / "momentum_log.csv";
    momentum_log.open(momentum_log_path, mode);
    if (!domain.restart)
    {
        momentum_log << "#ts\ttime";
        // Headers will be completed when species info is available
    }
    momentum_log.flush();
    
    // Collision frequency log (ts, time, species_name, avg_coll_freq)
    std::filesystem::path collfreq_log_path = log_dir / "collision_freq_log.csv";
    collfreq_log.open(collfreq_log_path, mode);
    if (!domain.restart)
    {
        collfreq_log << "#ts\ttime";
        for (Species& sp : species_list) // needs species_list passed in, or set header in datafile_headers()
            collfreq_log << '\t' << sp.name << "_coll_freq";
        collfreq_log << '\n';
    }
    collfreq_log.flush();
    
    // Domain average collision frequency log (ts, time, domain_avg_coll_freq)
    std::filesystem::path domain_collfreq_log_path = log_dir / "domain_collision_freq_log.csv";
    domain_collfreq_log.open(domain_collfreq_log_path, mode);
    if (!domain.restart)
    {
        domain_collfreq_log << "#ts\ttime\tavg_coll_freq\n";
    }
    domain_collfreq_log.flush();
    
    // Electronegativity (alpha) log (ts, time, alpha)
    std::filesystem::path alpha_log_path = log_dir / "electronegativity_log.csv";
    alpha_log.open(alpha_log_path, mode);
    if (!domain.restart)
    {
        alpha_log << "#ts\ttime\talpha\n";
    }
    alpha_log.flush();


    // Electronegativity (alpha) log (ts, time, alpha)
    std::filesystem::path particle_log_path = log_dir / "particle_number_log.csv";
    particle_log.open(particle_log_path, mode);
    if (!domain.restart)
    {
        particle_log << "#ts\ttime\tparticle_number\n";
    }
    particle_log.flush();
}


void Output::datafile_headers(std::vector<Species>& species_list)
{
    // Get species names
    std::vector<std::string> species_names;
    for (Species& sp : species_list)
    {
        species_names.push_back(sp.name);
    }
    // Complete KE components log header
    ke_log.seekp(0, std::ios::end);
    for (const auto& name : species_names)
    {
        ke_log << "\t" << name << "_KEx" << "\t" << name << "_KEy" << "\t" << name << "_KEz";
    }
    ke_log << "\tPE\tKE_total\tTE\n";
    ke_log.flush();
    
    //momentum log header
    momentum_log.seekp(0, std::ios::end);
    for (const auto& name : species_names)
    {
        momentum_log << "\t" << name << "_px" << "\t" << name << "_py" << "\t" << name << "_pz";
    }
    momentum_log << "\n";
    momentum_log.flush();
}


void Output::write_energy_log(int ts, std::vector<Species>& species_list)
{
    Species &norm_species = (domain.normscheme == 2 || domain.normscheme == 4) ? species_list[1] : species_list[0];
    
    double time = ts * domain.DT;
    
    // Compute KE components for each species
    double ke_total = 0.0;
    std::vector<double> ke_components;  // For detailed log
    
    for (Species& sp : species_list)
    {
        vec<double> ke = sp.Compute_KE(norm_species);
        
        // Accumulate total KE
        ke_total += ke(0) + ke(1) + ke(2);
        
        // Store components for detailed log
        ke_components.push_back(ke(0));
        ke_components.push_back(ke(1));
        ke_components.push_back(ke(2));
    }
    
    // Compute PE
    double pe = domain.ComputePE(norm_species);
    double total_energy = ke_total + pe;
    
    
    
    //Simple energy log: ts, time, KE_total, PE, TE
    if (energy_log.is_open())
    {
        energy_log << ts << '\t' << time << '\t' << ke_total << '\t' << pe << '\t' << total_energy << '\n'; energy_log.flush();
    }
    
    //Detailed KE components log: ts, time, KEx, KEy, KEz (per species), PE, KE_total, TE
    if (ke_log.is_open())
    {
        ke_log << ts << '\t' << time;
        for (double v : ke_components)
            ke_log << '\t' << v;
        ke_log << '\t' << pe << '\t' << ke_total << '\t' << total_energy << '\n';
        ke_log.flush();
    }
}

void Output::write_momentum_log(int ts, std::vector<Species>& species_list)
{
    Species &norm_species = (domain.normscheme == 2 || domain.normscheme == 4) ? species_list[1] : species_list[0];
    
    double time = ts * domain.DT;

    std::vector<double> p_components;
    
    for (Species& sp : species_list)
    {
        vec<double> pe = sp.Compute_Momentum(norm_species);
        
        // Store for logging
        p_components.push_back(pe(0));
        p_components.push_back(pe(1));
        p_components.push_back(pe(2));
    }
    
    // Momentum log: ts, time, px, py, pz (per species)
    if (momentum_log.is_open())
    {
        momentum_log << ts << '\t' << time;
        for (double v : p_components)
        {
            momentum_log << '\t' << v;
        }
        momentum_log << '\n';
        momentum_log.flush();
    }
}

void Output::write_particle_number_log(int ts, std::vector<Species>& species_list)
{
    double time = ts * domain.DT;
    
    // Particle number log: ts, time, num_particles (per species)
    if (particle_log.is_open())
    {
        particle_log << ts << '\t' << time;
        for (Species& sp : species_list)
        {
            particle_log << '\t' << sp.part_list.size();
        }
        particle_log << '\n';
        particle_log.flush();
    }
}


void Output::write_collision_freq_log(int ts, std::vector<Species>& species_list, CollisionHandler* collisionHandler)
{
    double time = ts * domain.DT;
    
    if (collfreq_log.is_open())
    {
        collfreq_log << ts << '\t' << time;
        for (Species& sp : species_list)
        {
            double avg_coll_freq = collisionHandler->average_collision_frequency(sp);
            collfreq_log << '\t' << avg_coll_freq;  // just the number, no name
        }
        collfreq_log << '\n';
        collfreq_log.flush();
    }
}

/*
void Output::write_collision_freq_log(int ts, Species& species, double avg_coll_freq, std::vector<Species>& species_list)
{
    double time = ts * domain.DT;
    
    // Collision frequency log: ts, time, species_name, avg_coll_freq
    if (collfreq_log.is_open())
    {
        collfreq_log << ts << '\t' << time << '\t' << species.name << '\t' << avg_coll_freq << '\n';
        collfreq_log.flush();
    }
}*/




// ============================================================================
// WRITE DOMAIN AVERAGE COLLISION FREQUENCY TO csv LOG
// ============================================================================
void Output::write_domain_avg_collision_freq(int ts)
{
    double time = ts * domain.DT;
    double avg_coll_freq = domain.avg_coll_freq / domain.W;
    
    // ========================================================================
    // APPEND TO csv LOG
    // ========================================================================
    
    // Domain collision frequency log: ts, time, avg_coll_freq
    if (domain_collfreq_log.is_open())
    {
        domain_collfreq_log << ts << '\t' 
                           << time << '\t' 
                           << avg_coll_freq << '\n';
        domain_collfreq_log.flush();
    }
}

// ============================================================================
// WRITE ELECTRONEGATIVITY (ALPHA) TO csv LOG
// ============================================================================
void Output::write_alpha_to_log(int ts)
{
    double time = ts * domain.DT;
    double alpha = domain.electronegativity;
    
    // ========================================================================
    // APPEND TO csv LOG
    // ========================================================================
    
    // Electronegativity log: ts, time, alpha
    if (alpha_log.is_open())
    {
        alpha_log << ts << '\t' 
                  << time << '\t' 
                  << alpha << '\n';
        alpha_log.flush();
    }
}

/*
void Output::write_dump_file(const std::filesystem::path& dump_path,int ts,std::vector<Species>& species_list)
{
    
    std::string filename = (dump_path / "dump.h5").string();
    H5::H5File dump_file(filename, H5F_ACC_TRUNC);

    Group dump_group = dump_file.createGroup("/dump");
    double time = ts * domain.DT;
    int ni = domain.ni;

    dump_group.createAttribute("ts", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &ts);
    dump_group.createAttribute("time", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &time);
    dump_group.createAttribute("DT", PredType::NATIVE_DOUBLE, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_DOUBLE, &domain.DT);
    dump_group.createAttribute("ni", PredType::NATIVE_INT, DataSpace(H5S_SCALAR)).write(PredType::NATIVE_INT, &ni);

   
    Group field_group = dump_group.createGroup("fields");

    hsize_t dims1d[1] = {static_cast<hsize_t>(ni)};
    DataSpace field_space(1, dims1d);

    // ---- phi
    {
        DataSet dset = field_group.createDataSet("phi", PredType::NATIVE_DOUBLE, field_space);

        std::vector<double> buf(ni);
        for (int i = 0; i < ni; ++i)
            buf[i] = domain.phi(i);

        dset.write(buf.data(), PredType::NATIVE_DOUBLE);
    }

    
    {
        DataSet dset = field_group.createDataSet("ef", PredType::NATIVE_DOUBLE, field_space);

        std::vector<double> buf(ni);
        for (int i = 0; i < ni; ++i)
            buf[i] = domain.ef(i);

        dset.write(buf.data(), PredType::NATIVE_DOUBLE);
    }

    Group particle_group = dump_group.createGroup("particles");

    for (Species& sp : species_list)
    {
        Group sp_group = particle_group.createGroup(sp.name);

        hsize_t np = sp.part_list.size();
        hsize_t dims_phase[2] = {np, 5};  // x vx vy vz id
        DataSpace phase_space(2, dims_phase);

        DataSet dset = sp_group.createDataSet("phase", PredType::NATIVE_DOUBLE, phase_space);

        std::vector<double> buf;
        buf.reserve(np * 5);

        for (Particle& p : sp.part_list)
        {
            buf.push_back(p.x);
            buf.push_back(p.vx);
            buf.push_back(p.vy);
            buf.push_back(p.vz);
            buf.push_back(p.id);
        }

        dset.write(buf.data(), PredType::NATIVE_DOUBLE);
    }

    dump_file.close();

}

*/
