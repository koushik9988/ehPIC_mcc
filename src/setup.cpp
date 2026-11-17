#include "setup.h"



ParsedArgs parse_cli_args(int argc, char *argv[])
{
    
    help_flags(argc, argv);
    
    if(argc<2)
    {
      std::cout<<"ERROR, at least one argument expected (the input file)."<<std::endl;
      std::exit (EXIT_FAILURE);
    }

    ParsedArgs args;

    args.inputfile = argv[1];
    args.num_threads = std::thread::hardware_concurrency();
    args.restart_from_dump = false;

    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--threads" && i + 1 < argc)
        {
            args.num_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--restart-file" && i + 1 < argc)
        {
            args.restartfile = argv[++i];
            args.restart_from_dump = true;
        }
        else
        {
            std::cout << "Unknown option: " << arg << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    unsigned int hw_threads = std::thread::hardware_concurrency();
    if (args.num_threads == 0 || args.num_threads > static_cast<int>(hw_threads))
    {
        args.num_threads = hw_threads;
    }

    return args;
}


SpeciesConfig load_species_config(const INIMap &iniData, double den)
{
    const auto &SpeciesSection = iniData.at("Species");

    SpeciesConfig cfg;
    cfg.count = INIParser::getInt(SpeciesSection, "count");

    cfg.names.resize(cfg.count);
    cfg.mass.resize(cfg.count);
    cfg.nParticles.resize(cfg.count);
    cfg.temps.resize(cfg.count);
    cfg.charge_signs.resize(cfg.count);
    cfg.frac_densities.resize(cfg.count);
    cfg.normden.assign(cfg.count, 0.0);
    cfg.vs.resize(cfg.count);
    cfg.is_fluid.assign(cfg.count, false);
    cfg.pos_init.resize(cfg.count);
    cfg.vel_init.resize(cfg.count);

    for (int i = 0; i < cfg.count; ++i)
    {
        std::string prefix = "species_" + std::to_string(i) + ".";
        cfg.names[i] = INIParser::getString(SpeciesSection, prefix + "name");
        cfg.mass[i] = INIParser::getDouble(SpeciesSection, prefix + "mass");
        cfg.temps[i] = INIParser::getDouble(SpeciesSection, prefix + "temp");
        cfg.charge_signs[i] = INIParser::getInt(SpeciesSection, prefix + "charge_sign");
        cfg.nParticles[i] = INIParser::getInt(SpeciesSection, prefix + "num");
        cfg.frac_densities[i] = INIParser::getDouble(SpeciesSection, prefix + "normden");
        cfg.vs[i] = INIParser::getDouble(SpeciesSection, prefix + "vs");
        try
        {
            cfg.is_fluid[i] = static_cast<bool>(INIParser::getInt(SpeciesSection, prefix + "isfluid"));
        }
        catch (...)
        {
            cfg.is_fluid[i] = false;
        }
        cfg.pos_init[i] = INIParser::getString(SpeciesSection, prefix + "loadtype_pos");
        cfg.vel_init[i] = INIParser::getString(SpeciesSection, prefix + "loadtype_vel");
    }

    double k = 0;
    for (int i = 0; i < cfg.count; ++i)
    {
        k += (-cfg.charge_signs[i]) * cfg.frac_densities[i];
    }

    display::print("kkk",k);

    if (cfg.count > 1)
    {
        cfg.normden[1] = den;
    }
    if (cfg.count > 0 && k != 0)
    {
        cfg.normden[0] = den / k;
    }
    for (int i = 2; i < cfg.count; ++i)
    {
        cfg.normden[i] = cfg.frac_densities[i] * cfg.normden[0];
    }

    return cfg;
}

std::vector<Species> create_species(const SpeciesConfig &cfg, Domain &domain, bool restart_from_dump, const std::string &restart_file)
{
    std::vector<Species> species_list;
    species_list.reserve(cfg.count);

    for (int i = 0; i < cfg.count; ++i)
    {
        double computed_spwt = cfg.normden[i] * domain.xL * domain.L / cfg.nParticles[i];

        std::string pos_init = cfg.pos_init[i];
        std::string vel_init = cfg.vel_init[i];

        if (restart_from_dump)
        {
            pos_init = "dump";
            vel_init = "dump";
        }

        species_list.emplace_back(cfg.names[i],cfg.mass[i],cfg.charge_signs[i] * Const::QE,computed_spwt,
            cfg.temps[i],cfg.nParticles[i],cfg.vs[i],cfg.frac_densities[i],cfg.normden[i],cfg.is_fluid[i],pos_init,vel_init,domain);
    }

    return species_list;
}

std::unique_ptr<CollisionHandler> create_collision_handler(const INIMap &iniData, Domain &domain)
{
    const auto &collisionSec = iniData.at("collision");
    std::string gas = INIParser::getString(collisionSec, "GAS_TYPE");

    std::unique_ptr<CollisionHandler> handler;
    if (gas == "H")
    {
        handler = std::make_unique<CollisionHandler>(domain, HYDROGEN, "../cross_section");
    }
    else if (gas == "AR")
    {
        handler = std::make_unique<CollisionHandler>(domain, ARGON, "../cross_section");
    }
    else
    {
        throw std::invalid_argument("Unsupported GAS_TYPE in collision section");
    }

    handler->set_electron_cross_sections();
    handler->set_pion_cross_sections();
    handler->calc_total_cross_sections();

    domain.max_electron_coll_freq = handler->max_electron_coll_freq();

    return handler;
}

std::vector<Emitter> create_emitters(const INIMap &iniData, Domain &domain)
{
    std::vector<Emitter> emitters;
    if (!iniData.count("Emitters"))
    {
        return emitters;
    }
 
    const auto &emitterSection = iniData.at("Emitters");
    int emitter_count = INIParser::getInt(emitterSection, "count");
    emitters.reserve(emitter_count);
 
    for (int i = 0; i < emitter_count; i++)
    {
        std::string prefix = "emitter_" + std::to_string(i) + ".";
 
        EmitterParams params;
 
        // required
        params.emitter_loc  = INIParser::getDouble(emitterSection, prefix + "emitter_loc");
        params.density      = INIParser::getDouble(emitterSection, prefix + "density");
        params.emitter_area = INIParser::getDouble(emitterSection, prefix + "emitter_area");
        params.species_id1  = INIParser::getInt(emitterSection, prefix + "species_id1");
 
        // optional keys: missing key -> default
        try 
        { 
            params.species_id2 = INIParser::getInt(emitterSection, prefix + "species_id2"); 
        }
        catch (...)
        { 
            params.species_id2 = -1; 
        }
 
        try
        { 
            params.temp = INIParser::getDouble(emitterSection, prefix + "temp"); 
        }
        catch (...)
        { 
            params.temp = 0; 
        }
 
        try
        { 
            params.numparticle = INIParser::getInt(emitterSection, prefix + "numparticle"); 
        } //deprecat
        catch (...)
        { 
            params.numparticle = 0; 
        }
 
        try
        { 
            params.direction = INIParser::getInt(emitterSection, prefix + "direction");
        }
        catch (...) 
        { 
            params.direction = 0; 
        }
 
        try
        { 
            params.flux_weighted = INIParser::getInt(emitterSection, prefix + "flux_weighted"); 
        }
        catch (...)
        { 
            params.flux_weighted = 1; 
        }
 
        try 
        { 
            params.temp1 = INIParser::getDouble(emitterSection, prefix + "temp1"); 
        }
        catch (...)
        { 
            params.temp1 = -1; 
        }
 
        try
        { 
            params.temp2 = INIParser::getDouble(emitterSection, prefix + "temp2"); 
        }
        catch (...)
        { 
            params.temp2 = -1; 
        }

        try
        { 
            params.spwt = INIParser::getDouble(emitterSection, prefix + "spwt");
            params.has_spwt = (params.spwt > 0); 
        }
        catch (...)
        { 
            params.spwt = -1; params.has_spwt = false; 
        }
 
        // drift_velocity present -> current-density injection; absent -> bulk reservoir injection
        try
        {
            params.vdrift    = INIParser::getDouble(emitterSection, prefix + "drift_velocity");
            params.has_drift = true;
        }
        catch (...)
        {
            params.vdrift    = 0;
            params.has_drift = false;
        }
        emitters.emplace_back(params, domain);
        //domain.current_density = params.density * params.vdrift * Const::QE * domain.vel_norm;
    }
 
    return emitters;
}
