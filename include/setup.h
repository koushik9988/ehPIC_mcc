#ifndef _SETUP_H
#define _SETUP_H

#include "collision.h"
#include "domain.h"
#include "emitter.h"
#include "iniparser.h"
#include "species.h"
#include "info.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <map>
#include <memory>
#include <string>
#include <vector>

class Domain;
class Species;
class CollisionHandler;
class Emitter;

struct ParsedArgs
{
    std::string inputfile;
    std::string restartfile;
    int num_threads;
    bool restart_from_dump;
};

struct SpeciesConfig
{
    int count = 0;
    std::vector<std::string> names;
    std::vector<double> mass;
    std::vector<int> nParticles;
    std::vector<double> temps;
    std::vector<int> charge_signs;
    std::vector<double> frac_densities;
    std::vector<double> normden;
    std::vector<double> vs;
    std::vector<bool> is_fluid;
    std::vector<std::string> pos_init;
    std::vector<std::string> vel_init;
};

using INIMap = std::map<std::string, std::map<std::string, std::string>>;

///function with return type struct
ParsedArgs parse_cli_args(int argc, char *argv[]);

SpeciesConfig load_species_config(const INIMap &iniData, double den);

std::vector<Species> create_species(const SpeciesConfig &cfg, Domain &domain, bool restart_from_dump, const std::string &restart_file);

std::unique_ptr<CollisionHandler> create_collision_handler(const INIMap &iniData, Domain &domain);

std::vector<Emitter> create_emitters(const INIMap &iniData, Domain &domain);

#endif
