#ifndef _INFO_H_
#define _INFO_H_

#include <iostream>
#include <cstdlib>
#include <string>
#include "output.h"

//Code Metadata 
#define NAME        "ePIC_mcc"
#define VERSION     "4.0.0"
#define AUTHOR      "Kaushik Kalita"
#define EMAIL       "koushikkalita99@gmail.com"
#define DESCRIPTION "A Simple 1D-3V Electrostatic PIC code with Monte-Carlo collisional Models"
#define USAGE       "./ePIC_mcc <input_file> [options]"

// version
inline void print_version()
{
    display::print(NAME, " v", VERSION);
    display::print("Author:  ", AUTHOR, " <", EMAIL, ">");
}

inline void print_help()
{
    display::print("================================================================");
    display::print("  ", NAME, " v", VERSION, " - ", DESCRIPTION);
    display::print("  Author : ", AUTHOR, " <", EMAIL, ">");
    display::print("================================================================\n");

    display::print("Usage:");
    display::print("  ", USAGE, "\n");

    display::print("Arguments:");
    display::print("  <input_file>                  Path to the input file (required).");
    display::print("                                Contains simulation parameters such");
    display::print("                                as grid size, timestep, species info etc.\n");

    display::print("Options:");
    display::print(" --threads <n>                  Number of threads to use.");
    display::print("                                Defaults to hardware maximum if not");
    display::print("                                specified or if <n> exceeds core count.\n");
    display::print("  --restart-file <dump_file>    Resume simulation from a dump phase datafile.");
    display::print("                                Loads particle positions and velocities,");
    display::print("                                data from a previous run.\n");
    display::print("  -h, --help                    Show this help message.\n");
    display::print("  -v, --version                 Show version info.\n");

    display::print("Examples :");
    display::print("  ", NAME, " ../inputfiles/input.ini");
    display::print("  ", NAME, " ../inputfiles/input.ini --threads 8");
    display::print("  ", NAME, " ../inputfiles/input.ini --restart-file  ../datafiles/data/phase_1000.h5");
    display::print("  ", NAME, " ../inputfiles/input.ini --restart-file  ../datafiles/data/phase_1000.h5 --threads 4");
    display::print("  ", NAME, "   path/to/input.ini     --restart-file    path/to/phase_{time_step_for_restart}.h5 --threads $(nproc)");
    //display::print("  ", NAME, " input.ini --restart\n");

    display::print("================================================================");
}

inline void help_flags(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--help"    || arg == "-h") { print_help();    std::exit(0); }
        if (arg == "--version" || arg == "-v") { print_version(); std::exit(0); }
    }
}
// ─────────────────────────────────────────────────────────────────────────────


#endif