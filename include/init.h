#ifndef _INIT_H_
#define _INIT_H_

#include <string>
#include <H5Cpp.h>
#include <cmath>
#include <tuple>
#include <vector>

class Domain;
class Species;

//using namespace Const;

class Init 
{
    public:
    //in the constructor we pass Species and Domain class instances to access the species and domain class
    //memebr variable and methods 
    Init(Species &species, Domain &domain,const std::string& restart_file = "");   
    static double SampleVel(Species &species, double temp);

    private:
    Species &species;
    Domain &domain;
    std::string restart_file;
    double SampleVel(Species &species);
};

#endif 
