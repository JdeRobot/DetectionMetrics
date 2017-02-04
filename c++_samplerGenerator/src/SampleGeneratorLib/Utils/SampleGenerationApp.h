//
// Created by frivas on 4/02/17.
//

#ifndef SAMPLERGENERATOR_SAMPLEGENERATIOAPP_H
#define SAMPLERGENERATOR_SAMPLEGENERATIOAPP_H


#include "Configuration.h"

class SampleGenerationApp {
public:
    SampleGenerationApp(int argc, char* argv[]);
    virtual void operator()() =0;
    void process();


protected:
    Configuration config;
    std::string configFilePath;
    std::vector<std::string> requiredArguments;

    bool verifyRequirements();
    int parse_arguments(const int argc, char* argv[], std::string& configFile);



};


#endif //SAMPLERGENERATOR_SAMPLEGENERATIOAPP_H
