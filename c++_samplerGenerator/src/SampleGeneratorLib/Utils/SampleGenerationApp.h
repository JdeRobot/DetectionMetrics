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
    ConfigurationPtr getConfig();


protected:
    ConfigurationPtr config;
    std::string configFilePath;
    std::vector<std::string> requiredArguments;

    bool verifyRequirements();
    int parse_arguments(const int argc, char* argv[], std::string& configFile);
    int argc;
    char** argv;


};


#endif //SAMPLERGENERATOR_SAMPLEGENERATIOAPP_H
