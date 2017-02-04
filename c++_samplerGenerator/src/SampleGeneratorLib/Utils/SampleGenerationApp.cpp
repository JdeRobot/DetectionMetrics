//
// Created by frivas on 4/02/17.
//

#include "SampleGenerationApp.h"
#include "Logger.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

namespace
{
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t SUCCESS = 0;
    const size_t ERROR_UNHANDLED_EXCEPTION = 2;

} // namespace



SampleGenerationApp::SampleGenerationApp(int argc, char **argv) {

    if (parse_arguments(argc,argv,configFilePath) != SUCCESS){
        exit(1);
    }

    config=Configuration(configFilePath);
}



int SampleGenerationApp::parse_arguments(const int argc, char* argv[], std::string& configFile){
    try
    {
        /** Define and parse the program options
         */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
                ("help", "Print help messages")
                ("configFile,c", po::value<std::string>(&configFile)->required());
        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc),
                      vm); // can throw

            /** --help option
             */
            if ( vm.count("help")  )
            {
                std::cout << "Basic Command Line Parameter App" << std::endl
                          << desc << std::endl;
                return SUCCESS;
            }

            po::notify(vm); // throws on error, so do after help in case
            // there are any problems
        }
        catch(po::error& e)
        {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return ERROR_IN_COMMAND_LINE;
        }

        // application code here //

    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
                  << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;

    }
    return SUCCESS;

}

void SampleGenerationApp::process() {
    if (verifyRequirements())
        (*this)();
}

bool SampleGenerationApp::verifyRequirements() {
    bool success=true;
    for (auto it = this->requiredArguments.begin(), end =this->requiredArguments.end(); it != end; ++it){
        if (!this->config.keyExists(*it)){
            Logger::getInstance()->error("Key: " + (*it) + " is not defined in the configuration file");
            success=false;
        }

    }
    return success;
}
