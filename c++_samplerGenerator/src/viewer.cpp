//
// Created by frivas on 21/01/17.
//


#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <SampleGeneratorLib/Logger.h>
#include <highgui.h>
#include <SampleGeneratorLib/RectRegions.h>
#include <SampleGeneratorLib/ContourRegions.h>
#include <SampleGeneratorLib/Sample.h>



namespace
{
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t SUCCESS = 0;
    const size_t ERROR_UNHANDLED_EXCEPTION = 2;

} // namespace





struct ViewerAguments{
    std::string path;
    bool depthEnabled;

    ViewerAguments():depthEnabled(false){};
};


int parse_arguments(const int argc, char* argv[], ViewerAguments& args){
    try
    {
        /** Define and parse the program options
         */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
                ("help", "Print help messages")
                ("word,w", po::value<std::string>(&args.path)->required())
                ("depth,d", po::value<bool>(&args.depthEnabled)->default_value(false));

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
}





int main (int argc, char* argv[]) {

    ViewerAguments args;
    parse_arguments(argc,argv,args);


    Logger::getInstance()->setLevel(Logger::INFO);
    Logger::getInstance()->info("Reviewing " + args.path);

    boost::filesystem::directory_iterator end_itr;
    boost::filesystem::path boostPath(args.path);


    std::vector<std::string> filesID;

    for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
    {
        if ((is_regular_file(itr->status()) && itr->path().extension()==".png") && (itr->path().string().find("-depth") == std::string::npos)) {
            filesID.push_back(itr->path().filename().stem().string());
        }

    }

    std::sort(filesID.begin(),filesID.end());

    for (auto it = filesID.begin(), end=filesID.end(); it != end; ++it){
        Sample sample(args.path,*it,args.depthEnabled);
        cv::imshow("color", sample.getSampledColorImage());
        if (args.depthEnabled) {
            cv::imshow("depth", sample.getSampledDepthImage());
        }

        cv::waitKey(0);
    }

}
