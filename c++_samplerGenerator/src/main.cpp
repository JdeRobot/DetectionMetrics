

#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>

#include <opencv2/highgui/highgui.hpp>
#include "SampleGeneratorLib/RecorderConverter.h"
#include "SampleGeneratorLib/DepthForegroundSegmentator.h"
#include "SampleGeneratorLib/BoundingValidator.h"
#include "SampleGeneratorLib/DetectionsValidator.h"
#include <SampleGeneratorLib/Logger.h>

namespace
{
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t SUCCESS = 0;
    const size_t ERROR_UNHANDLED_EXCEPTION = 2;

} // namespace


int parse_arguments(const int argc, char* argv[], std::string& path){
    try
    {
        /** Define and parse the program options
         */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
                ("help", "Print help messages")
                ("word,w", po::value<std::string>(&path)->required());
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


int main (int argc, char* argv[])
{

    Logger::getInstance()->setLevel(Logger::INFO);

    std::string path; // /mnt/large/pentalo/sample/data/images/
    parse_arguments(argc,argv,path);


    std::string depthImagesPath= path + "/" + "/camera2/";
    std::string colorImagesPath= path + "/" + "/camera1/";
    std::string outputPath= "./out";


    RecorderConverter converter(colorImagesPath,depthImagesPath);
    DepthForegroundSegmentator segmentator;

    std::string colorImagePath;
    std::string depthImagePath;

    DetectionsValidator validator(outputPath);
    cv::Mat previousImage;
    int counter=0;
    int maxElements= converter.getNumSamples();
    while (converter.getNext(colorImagePath,depthImagePath)){
        std::stringstream ss ;
        ss << counter << "/" << maxElements;
        Logger::getInstance()->info( "Processing [" +  ss.str() + "] : " + colorImagePath + ", " + depthImagePath );
        cv::Mat colorImage= cv::imread(colorImagePath);
        if (!previousImage.empty()){
            cv::Mat diff;
            cv::absdiff(colorImage,previousImage,diff);
            auto val = cv::sum(cv::sum(diff));
            if (val[0] < 1000){
                continue;
            }
        }
        colorImage.copyTo(previousImage);
        cv::Mat depthImage = cv::imread(depthImagePath);
        std::vector<std::vector<cv::Point>> detections = segmentator.process(depthImage);

        validator.validate(colorImage,depthImage,detections);
        counter++;
    }

    return 0;
}