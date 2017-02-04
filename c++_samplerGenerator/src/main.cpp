

#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>

#include <opencv2/highgui/highgui.hpp>
#include "SampleGeneratorLib/GenerationUtils/RecorderConverter.h"
#include "SampleGeneratorLib/GenerationUtils/DepthForegroundSegmentator.h"
#include "SampleGeneratorLib/GenerationUtils/BoundingValidator.h"
#include "SampleGeneratorLib/GenerationUtils/DetectionsValidator.h"
#include <SampleGeneratorLib/Utils/Logger.h>
#include <SampleGeneratorLib/Utils/SampleGenerationApp.h>

class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("colorImagesPath");
        this->requiredArguments.push_back("depthImagesPath");

    };
    void operator()(){
        Key outputPath=this->config.getKey("outputPath");
        Key colorImagesPathKey=this->config.getKey("colorImagesPath");
        Key depthImagesPathKey=this->config.getKey("depthImagesPath");



        RecorderConverter converter(colorImagesPathKey.getValue(),depthImagesPathKey.getValue());
        DepthForegroundSegmentator segmentator;

        std::string colorImagePath;
        std::string depthImagePath;

        DetectionsValidator validator(outputPath.getValue());
        cv::Mat previousImage;
        int counter=0;
        int maxElements= converter.getNumSamples();
        while (converter.getNext(colorImagePath,depthImagePath)){
            std::stringstream ss ;
            ss << counter << "/" << maxElements;
            Logger::getInstance()->info( "Processing [" +  ss.str() + "] : " + colorImagePath + ", " + depthImagePath );
            cv::Mat colorImage= cv::imread(colorImagePath);
            cv::cvtColor(colorImage,colorImage,CV_RGB2BGR);
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

    };
};



int main (int argc, char* argv[])
{

    MyApp myApp(argc,argv);
    myApp.process();
}