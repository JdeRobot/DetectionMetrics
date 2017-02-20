

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
#include <SampleGeneratorLib/FrameworkEvaluator/DarknetInferencer.h>

class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("colorImagesPath");
        this->requiredArguments.push_back("depthImagesPath");
        this->requiredArguments.push_back("detector");


    };
    void operator()(){
        Key outputPath=this->config->getKey("outputPath");
        Key colorImagesPathKey=this->config->getKey("colorImagesPath");
        Key depthImagesPathKey=this->config->getKey("depthImagesPath");
        Key detectorKey=this->config->getKey("detector");


        if (detectorKey.getValue().compare("pentalo-bg")==0) {

            RecorderConverter converter(colorImagesPathKey.getValue(), depthImagesPathKey.getValue());
            DepthForegroundSegmentator segmentator;


            DetectionsValidator validator(outputPath.getValue());
            cv::Mat previousImage;
            int counter = 0;
            int maxElements = converter.getNumSamples();
            Sample sample;
            while (converter.getNext(sample)) {
                counter++;
                std::stringstream ss;
                ss << counter << "/" << maxElements;
                Logger::getInstance()->info( "Processing [" + ss.str() + "]");
                cv::Mat colorImage = sample.getColorImage().clone();
                cv::cvtColor(colorImage, colorImage, CV_RGB2BGR);
                if (!previousImage.empty()) {
                    cv::Mat diff;
                    cv::absdiff(colorImage, previousImage, diff);
                    auto val = cv::sum(cv::sum(diff));
                    if (val[0] < 1000) {
                        continue;
                    }
                }
                colorImage.copyTo(previousImage);
                cv::Mat depthImage = sample.getDepthImage().clone();
                std::vector<std::vector<cv::Point>> detections = segmentator.process(depthImage);

                validator.validate(colorImage, depthImage, detections);
            }
        }
        else{
            Key inferencerImplementationKey=this->config->getKey("inferencerImplementation");
            Key inferencerNamesKey=this->config->getKey("inferencerNames");
            Key inferencerConfigKey=this->config->getKey("inferencerConfig");
            Key inferencerWeightsKey=this->config->getKey("inferencerWeights");

            RecorderConverter converter(colorImagesPathKey.getValue(), depthImagesPathKey.getValue());

            FrameworkInferencerPtr inferencer;

            if (inferencerImplementationKey.getValue().compare("yolo")==0) {
                inferencer = DarknetInferencerPtr( new DarknetInferencer(inferencerConfigKey.getValue(), inferencerWeightsKey.getValue(), inferencerNamesKey.getValue()));
            }
            else{
                Logger::getInstance()->error(inferencerImplementationKey.getValue() + " is not a valid inferencer implementation");
            }

            DetectionsValidator validator(outputPath.getValue());
            int maxElements = converter.getNumSamples();
            Sample sample;
            int counter=0;
            while (converter.getNext(sample)) {
                counter++;
                std::stringstream ss;
                ss << counter << "/" << maxElements;
                Logger::getInstance()->info("Processing [" + ss.str() + "]");

                Sample detectedSample = inferencer->detect(sample.getColorImage());

                validator.validate(detectedSample);


            }
        }

    };
};



int main (int argc, char* argv[])
{

    MyApp myApp(argc,argv);
    myApp.process();
}