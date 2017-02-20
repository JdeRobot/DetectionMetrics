//
// Created by frivas on 21/01/17.
//


#include <iostream>
#include <string>
#include <SampleGeneratorLib/Utils/SampleGenerationApp.h>
#include <SampleGeneratorLib/DatasetConverters/GenericDatasetReader.h>
#include <SampleGeneratorLib/FrameworkEvaluator/DetectionsEvaluator.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("readerNames");


    };
    void operator()(){
        Key inputPathKey=this->config->getKey("inputPath");
        Key readerImplementationKey = this->config->getKey("readerImplementation");
        Key readerNamesKey = this->config->getKey("readerNames");


        GenericDatasetReaderPtr reader;
        if (inputPathKey.isVector()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathKey.getValues(),readerNamesKey.getValue(), readerImplementationKey.getValue()));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathKey.getValue(),readerNamesKey.getValue(), readerImplementationKey.getValue()));
        }



        Sample sample;
        while (reader->getReader()->getNetxSample(sample)){
            std::cout << "number of elements: " << sample.getRectRegions()->getRegions().size() << std::endl;
            cv::Mat image =sample.getSampledColorImage();
            cv::imshow("Viewer", image);
            cv::waitKey(0);
        }

    };
};

int main (int argc, char* argv[])
{

    MyApp myApp(argc,argv);
    myApp.process();
}