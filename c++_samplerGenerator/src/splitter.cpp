//
// Created by frivas on 5/02/17.
//

#include <iostream>
#include <string>
#include <SampleGeneratorLib/Utils/SampleGenerationApp.h>
#include <SampleGeneratorLib/DatasetConverters/GenericDatasetReader.h>
#include <SampleGeneratorLib/FrameworkEvaluator/DetectionsEvaluator.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <SampleGeneratorLib/Utils/Logger.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("readerImplementation");


    };
    void operator()(){
        Key inputPathKey=this->config.getKey("inputPath");
        Key outputPathKey=this->config.getKey("outputPath");

        Key readerImplementationKey = this->config.getKey("readerImplementation");


        std::string trainPath=outputPathKey.getValue() + "/train";
        std::string testPath=outputPathKey.getValue() + "/test";


        auto boostPath= boost::filesystem::path(outputPathKey.getValue());
        if (!boost::filesystem::exists(boostPath)){
            boost::filesystem::create_directories(boostPath);
            auto boostPathTest= boost::filesystem::path(testPath);
            boost::filesystem::create_directories(boostPathTest);
            auto boostPathTrain= boost::filesystem::path(trainPath);
            boost::filesystem::create_directories(boostPathTrain);
        }
        else {
            Logger::getInstance()->error("Output directory already exists");
            Logger::getInstance()->error("Continuing detecting");
            exit(1);

        }

        GenericDatasetReaderPtr reader;
        if (inputPathKey.isVector()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathKey.getValues(), readerImplementationKey.getValue()));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathKey.getValue(), readerImplementationKey.getValue()));
        }


        int ratio=7;

        Sample sample;
        int counter=0;
        while (reader->getReader()->getNetxSample(sample)){
            if (counter <7){
                sample.save(trainPath);
            }
            else{
                sample.save(testPath);
            }
            counter= counter % 10;
        }

    };
};

int main (int argc, char* argv[]) {

    MyApp myApp(argc, argv);
    myApp.process();
}