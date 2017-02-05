//
// Created by frivas on 1/02/17.
//


#include <iostream>
#include <Utils/SampleGenerationApp.h>
#include <SampleGeneratorLib/DatasetConverters/OwnDatasetReader.h>
#include <SampleGeneratorLib/FrameworkEvaluator/MassInferencer.h>
#include <SampleGeneratorLib/DatasetConverters/GenericDatasetReader.h>
#include <SampleGeneratorLib/FrameworkEvaluator/DarknetInferencer.h>
#include <SampleGeneratorLib/FrameworkEvaluator/GenericInferencer.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("inferencerImplementation");
        this->requiredArguments.push_back("inferencerConfig");
        this->requiredArguments.push_back("inferencerWeights");




    };
    void operator()(){
        Key inputPath=this->config.getKey("inputPath");
        Key outputPath=this->config.getKey("outputPath");

        Key readerImplementationKey = this->config.getKey("readerImplementation");
        Key infererImplementationKey = this->config.getKey("inferencerImplementation");
        Key inferencerConfigKey = this->config.getKey("inferencerConfig");
        Key inferencerWeightsKey = this->config.getKey("inferencerWeights");

        GenericDatasetReaderPtr reader;
        if (inputPath.isVector()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPath.getValues(), readerImplementationKey.getValue()));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPath.getValue(), readerImplementationKey.getValue()));
        }


        GenericInferencerPtr inferencer(new GenericInferencer(inferencerConfigKey.getValue(),inferencerWeightsKey.getValue(),infererImplementationKey.getValue()));
        MassInferencer massInferencer(reader->getReader(),inferencer->getInferencer(),outputPath.getValue(), true);
        massInferencer.process();

    };
};



int main (int argc, char* argv[]) {

    MyApp myApp(argc,argv);
    myApp.process();
}


