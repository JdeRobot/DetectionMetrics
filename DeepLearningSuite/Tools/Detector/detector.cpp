//
// Created by frivas on 1/02/17.
//


#include <iostream>
#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/OwnDatasetReader.h>
#include <FrameworkEvaluator/MassInferencer.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <FrameworkEvaluator/GenericInferencer.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("inferencerImplementation");
        this->requiredArguments.push_back("inferencerConfig");
        this->requiredArguments.push_back("inferencerWeights");
        this->requiredArguments.push_back("inferencerNames");
        this->requiredArguments.push_back("readerNames");
    };
    void operator()(){
        Key inputPath=this->config->getKey("inputPath");
        Key outputPath=this->config->getKey("outputPath");

        Key readerImplementationKey = this->config->getKey("readerImplementation");
        Key infererImplementationKey = this->config->getKey("inferencerImplementation");
        Key inferencerConfigKey = this->config->getKey("inferencerConfig");
        Key inferencerWeightsKey = this->config->getKey("inferencerWeights");
        Key inferencerNamesKey = this->config->getKey("inferencerNames");
        Key readerNamesKey = this->config->getKey("readerNames");

        GenericDatasetReaderPtr reader;
        if (inputPath.isVector()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPath.getValues(),readerNamesKey.getValue(), readerImplementationKey.getValue()));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPath.getValue(),readerNamesKey.getValue(), readerImplementationKey.getValue()));
        }


        GenericInferencerPtr inferencer(new GenericInferencer(inferencerConfigKey.getValue(),inferencerWeightsKey.getValue(),inferencerNamesKey.getValue(),infererImplementationKey.getValue()));
        MassInferencer massInferencer(reader->getReader(),inferencer->getInferencer(),outputPath.getValue(), true);
        massInferencer.process(false);

    };
};



int main (int argc, char* argv[]) {

    MyApp myApp(argc,argv);
    myApp.process();
}


