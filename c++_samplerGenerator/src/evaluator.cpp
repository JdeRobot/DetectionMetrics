//
// Created by frivas on 1/02/17.
//


#include <iostream>
#include <string>
#include <SampleGeneratorLib/Utils/SampleGenerationApp.h>
#include <SampleGeneratorLib/DatasetConverters/GenericDatasetReader.h>
#include <SampleGeneratorLib/FrameworkEvaluator/DetectionsEvaluator.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("inputPathGT");
        this->requiredArguments.push_back("inputPathDetection");
        this->requiredArguments.push_back("readerImplementationGT");
        this->requiredArguments.push_back("readerImplementationDetection");

    };
    void operator()(){
        Key outputPath=this->config.getKey("outputPath");
        Key inputPathGT=this->config.getKey("inputPathGT");
        Key inputPathDetection=this->config.getKey("inputPathDetection");
        Key readerImplementationGTKey=this->config.getKey("readerImplementationGT");
        Key readerImplementationDetectionKey=this->config.getKey("readerImplementationDetection");



        GenericDatasetReaderPtr readerGT(new GenericDatasetReader(inputPathGT.getValue(), readerImplementationGTKey.getValue()));
        GenericDatasetReaderPtr readerDetection(new GenericDatasetReader(inputPathDetection.getValue(), readerImplementationDetectionKey.getValue()));


        DetectionsEvaluatorPtr evaluator(new DetectionsEvaluator(readerGT->getReader(),readerDetection->getReader(),true));
        evaluator->evaluate();


    };
};





int main (int argc, char* argv[]) {
    MyApp myApp(argc,argv);
    myApp.process();
}
