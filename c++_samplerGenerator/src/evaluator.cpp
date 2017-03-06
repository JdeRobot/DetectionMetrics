//
// Created by frivas on 1/02/17.
//


#include <iostream>
#include <string>
#include <SampleGeneratorLib/Utils/SampleGenerationApp.h>
#include <SampleGeneratorLib/DatasetConverters/readers/GenericDatasetReader.h>
#include <SampleGeneratorLib/FrameworkEvaluator/DetectionsEvaluator.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("inputPathGT");
        this->requiredArguments.push_back("inputPathDetection");
        this->requiredArguments.push_back("readerImplementationGT");
        this->requiredArguments.push_back("readerImplementationDetection");
        this->requiredArguments.push_back("readerNames");


    };
    void operator()(){
        Key outputPath=this->config->getKey("outputPath");
        Key inputPathGT=this->config->getKey("inputPathGT");
        Key inputPathDetection=this->config->getKey("inputPathDetection");
        Key readerImplementationGTKey=this->config->getKey("readerImplementationGT");
        Key readerImplementationDetectionKey=this->config->getKey("readerImplementationDetection");
        Key readerNamesKey=this->config->getKey("readerNames");



        GenericDatasetReaderPtr readerGT(new GenericDatasetReader(inputPathGT.getValue(),readerNamesKey.getValue(), readerImplementationGTKey.getValue()));
        GenericDatasetReaderPtr readerDetection(new GenericDatasetReader(inputPathDetection.getValue(),readerNamesKey.getValue(), readerImplementationDetectionKey.getValue()));


        DetectionsEvaluatorPtr evaluator(new DetectionsEvaluator(readerGT->getReader(),readerDetection->getReader(),true));
        //todo ñapa
        evaluator->addValidMixClass("person", "person-falling");
        evaluator->addValidMixClass("person", "person-fall");
        evaluator->addClassToDisplay("person");
        evaluator->addClassToDisplay("person-falling");
        evaluator->addClassToDisplay("person-fall");
        evaluator->evaluate();


    };
};





int main (int argc, char* argv[]) {
    MyApp myApp(argc,argv);
    myApp.process();
}
