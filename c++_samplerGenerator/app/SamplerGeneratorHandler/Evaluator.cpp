//
// Created by frivas on 20/02/17.
//

#include <SampleGeneratorLib/FrameworkEvaluator/DetectionsEvaluator.h>
#include "Evaluator.h"
#include "SamplerGenerationHandler.h"

void
SampleGeneratorHandler::Evaluator::process(QListView *datasetListGT, QListView *namesListGT, QListView *readerImpListGT,
                                           QListView *datasetListDetect, QListView *namesListDetect,
                                           QListView *readerImpListDetect, QListView *filterClasses,
                                           const std::string &datasetPath, const std::string &namesGTPath,
                                           const std::string &inferencesPath, const std::string &inferencesNamesPath) {

    GenericDatasetReaderPtr readerGT = SamplerGenerationHandler::createReaderPtr(datasetListGT,namesListGT,readerImpListGT,filterClasses,datasetPath,datasetPath);
    GenericDatasetReaderPtr readerDetection = SamplerGenerationHandler::createReaderPtr(datasetListDetect,namesListDetect,readerImpListDetect,filterClasses,datasetPath,inferencesNamesPath);



    DetectionsEvaluatorPtr evaluator(new DetectionsEvaluator(readerGT->getReader(),readerDetection->getReader(),true));

    //todo ñapa
    evaluator->addClassToDisplay("person");
    evaluator->addClassToDisplay("person-falling");
    evaluator->addClassToDisplay("person-fall");
    evaluator->evaluate();


}
