//
// Created by frivas on 20/02/17.
//

#include <SampleGeneratorLib/DatasetConverters/readers/GenericDatasetReader.h>
#include <gui/Utils.h>
#include <glog/logging.h>
#include <SampleGeneratorLib/FrameworkEvaluator/GenericInferencer.h>
#include <SampleGeneratorLib/FrameworkEvaluator/MassInferencer.h>
#include "Detector.h"
#include "SamplerGenerationHandler.h"

void SampleGeneratorHandler::Detector::process(QListView* datasetList,QListView* namesList,QListView* readerImpList, const std::string& datasetPath,
                                               QListView* weightsList, QListView* netConfigList, QListView* inferencerImpList, QListView* inferencerNamesList,
                                               const std::string& weightsPath, const std::string& cfgPath, const std::string& outputPath,
                                               const std::string& inferencerNamesPath, bool useDepth, bool singleEvaluation) {

    GenericDatasetReaderPtr reader = SamplerGenerationHandler::createReaderPtr(datasetList,namesList,readerImpList,NULL,datasetPath,inferencerNamesPath);

    std::vector<std::string> weights;
    if (! Utils::getListViewContent(weightsList,weights,weightsPath+ "/")){
        LOG(WARNING)<<"Select the weightsList";
        return;
    }

    std::vector<std::string> netConfiguration;
    if (! Utils::getListViewContent(netConfigList,netConfiguration,cfgPath+ "/")){
        LOG(WARNING)<<"Select the netConfiguration";
        return;
    }

    std::vector<std::string> inferencerImp;
    if (! Utils::getListViewContent(inferencerImpList,inferencerImp,"")){
        LOG(WARNING)<<"Select the inferencer type";
        return;
    }

    std::vector<std::string> inferencerNames;
    if (! Utils::getListViewContent(inferencerNamesList,inferencerNames,inferencerNamesPath + "/")){
        LOG(WARNING)<<"Select the inferencer type";
        return;
    }

    GenericInferencerPtr inferencer(new GenericInferencer(netConfiguration[0],weights[0],inferencerNames[0],inferencerImp[0]));
    MassInferencer massInferencer(reader->getReader(),inferencer->getInferencer(),outputPath, true);
    massInferencer.process(useDepth);
}
