//
// Created by frivas on 19/02/17.
//

#include <SampleGeneratorLib/Utils/Logger.h>
#include <gui/Utils.h>
#include "SamplerGenerationHandler.h"

GenericDatasetReaderPtr SampleGeneratorHandler::SamplerGenerationHandler::createReaderPtr(const QListView *datasetList,
                                                                                          const QListView *namesList,
                                                                                          const QListView *readerImpList,
                                                                                          const QListView *filterClasses,
                                                                                          const std::string& datasetPath, const std::string& namesPath) {
    std::vector<std::string> datasetsToShow;

    if (! Utils::getListViewContent(datasetList,datasetsToShow,datasetPath + "/")){
        Logger::getInstance()->error("Select at least one dataset to read");
        return GenericDatasetReaderPtr();
    }

    std::vector<std::string> names;
    if (! Utils::getListViewContent(namesList,names,namesPath+"/")){
        Logger::getInstance()->error("Select the dataset names related to the input dataset");
        return GenericDatasetReaderPtr();
    }

    std::vector<std::string> readerImplementation;
    if (! Utils::getListViewContent(readerImpList,readerImplementation,"")){
        Logger::getInstance()->error("Select the reader implementation");
        return GenericDatasetReaderPtr();
    }

    std::vector<std::string> classesToFilter;
    if (filterClasses)
        Utils::getListViewContent(filterClasses,classesToFilter,"");


    GenericDatasetReaderPtr reader;
    if (datasetsToShow.size()>1) {
        reader = GenericDatasetReaderPtr(
                new GenericDatasetReader(datasetsToShow,names[0], readerImplementation[0]));
    }
    else {
        reader = GenericDatasetReaderPtr(
                new GenericDatasetReader(datasetsToShow[0],names[0], readerImplementation[0]));
    }


    if (classesToFilter.size()){
        reader->getReader()->filterSamplesByID(classesToFilter);
    }

    return reader;
}
