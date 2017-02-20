//
// Created by frivas on 19/02/17.
//

#ifndef SAMPLERGENERATOR_SAMPLERGENERATIONHANDLER_H
#define SAMPLERGENERATOR_SAMPLERGENERATIONHANDLER_H

#include <SampleGeneratorLib/DatasetConverters/GenericDatasetReader.h>
#include <QtWidgets/QListView>

namespace SampleGeneratorHandler {

    class SamplerGenerationHandler {
    public:
        static GenericDatasetReaderPtr createReaderPtr(const QListView* datasetList,const QListView* namesList,const QListView* readerImpList, const QListView* filterClasses,  const std::string& datasetPath, const std::string& namesPath);
    };

}

#endif //SAMPLERGENERATOR_SAMPLERGENERATIONHANDLER_H
