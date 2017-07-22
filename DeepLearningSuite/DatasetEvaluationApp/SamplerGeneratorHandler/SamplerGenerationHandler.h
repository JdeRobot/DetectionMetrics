//
// Created by frivas on 19/02/17.
//

#ifndef SAMPLERGENERATOR_SAMPLERGENERATIONHANDLER_H
#define SAMPLERGENERATOR_SAMPLERGENERATIONHANDLER_H

#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <QtWidgets/QListView>
#include <DatasetConverters/liveReaders/GenericLiveReader.h>

namespace SampleGeneratorHandler {

    class SamplerGenerationHandler {
    public:
        static GenericDatasetReaderPtr createDatasetReaderPtr(const QListView *datasetList, const QListView *namesList,
                                                              const QListView *readerImpList,
                                                              const QListView *filterClasses,
                                                              const std::string &datasetPath,
                                                              const std::string &namesPath);

        static GenericLiveReaderPtr createLiveReaderPtr(const QListView *namesList,
                                                        const QListView *readerImpList,
                                                        const std::string &infoPath,
                                                        const std::string &namesPath);
    };

}

#endif //SAMPLERGENERATOR_SAMPLERGENERATIONHANDLER_H
