//
// Created by frivas on 5/02/17.
//

#ifndef SAMPLERGENERATOR_DATASETWRITTER_H
#define SAMPLERGENERATOR_DATASETWRITTER_H


#include "DatasetReader.h"

class DatasetWriter {
public:
    DatasetWriter(const std::string& outPath, DatasetReaderPtr& reader);
    virtual void process(bool usedColorImage=true)=0;

protected:
    std::string outPath;
    DatasetReaderPtr& reader;
};


typedef  boost::shared_ptr<DatasetWriter> DatasetWriterPtr;

#endif //SAMPLERGENERATOR_DATASETWRITTER_H
