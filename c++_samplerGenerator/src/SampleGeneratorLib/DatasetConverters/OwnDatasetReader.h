//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_OWNDATASETREADER_H
#define SAMPLERGENERATOR_OWNDATASETREADER_H

#include "DatasetReader.h"

class OwnDatasetReader:public DatasetReader {
public:
    OwnDatasetReader(const std::string& path);
    virtual bool getNetxSample(Sample& sample);
private:
    int currentIndex;
};


typedef boost::shared_ptr<OwnDatasetReader> OwnDatasetReaderPtr;


#endif //SAMPLERGENERATOR_OWNDATASETREADER_H
