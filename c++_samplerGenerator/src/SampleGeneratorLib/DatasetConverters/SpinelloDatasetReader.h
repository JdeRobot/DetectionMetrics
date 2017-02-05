//
// Created by frivas on 29/01/17.
//

#ifndef SAMPLERGENERATOR_SPINELLODATASETREADER_H
#define SAMPLERGENERATOR_SPINELLODATASETREADER_H


#include "DatasetReader.h"

class SpinelloDatasetReader: public DatasetReader {
public:
    SpinelloDatasetReader(const std::string& path);
    SpinelloDatasetReader();
    bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");

private:
};


typedef boost::shared_ptr<SpinelloDatasetReader> SpinelloDatasetReaderPtr;


#endif //SAMPLERGENERATOR_SPINELLODATASETREADER_H
