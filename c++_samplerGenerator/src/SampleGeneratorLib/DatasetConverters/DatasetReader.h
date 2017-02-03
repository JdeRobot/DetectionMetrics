//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_DATASETREADER_H
#define SAMPLERGENERATOR_DATASETREADER_H

#include <string>
#include <Sample.h>
#include <boost/shared_ptr.hpp>

enum DatasetType{ DST_OWN, DST_YOLO};

class DatasetReader {
public:
    DatasetReader();
    bool getNetxSample(Sample& sample);
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    int getNumberOfElements();
    void resetReaderCounter();
    bool getSampleBySampleID(Sample** sample, const std::string& sampleID);

protected:
    std::vector<Sample> samples;
    std::string datasetPath;
    int readerCounter;
};


typedef boost::shared_ptr<DatasetReader> DatasetReaderPtr;

#endif //SAMPLERGENERATOR_DATASETREADER_H
