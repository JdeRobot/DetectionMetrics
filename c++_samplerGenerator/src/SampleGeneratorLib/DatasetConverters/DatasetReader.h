//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_DATASETREADER_H
#define SAMPLERGENERATOR_DATASETREADER_H

#include <string>
#include <Sample.h>
#include <boost/shared_ptr.hpp>


class DatasetReader {
public:
    DatasetReader();
    bool getNetxSample(Sample& sample);
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    int getNumberOfElements();
    void resetReaderCounter();
    bool getSampleBySampleID(Sample** sample, const std::string& sampleID);
    void printDatasetStats();
    virtual bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
    void addSample(Sample sample);

protected:
    std::vector<Sample> samples;
    //std::string datasetPath;
    int readerCounter;
};


typedef boost::shared_ptr<DatasetReader> DatasetReaderPtr;

#endif //SAMPLERGENERATOR_DATASETREADER_H
