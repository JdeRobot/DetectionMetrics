//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_DATASETREADER_H
#define SAMPLERGENERATOR_DATASETREADER_H

#include <string>
#include <Sample.h>
enum DatasetType{ DST_OWN, DST_YOLO};

class DatasetReader {
public:
    DatasetReader();
    bool getNetxSample(Sample& sample);
    void filterSamplesByID(std::vector<int> filteredIDS);
    int getNumberOfElements();
    void resetReaderCounter();

protected:
    std::vector<Sample> samples;
    std::string datasetPath;
    int readerCounter;
};


#endif //SAMPLERGENERATOR_DATASETREADER_H
