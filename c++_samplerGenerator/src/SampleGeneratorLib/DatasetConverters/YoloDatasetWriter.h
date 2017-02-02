//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_YOLODATASETCONVERTER_H
#define SAMPLERGENERATOR_YOLODATASETCONVERTER_H

#include <string>
#include "DatasetReader.h"

class YoloDatasetWriter {
public:
    YoloDatasetWriter(const std::string& outPath, DatasetReader& reader);
    void process(bool overWriteclassWithZero);

private:
    std::string outPath;
    DatasetReader& reader;
    std::string fullImagesPath;
    std::string fullLabelsPath;
};


#endif //SAMPLERGENERATOR_YOLODATASETCONVERTER_H
