//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_YOLODATASETCONVERTER_H
#define SAMPLERGENERATOR_YOLODATASETCONVERTER_H

#include <string>
#include "DatasetReader.h"

class YoloDatasetConverter {
public:
    YoloDatasetConverter(const std::string& outPath, DatasetReader& reader);
    void process(bool overWriteclassWithZero);
    void finishConversion();

private:
    std::string outPath;
    DatasetReader& reader;
};


#endif //SAMPLERGENERATOR_YOLODATASETCONVERTER_H
