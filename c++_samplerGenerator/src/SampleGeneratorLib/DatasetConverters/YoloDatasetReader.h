//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_YOLODATASETREADER_H
#define SAMPLERGENERATOR_YOLODATASETREADER_H


#include "DatasetReader.h"

class YoloDatasetReader: public DatasetReader {
public:
    YoloDatasetReader(const std::string& path);
};

typedef boost::shared_ptr<YoloDatasetReader> YoloDatasetReaderPtr;

#endif //SAMPLERGENERATOR_YOLODATASETREADER_H
