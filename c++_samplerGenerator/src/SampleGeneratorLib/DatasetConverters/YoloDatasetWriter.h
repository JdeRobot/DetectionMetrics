//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_YOLODATASETCONVERTER_H
#define SAMPLERGENERATOR_YOLODATASETCONVERTER_H

#include <string>
#include "DatasetReader.h"
#include "DatasetWriter.h"

class YoloDatasetWriter: public DatasetWriter {
public:
    YoloDatasetWriter(const std::string& outPath, DatasetReaderPtr& reader, bool overWriteclassWithZero=true);
    void process(bool usedColorImage=true);

private:
    std::string fullImagesPath;
    std::string fullLabelsPath;
    bool overWriteclassWithZero;
};

typedef  boost::shared_ptr<YoloDatasetWriter> YoloDatasetWriterPtr;


#endif //SAMPLERGENERATOR_YOLODATASETCONVERTER_H
