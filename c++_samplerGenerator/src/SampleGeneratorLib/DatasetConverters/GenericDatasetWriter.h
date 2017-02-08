//
// Created by frivas on 5/02/17.
//

#ifndef SAMPLERGENERATOR_GENERICDATASETWRITTER_H
#define SAMPLERGENERATOR_GENERICDATASETWRITTER_H

enum WRITER_IMPLEMENTATIONS{WR_OWN, WR_YOLO};

#include <string>
#include "DatasetWriter.h"
#include "OwnDatasetWriter.h"
#include "YoloDatasetWriter.h"

class GenericDatasetWriter {
public:
    GenericDatasetWriter(const std::string& path,DatasetReaderPtr &reader, const std::string& writerImplementation);
    DatasetWriterPtr getWriter();

private:
    WRITER_IMPLEMENTATIONS imp;


    YoloDatasetWriterPtr yoloDatasetWriterPtr;
    OwnDatasetWriterPtr ownDatasetWriterPtr;

    std::vector<std::string> availableImplementations;

    void configureAvailablesImplementations();
    WRITER_IMPLEMENTATIONS getImplementation(const std::string& writerImplementation);

};


typedef boost::shared_ptr<GenericDatasetWriter> GenericDatasetWriterPtr;


#endif //SAMPLERGENERATOR_GENERICDATASETWRITTER_H
