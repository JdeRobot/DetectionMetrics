//
// Created by frivas on 5/02/17.
//

#include "OwnDatasetWriter.h"

OwnDatasetWriter::OwnDatasetWriter(const std::string &outPath, DatasetReaderPtr &reader) : DatasetWriter(outPath,
                                                                                                       reader) {

}

void OwnDatasetWriter::process() {
    Sample sample;
    while (reader->getNetxSample(sample)){
        sample.save(outPath);
    }
}
