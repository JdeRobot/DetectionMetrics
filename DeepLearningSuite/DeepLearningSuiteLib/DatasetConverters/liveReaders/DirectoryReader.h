
#ifndef SAMPLERGENERATOR_DIRECTORYREADER_H
#define SAMPLERGENERATOR_DIRECTORYREADER_H

#include <Common/Sample.h>
#include "DatasetConverters/readers/DatasetReader.h"
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <unordered_set>

class DirectoryReader: public DatasetReader {

public:
    DirectoryReader(const std::string& directoryPath);

    bool getNextSample(Sample &sample);
    int getNumberOfElements();

private:
    std::vector<std::string> listOfImages;
    long long int sample_offset = 0;
};

typedef boost::shared_ptr<DirectoryReader> DirectoryReaderPtr;

#endif
