//
// Created by frivas on 4/02/17.
//

#ifndef SAMPLERGENERATOR_LIVEREADER_H
#define SAMPLERGENERATOR_LIVEREADER_H

#include <string>
#include <DatasetConverters/readers/DatasetReader.h>
#include <DatasetConverters/liveReaders/VideoReader.h>

enum LIVEREADER_IMPLEMENTATIONS{RECORDER, JDEROBOT, VIDEO};


class GenericLiveReader {
public:
    GenericLiveReader(const std::string& path, const std::string& classNamesFile, const std::string& readerImplementation);
    GenericLiveReader(const std::vector<std::string>& paths,const std::string& classNamesFile, const std::string& readerImplementation);

    DatasetReaderPtr getReader();

    static std::vector<std::string> getAvailableImplementations();

private:
    LIVEREADER_IMPLEMENTATIONS imp;
    VideoReaderPtr videoReaderPtr;

    std::vector<std::string> availableImplementations;

    static void configureAvailablesImplementations(std::vector<std::string>& data);
    LIVEREADER_IMPLEMENTATIONS getImplementation(const std::string& readerImplementation);
};


typedef boost::shared_ptr<GenericLiveReader> GenericLiveReaderPtr;

#endif //SAMPLERGENERATOR_GENERICDATASETREADER_H
