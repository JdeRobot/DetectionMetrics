//
// Created by frivas on 5/02/17.
//

#include <Utils/Logger.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include "DatasetWriter.h"

DatasetWriter::DatasetWriter(const std::string &outPath, DatasetReaderPtr &reader):outPath(outPath), reader(reader) {
    auto boostPath= boost::filesystem::path(outPath);
    if (!boost::filesystem::exists(boostPath)){
        boost::filesystem::create_directories(boostPath);
    }
    else{
        boost::filesystem::directory_iterator end_it;
        boost::filesystem::directory_iterator it(boostPath);
        if(it != end_it) {
            Logger::getInstance()->error("Output directory already exists and is not empty");
            exit(-1);
        }
    }
}
