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
        Logger::getInstance()->error("Output directory already exists");
        exit(-1);
    }
}
