#include "OpenImagesDatasetWriter.h"
#include "DatasetConverters/ClassTypeMapper.h"
#include <iomanip>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>

using namespace rapidjson;

OpenImagesDatasetWriter::OpenImagesDatasetWriter(const std::string &outPath, DatasetReaderPtr &reader, const std::string& writerNamesFile, bool overWriteclassWithZero):DatasetWriter(outPath,reader),overWriteclassWithZero(overWriteclassWithZero), writerNamesFile(writerNamesFile){
    LOG(INFO) << "--1--" << "\n"; 
}



void OpenImagesDatasetWriter::process(bool writeImages, bool useDepth) {
    LOG(INFO) << "--2--" << "\n";

}
