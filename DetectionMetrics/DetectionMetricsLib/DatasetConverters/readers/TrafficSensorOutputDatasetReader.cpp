#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <boost/lexical_cast.hpp>
#include "DatasetConverters/readers/TrafficSensorOutputDatasetReader.h"
#include "DatasetConverters/ClassTypeGeneric.h"

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

TrafficSensorOutputDatasetReader::TrafficSensorOutputDatasetReader(const std::string &path,const std::string& classNamesFile,const bool imagesRequired):DatasetReader(imagesRequired){
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}

TrafficSensorOutputDatasetReader::TrafficSensorOutputDatasetReader(const bool imagesRequired):DatasetReader(imagesRequired) {}

bool TrafficSensorOutputDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    boost::filesystem::directory_iterator end_itr;
    boost::filesystem::path boostPath(datasetPath);
    std::vector<std::string> filesID;
    ClassTypeGeneric typeConverter(this->classNamesFile);

    for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr) {
        if ((is_regular_file(itr->status()) && itr->path().extension()==".txt") && (itr->path().string().find("-region") == std::string::npos)) {
            filesID.push_back(itr->path().string());
        }
    }

    std::sort(filesID.begin(),filesID.end());
    for (auto it = filesID.begin(), end=filesID.end(); it != end; ++it) {
        std::ifstream inFile(boost::filesystem::path(*it).string());
        std::string line;
        while (getline(inFile,line) && !line.empty()) {
            Sample sample;
            sample.setSampleID(datasetPrefix + boost::filesystem::path(*it).filename().stem().string());

            std::ifstream labelFile(line);
            std::string data;
            RectRegionsPtr rectRegions(new RectRegions());

            std::istringstream iss(line);
            std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

            cv::Rect bounding(std::stoi(tokens.at(0)), std::stoi(tokens.at(1)), std::stoi(tokens.at(0)) + std::stoi(tokens.at(2)), std::stoi(tokens.at(1)) + std::stoi(tokens.at(4)));
            typeConverter.setId(std::stoi(tokens.at(0)));
            rectRegions->add(bounding, typeConverter.getClassString());
            labelFile.close();
            sample.setRectRegions(rectRegions);
            this->samples.push_back(sample);
        }
    }

    LOG(INFO) << "Loaded: " + boost::lexical_cast<std::string>(this->samples.size()) + " samples";
    printDatasetStats();
    return true;
}