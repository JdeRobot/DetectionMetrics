//
// Created by frivas on 22/01/17.
//

#include <boost/filesystem.hpp>
#include <Logger.h>
#include <boost/lexical_cast.hpp>
#include "OwnDatasetReader.h"

OwnDatasetReader::OwnDatasetReader(const std::string &path):currentIndex(0) {
    this->datasetPath=path;

    boost::filesystem::directory_iterator end_itr;
    boost::filesystem::path boostPath(this->datasetPath);


    std::vector<std::string> filesID;

    for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
    {
        if ((is_regular_file(itr->status()) && itr->path().extension()==".png") && (itr->path().string().find("-depth") == std::string::npos)) {
            filesID.push_back(itr->path().filename().stem().string());
        }

    }

    std::sort(filesID.begin(),filesID.end());

    for (auto it = filesID.begin(), end=filesID.end(); it != end; ++it){
        Sample sample(this->datasetPath,*it);
        this->samples.push_back(sample);
    }

    Logger::getInstance()->info("Loaded: " + boost::lexical_cast<std::string>(this->samples.size()) + " samples");
}

bool OwnDatasetReader::getNetxSample(Sample& sample) {
    if (currentIndex < this->samples.size()) {
        sample = this->samples[this->currentIndex];
        this->currentIndex++;
        return true;
    }
    else
        return false;
}
