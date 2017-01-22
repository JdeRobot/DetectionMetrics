//
// Created by frivas on 16/11/16.
//

#include "RecorderConverter.h"
#include "Logger.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <iostream>

RecorderConverter::RecorderConverter(const std::string &colorImagesPath, const std::string &depthImagesPath):colorPath(colorImagesPath), depthPath(depthImagesPath) {
    currentIndex=0;

    getImagesByIndexes(depthPath,depthIndexes);
    getImagesByIndexes(colorPath,colorIndexes);
}


void RecorderConverter::getImagesByIndexes(const std::string& path, std::vector<int>& indexes){
    indexes.clear();
    if(boost::filesystem::is_directory(path)) {


        boost::filesystem::directory_iterator end_iter;

        for (boost::filesystem::directory_iterator dir_itr(path);
             dir_itr != end_iter; dir_itr++) {

            if (boost::filesystem::is_regular_file(*dir_itr) && dir_itr->path().extension() == ".png") {
                boost::filesystem::path filePath;
                indexes.push_back(std::stoi(dir_itr->path().filename().stem().string()));
            }
        }
    }
    std::sort(indexes.begin(), indexes.end());
}


std::string RecorderConverter::getPathByIndex(const std::string& path, const int id){
    std::stringstream ss;
    ss << id << ".png";
    return path + ss.str();
}



int RecorderConverter::closest(std::vector<int> const& vec, int value) {
    auto const it = std::lower_bound(vec.begin(), vec.end(), value);
    if (it == vec.end()) { return -1; }

    return *it;
}

bool RecorderConverter::getNext(std::string &colorImagePath, std::string &depthImagePath) {
    if (this->currentIndex < this->depthIndexes.size()){
        int indexValue = this->depthIndexes[currentIndex];
        Logger::getInstance()->info("Time stamp: " + boost::lexical_cast<std::string>(indexValue));
        colorImagePath=getPathByIndex(this->colorPath,closest(colorIndexes,indexValue));
        depthImagePath=getPathByIndex(this->depthPath,indexValue);
        this->currentIndex++;
        return true;
    }
    else{
        return false;
    }
}

int RecorderConverter::getNumSamples() {
    return this->depthIndexes.size();
}
