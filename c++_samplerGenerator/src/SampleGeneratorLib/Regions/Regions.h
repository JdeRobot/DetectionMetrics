//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_REGION_H
#define SAMPLERGENERATOR_REGION_H


#include <opencv2/opencv.hpp>

struct Regions{
    Regions(){};
    virtual void saveJson(const std::string& outPath)=0;
    virtual void drawRegions(cv::Mat& image)=0;
    virtual void filterSamplesByID(std::vector<std::string> filteredIDS)=0;
    virtual bool empty()=0;
    virtual void print()=0;

};

#endif //SAMPLERGENERATOR_REGION_H
