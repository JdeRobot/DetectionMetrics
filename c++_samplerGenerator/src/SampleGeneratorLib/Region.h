//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_REGION_H
#define SAMPLERGENERATOR_REGION_H


#include <opencv2/opencv.hpp>

struct Region{
    Region(){};
    virtual void saveJson(const std::string& outPath)=0;
    virtual void drawRegion(cv::Mat& image)=0;

};

#endif //SAMPLERGENERATOR_REGION_H
