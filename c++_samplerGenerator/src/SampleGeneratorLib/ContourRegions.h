//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_CONTOURREGION_H
#define SAMPLERGENERATOR_CONTOURREGION_H

#include <opencv2/opencv.hpp>
#include "Region.h"
struct ContourRegion:Region {
    ContourRegion(const std::vector<cv::Point>& detections);
    ContourRegion(const std::string& jsonPath);
    void saveJson(const std::string& outPath);
    std::vector<cv::Point> getRegion();
    void drawRegion(cv::Mat& image);

private:
    std::vector<cv::Point> regions;
};


#endif //SAMPLERGENERATOR_CONTOURREGION_H
