//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_CONTOURREGION_H
#define SAMPLERGENERATOR_CONTOURREGION_H

#include <opencv2/opencv.hpp>
#include "Region.h"
struct ContourRegions:Region {
    ContourRegions(const std::string& jsonPath);
    ContourRegions();
    void saveJson(const std::string& outPath);
    void add(const std::vector<cv::Point>& detections);
    std::vector<cv::Point> getRegion(int idx);
    std::vector<std::vector<cv::Point>> getRegions();
    void drawRegions(cv::Mat& image);

private:
    std::vector<std::vector<cv::Point>> regions;
};


#endif //SAMPLERGENERATOR_CONTOURREGION_H
