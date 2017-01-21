//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_RECTREGION_H
#define SAMPLERGENERATOR_RECTREGION_H

#include <opencv2/opencv.hpp>
#include "Region.h"

struct RectRegion:Region {
    RectRegion(const cv::Rect rect);
    RectRegion(const std::vector<cv::Point>& detections);
    RectRegion(int x, int y, int w, int h);
    RectRegion(const std::string& jsonPath);
    void saveJson(const std::string& outPath);
    cv::Rect getRegion();
    void drawRegion(cv::Mat& image);



private:
    cv::Rect region;
};


#endif //SAMPLERGENERATOR_RECTREGION_H
