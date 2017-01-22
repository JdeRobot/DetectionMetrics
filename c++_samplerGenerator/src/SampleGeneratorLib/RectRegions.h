//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_RECTREGION_H
#define SAMPLERGENERATOR_RECTREGION_H

#include <opencv2/opencv.hpp>
#include "Region.h"

struct RectRegions:Region {
    RectRegions(const std::string& jsonPath);
    RectRegions();

    void add(const cv::Rect rect);
    void add(const std::vector<cv::Point>& detections);
    void add(int x, int y, int w, int h);


    void saveJson(const std::string& outPath);
    cv::Rect getRegion(int id);
    std::vector<cv::Rect> getRegions();
    void drawRegions(cv::Mat& image);



private:
    std::vector<cv::Rect> regions;
};


#endif //SAMPLERGENERATOR_RECTREGION_H
