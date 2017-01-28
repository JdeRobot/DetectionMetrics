//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_RECTREGIONS_H
#define SAMPLERGENERATOR_RECTREGIONS_H

#include <opencv2/opencv.hpp>
#include "Regions.h"
#include "RectRegion.h"

struct RectRegions:Regions {
    RectRegions(const std::string& jsonPath);
    RectRegions();

    void add(const cv::Rect rect, int classId);
    void add(const std::vector<cv::Point>& detections, int classId);
    void add(int x, int y, int w, int h, int classId);


    void saveJson(const std::string& outPath);
    RectRegion getRegion(int id);
    std::vector<RectRegion> getRegions();
    void drawRegions(cv::Mat& image);
    void filterSamplesByID(std::vector<int> filteredIDS);
    bool empty();




private:
    std::vector<RectRegion> regions;
};


#endif //SAMPLERGENERATOR_RECTREGIONS_H
