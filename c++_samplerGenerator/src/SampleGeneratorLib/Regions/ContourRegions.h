//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_CONTOURREGIONS_H
#define SAMPLERGENERATOR_CONTOURREGIONS_H

#include <opencv2/opencv.hpp>
#include "Regions.h"
#include "ContourRegion.h"

struct ContourRegions:Regions {
    ContourRegions(const std::string& jsonPath);
    ContourRegions();
    void saveJson(const std::string& outPath);
    void add(const std::vector<cv::Point>& detections, int classId);
    ContourRegion getRegion(int idx);
    std::vector<ContourRegion> getRegions();
    void drawRegions(cv::Mat& image);
    void filterSamplesByID(std::vector<int> filteredIDS);
    bool empty();



private:
    std::vector<ContourRegion> regions;
};


#endif //SAMPLERGENERATOR_CONTOURREGIONS_H
