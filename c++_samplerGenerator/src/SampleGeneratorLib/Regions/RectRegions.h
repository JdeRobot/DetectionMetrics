//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_RECTREGIONS_H
#define SAMPLERGENERATOR_RECTREGIONS_H

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include "Regions.h"
#include "RectRegion.h"

struct RectRegions:Regions {
    RectRegions(const std::string& jsonPath);
    RectRegions();

    void add(const cv::Rect rect, const std::string classId);
    void add(const std::vector<cv::Point>& detections, const std::string classId);
    void add(int x, int y, int w, int h, const std::string classId);


    void saveJson(const std::string& outPath);
    RectRegion getRegion(int id);
    std::vector<RectRegion> getRegions();
    void drawRegions(cv::Mat& image);
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    bool empty();
    void print();

    std::vector<RectRegion> regions;
};

typedef boost::shared_ptr<RectRegions> RectRegionsPtr;


#endif //SAMPLERGENERATOR_RECTREGIONS_H
