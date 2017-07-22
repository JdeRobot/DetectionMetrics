//
// Created by frivas on 25/01/17.
//

#ifndef SAMPLERGENERATOR_RECTREGION_H
#define SAMPLERGENERATOR_RECTREGION_H

#include <opencv2/opencv.hpp>

struct RectRegion {

    RectRegion():valid(false){};
    RectRegion(const cv::Rect& region, const std::string& classID):region(region),classID(classID),valid(true){};

    cv::Rect region;
    std::string classID;
    bool valid;

};


#endif //SAMPLERGENERATOR_RECTREGION_H
