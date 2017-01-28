//
// Created by frivas on 25/01/17.
//

#ifndef SAMPLERGENERATOR_RECTREGION_H
#define SAMPLERGENERATOR_RECTREGION_H

#include <opencv2/opencv.hpp>

struct RectRegion {

    RectRegion():valid(false){};
    RectRegion(const cv::Rect& region, int id):region(region),id(id),valid(true){};

    cv::Rect region;
    int id;
    bool valid;

};


#endif //SAMPLERGENERATOR_RECTREGION_H
