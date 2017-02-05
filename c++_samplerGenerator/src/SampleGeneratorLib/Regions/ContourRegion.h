//
// Created by frivas on 26/01/17.
//

#ifndef SAMPLERGENERATOR_CONTOURREGION_H
#define SAMPLERGENERATOR_CONTOURREGION_H

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

struct ContourRegion {
    ContourRegion():valid(false){};
    ContourRegion(const ContourRegion& other);
    ContourRegion(const std::vector<cv::Point>& region, std::string id):region(region),id(id),valid(true){}; //person by default

    std::vector<cv::Point>region;
    std::string id;
    bool valid;

};


#endif //SAMPLERGENERATOR_CONTOURREGION_H
