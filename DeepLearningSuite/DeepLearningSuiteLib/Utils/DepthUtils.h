//
// Created by frivas on 30/07/17.
//

#ifndef SAMPLERGENERATOR_DEPTHUTILS_H
#define SAMPLERGENERATOR_DEPTHUTILS_H

#include <opencv2/core/core.hpp>

class DepthUtils {
public:
    static void mat16_to_ownFormat(const cv::Mat& inputImage, cv::Mat& outImage);
};


#endif //SAMPLERGENERATOR_DEPTHUTILS_H
