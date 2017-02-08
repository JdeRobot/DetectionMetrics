//
// Created by frivas on 7/02/17.
//

#ifndef SAMPLERGENERATOR_STATSUTILS_H
#define SAMPLERGENERATOR_STATSUTILS_H

#include <opencv2/opencv.hpp>

class StatsUtils {
public:
    static double getIOU(const cv::Rect& gt, const cv::Rect& detection,const cv::Size& imageSize);

};


#endif //SAMPLERGENERATOR_STATSUTILS_H
