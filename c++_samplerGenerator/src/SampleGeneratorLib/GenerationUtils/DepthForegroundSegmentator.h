//
// Created by frivas on 17/11/16.
//

#ifndef SAMPLERGENERATOR_DEPTHFOREGROUNDSEGMENTADOR_H
#define SAMPLERGENERATOR_DEPTHFOREGROUNDSEGMENTADOR_H
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <jderobot/depthLib/DepthFilter.h>

class DepthForegroundSegmentator {
public:
    DepthForegroundSegmentator(bool filterActive=true);
    std::vector<std::vector<cv::Point>>process(const cv::Mat& image);
    cv::Mat process2(const cv::Mat& image);



private:
    cv::Ptr<cv::BackgroundSubtractor> bg;
    //cv::BackgroundSubtractorMOG2* bg;
    cv::Mat fore;
    bool filterActive;
    boost::shared_ptr<jderobot::DepthFilter> filter;
    double defaultLearningRate;
    double minBlobArea;
};


#endif //SAMPLERGENERATOR_DEPTHFOREGROUNDSEGMENTADOR_H
