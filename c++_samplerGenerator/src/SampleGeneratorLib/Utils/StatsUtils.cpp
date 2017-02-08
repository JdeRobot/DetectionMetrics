//
// Created by frivas on 7/02/17.
//

#include "StatsUtils.h"

double StatsUtils::getIOU(const cv::Rect &gt, const cv::Rect &detection, const cv::Size &imageSize) {
    //compute iou
    cv::Mat maskGT(imageSize, CV_8UC1, cv::Scalar(0));
    cv::Mat maskDetection(imageSize, CV_8UC1, cv::Scalar(0));

    cv::rectangle(maskGT, gt, cv::Scalar(255), -1);
    cv::rectangle(maskDetection, detection, cv::Scalar(255), -1);

    cv::Mat unionMask(imageSize, CV_8UC1, cv::Scalar(0));
    cv::rectangle(unionMask, gt, cv::Scalar(150), -1);
    cv::rectangle(unionMask, detection, cv::Scalar(255), -1);

    cv::Mat interSection = maskGT & maskDetection;

    int interSectionArea = cv::countNonZero(interSection);
    int unionArea = cv::countNonZero(unionMask);
    double iouValue = double(interSectionArea) / double(unionArea);
    return iouValue;
}
