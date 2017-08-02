//
// Created by frivas on 30/07/17.
//

#include <iostream>
#include "DepthUtils.h"

void DepthUtils::mat16_to_ownFormat(const cv::Mat &inputImage, cv::Mat& outImage) {

    double MAX_LENGHT=10000;

    auto imageSize = inputImage.size();
    outImage=cv::Mat(imageSize,CV_8UC3,cv::Scalar(0,0,0));


    cv::Mat evalImage= inputImage.clone();
    evalImage=cv::Scalar(0,0,0);

    for (int y = 0; y < imageSize.height; ++y) {
        for (int x = 0; x < imageSize.width; ++x) {
            uint16_t value = inputImage.at<uint16_t>(y, x);
            value= (value >> 3);
            evalImage.at<uint16_t>(y, x) = value;
            outImage.data[(y*imageSize.width+ x)*3+0] = (float(value)/(float)MAX_LENGHT)*255.;
            outImage.data[(y*imageSize.width+ x)*3+1] = (value)>>8;
            outImage.data[(y*imageSize.width+ x)*3+2] = (value)&0xff;
        }
    }


    double min,max;
    cv::minMaxLoc(inputImage,&min,&max);

    std::cout << "min: " << min << std::endl;
    std::cout << "max: " << max << std::endl;

    cv::minMaxLoc(evalImage,&min,&max);

    std::cout << "min: " << min << std::endl;
    std::cout << "max: " << max << std::endl;



}
