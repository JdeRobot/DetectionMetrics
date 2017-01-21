//
// Created by frivas on 22/11/16.
//

#ifndef SAMPLERGENERATOR_BOUNDINGVALIDATOR_H
#define SAMPLERGENERATOR_BOUNDINGVALIDATOR_H


#include <opencv2/opencv.hpp>

class BoundingValidator {
public:
    BoundingValidator(const cv::Mat& image_in);
    bool validate(std::vector<cv::Point>& bounding);

private:
    double scale;
    cv::Mat image;


    static void CallBackFunc(int event, int x, int y, int flags, void* userdata);

};


#endif //SAMPLERGENERATOR_BOUNDINGVALIDATOR_H
