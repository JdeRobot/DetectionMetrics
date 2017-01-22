//
// Created by frivas on 20/01/17.
//

#ifndef SAMPLERGENERATOR_DETECTIONSVALIDATOR_H
#define SAMPLERGENERATOR_DETECTIONSVALIDATOR_H


#include <opencv2/opencv.hpp>

class DetectionsValidator {
public:
    DetectionsValidator(const std::string& pathToSave);
    ~DetectionsValidator();
    void validate(const cv::Mat& colorImage,const cv::Mat& depthImage, std::vector<std::vector<cv::Point>>& detections);

private:
    int validationCounter;
    std::string path;
};


#endif //SAMPLERGENERATOR_DETECTIONSVALIDATOR_H
