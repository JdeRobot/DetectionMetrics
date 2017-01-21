//
// Created by frivas on 20/01/17.
//

#include <boost/filesystem.hpp>
#include <iomanip>
#include "DetectionsValidator.h"
#include "BoundingValidator.h"
#include "Logger.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <fstream>
#include <RectRegion.h>
#include <ContourRegion.h>



DetectionsValidator::DetectionsValidator(const std::string& pathToSave):validationCounter(0), path(pathToSave) {
    auto boostPath= boost::filesystem::path(this->path);
    if (!boost::filesystem::exists(boostPath)){
        boost::filesystem::create_directories(boostPath);
    }

}

DetectionsValidator::~DetectionsValidator() {

}



void DetectionsValidator::validate(const cv::Mat&image, std::vector<std::vector<cv::Point>>& detections){

    cv::Mat mask=cv::Mat(image.size(), CV_8UC1,cv::Scalar(0));

    for (auto it= detections.begin(), end = detections.end(); it != end; ++it){
        int idx= std::distance(detections.begin(),it);
        cv::Scalar color( 255);
        cv::drawContours( mask, detections, idx, color, CV_FILLED, 8);
    }


    std::vector<cv::Mat> channels;
    cv::split(image,channels);
    cv::Mat colorMask(image.size(),CV_8UC1,cv::Scalar(255));
    colorMask.copyTo(channels[0],mask);
    cv::Mat image2show;
    cv::merge(channels,image2show);


    BoundingValidator validator(image2show);
    for (auto it= detections.begin(), end=detections.end(); it != end; ++it){

        if (validator.validate(*it)){
            Logger::getInstance()->info("Validated");
            saveDetection(image,*it);
        }
        else{
            Logger::getInstance()->info("Discarded");
        }
    }
}




void DetectionsValidator::saveDetection(const cv::Mat&image, std::vector<cv::Point>& detections){

    RectRegion region(detections);
    ContourRegion cRegion(detections);


    std::cout << "Saving: " << region.getRegion()  << " [" << this->validationCounter << "]" << std::endl;
    std::stringstream ss ;
    ss << std::setfill('0') << std::setw(5) << this->validationCounter;
    cv::Mat imageRGB;
    cv::cvtColor(image,imageRGB,CV_RGB2BGR);
    cv::imwrite(this->path + "/" + ss.str() + ".png",imageRGB);
    region.saveJson(this->path + "/" + ss.str() + ".json");
    cRegion.saveJson(this->path + "/" + ss.str() + "-region.json");


    this->validationCounter++;
}