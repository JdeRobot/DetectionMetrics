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
#include <Sample.h>
#include <boost/lexical_cast.hpp>


DetectionsValidator::DetectionsValidator(const std::string& pathToSave):validationCounter(0), path(pathToSave) {
    auto boostPath= boost::filesystem::path(this->path);
    if (!boost::filesystem::exists(boostPath)){
        boost::filesystem::create_directories(boostPath);
    }
    else{
        boost::filesystem::directory_iterator end_itr;

        for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
        {
            if ((is_regular_file(itr->status()) && itr->path().extension()==".png") && (itr->path().string().find("-depth") == std::string::npos)) {
                validationCounter++;
            }

        }
        Logger::getInstance()->warning("Including samples to an existing dataset, starting with: " + boost::lexical_cast<std::string>(this->validationCounter));

    }

}

DetectionsValidator::~DetectionsValidator() {

}



void DetectionsValidator::validate(const cv::Mat& colorImage,const cv::Mat& depthImage, std::vector<std::vector<cv::Point>>& detections){

    cv::Mat mask=cv::Mat(colorImage.size(), CV_8UC1,cv::Scalar(0));

    for (auto it= detections.begin(), end = detections.end(); it != end; ++it){
        int idx= std::distance(detections.begin(),it);
        cv::Scalar color( 255);
        cv::drawContours( mask, detections, idx, color, CV_FILLED, 8);
    }


    std::vector<cv::Mat> channels;
    cv::split(colorImage,channels);
    cv::Mat colorMask(colorImage.size(),CV_8UC1,cv::Scalar(255));
    colorMask.copyTo(channels[0],mask);
    cv::Mat image2show;
    cv::merge(channels,image2show);

    int validationID=14; //todo configure anywhere

    RectRegions regions;
    ContourRegions cRegions;

    BoundingValidator validator(image2show);
    for (auto it= detections.begin(), end=detections.end(); it != end; ++it){
        cv::Rect validatedRect;
        if (validator.validate(*it,validatedRect)){
            Logger::getInstance()->info("Validated");
            regions.add(validatedRect,validationID);
            cRegions.add(*it,validationID);
        }
        else{
            Logger::getInstance()->info("Discarded");
        }
    }

    if (regions.getRegions().size()){
        Sample sample(colorImage,depthImage,regions,cRegions);
        sample.save(this->path,this->validationCounter);
        this->validationCounter++;
    }
}
