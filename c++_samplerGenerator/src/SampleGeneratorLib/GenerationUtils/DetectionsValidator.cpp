//
// Created by frivas on 20/01/17.
//

#include <boost/filesystem.hpp>
#include <iomanip>
#include "DetectionsValidator.h"
#include "BoundingValidator.h"
#include "Utils/Logger.h"
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
        char confirmation='a';
        while (confirmation != 'y' && confirmation != 'n'){
            std::cout << "Do you want to continue? (y/n)" << std::endl;
            std::cin >> confirmation;
        }
        if (confirmation=='n'){
            Logger::getInstance()->warning("Exiting");
            exit(1);
        }

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


    RectRegionsPtr regions(new RectRegions());
    ContourRegionsPtr cRegions(new ContourRegions());

    cv::cvtColor(image2show,image2show,CV_RGB2BGR);
    BoundingValidator validator(image2show);
    for (auto it= detections.begin(), end=detections.end(); it != end; ++it){
        cv::Rect validatedRect;
        int classVal;
        if (validator.validate(*it,validatedRect,classVal)){
            std::string validationID;
            if (char(classVal)=='1') {
                validationID = "person";
            }
            else if (char(classVal)=='2')
                validationID="person-falling";
            else if (char(classVal)=='3')
                validationID="person-fall";


            fillRectIntoImageDimensions(validatedRect,colorImage.size());
            Logger::getInstance()->info("Validated");
            regions->add(validatedRect,validationID);
            cRegions->add(*it,validationID);
        }
        else{
            Logger::getInstance()->info("Discarded");
        }
    }

    if (regions->getRegions().size()){
        Sample sample(colorImage,depthImage,regions,cRegions);
        sample.save(this->path,this->validationCounter);
        this->validationCounter++;
    }
}

void DetectionsValidator::fillRectIntoImageDimensions(cv::Rect &rect, const cv::Size size) {
    if (rect.x + rect.width > size.width){
        rect.width = size.width - rect.x - 1;
    }
    if (rect.y + rect.height > size.height){
        rect.height = size.height - rect.y -1;
    }

    if (rect.x < 0){
        rect.width=rect.width + rect.x;
        rect.x=0;
    }
    if (rect.y <0){
        rect.height = rect.height + rect.y;
        rect.y=0;
    }

}
