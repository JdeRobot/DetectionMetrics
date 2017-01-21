//
// Created by frivas on 21/01/17.
//

#include "ContourRegion.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <fstream>
#include "rapidjson/filereadstream.h"

ContourRegion::ContourRegion(const std::vector<cv::Point> &detections) {
    this->regions = std::vector<cv::Point>(detections);
}



void ContourRegion::saveJson(const std::string &outPath) {
    rapidjson::Document d;
    d.SetObject();
    d.SetArray();
    for (auto it = this->regions.begin(), end=this->regions.end(); it != end; it++){
        rapidjson::Value point;
        point.SetObject();
        rapidjson::Value xValue(it->x);
        point.AddMember("x",xValue,d.GetAllocator());

        rapidjson::Value yValue(it->y);
        point.AddMember("y",yValue,d.GetAllocator());

        d.PushBack(point,d.GetAllocator());
    }

    rapidjson::StringBuffer buffer;

    buffer.Clear();

    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);

    std::ofstream outFile(outPath);
    outFile << buffer.GetString() << std::endl;
    outFile.close();
}


ContourRegion::ContourRegion(const std::string &jsonPath) {
    FILE* fp = fopen(jsonPath.c_str(), "rb"); // non-Windows use "r"
    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);
    this->regions.clear();
    for (auto it = d.Begin(), end= d.End(); it != end; ++it){
        cv::Point point;
        point.x = (*it)["x"].GetInt();
        point.y = (*it)["y"].GetInt();
        this->regions.push_back(point);
    }
}

std::vector<cv::Point> ContourRegion::getRegion() {
    return this->regions;
}

void ContourRegion::drawRegion(cv::Mat &image) {
    cv::Mat mask=cv::Mat(image.size(), CV_8UC1,cv::Scalar(0));
    cv::Scalar color( 255);
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(this->regions);
    cv::drawContours( mask, contours, 0, color, CV_FILLED, 8);
    std::vector<cv::Mat> channels;
    cv::split(image,channels);
    cv::Mat colorMask(image.size(),CV_8UC1,cv::Scalar(255));
    colorMask.copyTo(channels[0],mask);
    colorMask.copyTo(channels[1],mask);
    cv::Mat image2show;
    cv::merge(channels,image2show);
    std::cout << image2show.size() << std::endl;

    image2show.copyTo(image);
}



