//
// Created by frivas on 21/01/17.
//

#include "RectRegions.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <fstream>
#include "rapidjson/filereadstream.h"



RectRegions::RectRegions(const std::string &jsonPath) {
    FILE* fp = fopen(jsonPath.c_str(), "rb"); // non-Windows use "r"
    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);
    for (auto it=d.Begin(), end = d.End(); it != end; ++it){
        regions.push_back(cv::Rect((*it)["x"].GetInt(),(*it)["y"].GetInt(),(*it)["w"].GetInt(),(*it)["h"].GetInt()));
    }
}
RectRegions::RectRegions() {

}


void RectRegions::saveJson(const std::string &outPath) {
    rapidjson::Document d;
    d.SetArray();
    for (auto it = regions.begin(), end= regions.end(); it != end; ++it){
        rapidjson::Value node;
        node.SetObject();
        rapidjson::Value xValue(it->x);
        node.AddMember("x",xValue,d.GetAllocator());

        rapidjson::Value yValue(it->y);
        node.AddMember("y",yValue,d.GetAllocator());

        rapidjson::Value wValue(it->width);
        node.AddMember("w",wValue,d.GetAllocator());

        rapidjson::Value hValue(it->height);
        node.AddMember("h",hValue,d.GetAllocator());
        d.PushBack(node,d.GetAllocator());
    }



    rapidjson::StringBuffer buffer;

    buffer.Clear();

    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);

    std::ofstream outFile(outPath);
    outFile << buffer.GetString() << std::endl;
    outFile.close();
}

void RectRegions::add(const cv::Rect rect){
    regions.push_back(rect);
}

void RectRegions::add(const std::vector<cv::Point> &detections) {
    regions.push_back(cv::boundingRect(detections));
}

void RectRegions::add(int x, int y, int w, int h) {
    regions.push_back(cv::Rect(x,y,w,h));
}

cv::Rect RectRegions::getRegion(int id) {
    if (this->regions.size() -1 >= id)
        return this->regions[id];
    else
        return cv::Rect();
}

void RectRegions::drawRegions(cv::Mat &image) {
    for (auto it = regions.begin(), end=regions.end(); it != end; ++it) {
        cv::rectangle(image, *it, cv::Scalar(255, 0, 0), 2);
    }

}

std::vector<cv::Rect> RectRegions::getRegions() {
    return this->regions;
}

