//
// Created by frivas on 21/01/17.
//

#include "RectRegion.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <fstream>
#include "rapidjson/filereadstream.h"


void RectRegion::saveJson(const std::string &outPath) {
    rapidjson::Document d;
    d.SetObject();
    rapidjson::Value xValue(region.x);
    d.AddMember("x",xValue,d.GetAllocator());

    rapidjson::Value yValue(region.y);
    d.AddMember("y",yValue,d.GetAllocator());

    rapidjson::Value wValue(region.width);
    d.AddMember("w",wValue,d.GetAllocator());

    rapidjson::Value hValue(region.height);
    d.AddMember("h",hValue,d.GetAllocator());

    rapidjson::StringBuffer buffer;

    buffer.Clear();

    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);

    std::ofstream outFile(outPath);
    outFile << buffer.GetString() << std::endl;
    outFile.close();
}

RectRegion::RectRegion(const cv::Rect rect):region(rect){}

RectRegion::RectRegion(const std::vector<cv::Point> &detections) {
    region = cv::boundingRect(detections);
}

RectRegion::RectRegion(int x, int y, int w, int h) {
    region =cv::Rect(x,y,w,h);
}

RectRegion::RectRegion(const std::string &jsonPath) {
    FILE* fp = fopen(jsonPath.c_str(), "rb"); // non-Windows use "r"
    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);
    region = cv::Rect(d["x"].GetInt(),d["y"].GetInt(),d["w"].GetInt(),d["h"].GetInt());
}

cv::Rect RectRegion::getRegion() {
    return this->region;
}

void RectRegion::drawRegion(cv::Mat &image) {
    cv::rectangle(image,this->region,cv::Scalar(255,0,0),2);

}
