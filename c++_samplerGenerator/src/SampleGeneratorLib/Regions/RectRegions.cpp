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
#include <DatasetConverters/ClassTypeOwn.h>



RectRegions::RectRegions(const std::string &jsonPath) {
    FILE* fp = fopen(jsonPath.c_str(), "rb"); // non-Windows use "r"
    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);
    for (auto it=d.Begin(), end = d.End(); it != end; ++it){
        cv::Rect reg((*it)["x"].GetInt(),(*it)["y"].GetInt(),(*it)["w"].GetInt(),(*it)["h"].GetInt());
        std::string id;
        id= (*it)["id"].GetString();
        regions.push_back(RectRegion(reg,id));
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
        rapidjson::Value xValue(it->region.x);
        node.AddMember("x",xValue,d.GetAllocator());

        rapidjson::Value yValue(it->region.y);
        node.AddMember("y",yValue,d.GetAllocator());

        rapidjson::Value wValue(it->region.width);
        node.AddMember("w",wValue,d.GetAllocator());

        rapidjson::Value hValue(it->region.height);
        node.AddMember("h",hValue,d.GetAllocator());

        rapidjson::Value idValue(it->classID.c_str(), d.GetAllocator());
        node.AddMember("id",idValue,d.GetAllocator());

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

void RectRegions::add(const cv::Rect rect,const std::string classId){
    regions.push_back(RectRegion(rect,classId));
}

void RectRegions::add(const std::vector<cv::Point> &detections,const std::string classId) {
    regions.push_back(RectRegion(cv::boundingRect(detections),classId));
}

void RectRegions::add(int x, int y, int w, int h,const std::string classId) {
    regions.push_back(RectRegion(cv::Rect(x,y,w,h),classId));
}

RectRegion RectRegions::getRegion(int id) {
    if (this->regions.size() -1 >= id)
        return this->regions[id];
    else
        return RectRegion();
}

void RectRegions::drawRegions(cv::Mat &image) {
    for (auto it = regions.begin(), end=regions.end(); it != end; ++it) {
        ClassTypeOwn classType(it->classID);
        cv::rectangle(image, it->region, classType.getColor(), 2);
    }

}

std::vector<RectRegion> RectRegions::getRegions() {
    return this->regions;
}

void RectRegions::filterSamplesByID(std::vector<std::string> filteredIDS) {
    std::vector<RectRegion> oldRegions(this->regions);
    this->regions.clear();
    for(auto it = oldRegions.begin(), end=oldRegions.end(); it != end; ++it) {
        if (std::find(filteredIDS.begin(), filteredIDS.end(), it->classID) != filteredIDS.end()) {
            this->regions.push_back(*it);
        }
    }
}

bool RectRegions::empty() {
    return (this->regions.size()==0);
}

void RectRegions::print() {
    std::cout << "-------------------" << std::endl;
    for (auto it = this->regions.begin(), end = this->regions.end(); it != end; ++it){
        int idx = std::distance(this->regions.begin(),it);
        std::cout << "[" << idx << "]: " << it->region << " (" << it->classID << ")" << std::endl;
    }
    std::cout << "-------------------" << std::endl;
}

