#include "RleRegions.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <fstream>
#include "rapidjson/filereadstream.h"


RleRegions::RleRegions(){

}

void RleRegions::add(RLE region, const std::string& classId, const bool isCrowd) {
    RleRegion regionToInsert(region, classId, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);
}

void RleRegions::add(RLE region, const std::string& classId, const double confidence_score, const bool isCrowd) {
    RleRegion regionToInsert(region, classId, confidence_score, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);
    //regions.push_back(RectRegion(rect, cla
}


void RleRegions::saveJson(const std::string &outPath) {
    rapidjson::Document d;
    /*d.SetObject();
    d.SetArray();
    for (auto it = this->regions.begin(), end=this->regions.end(); it != end; it++){
        rapidjson::Value detection;
        detection.SetObject();
        rapidjson::Value idValue(it->classID.c_str(),d.GetAllocator());
        detection.AddMember("classID",idValue,d.GetAllocator());

        rapidjson::Value regionValue;
        regionValue.SetArray();

        for (auto it2=it->region.begin(), end2= it->region.end(); it2 != end2; ++it2) {
            rapidjson::Value point;
            point.SetObject();
            rapidjson::Value xValue(it2->x);
            point.AddMember("x", xValue, d.GetAllocator());

            rapidjson::Value yValue(it2->y);
            point.AddMember("y", yValue, d.GetAllocator());

            regionValue.PushBack(point, d.GetAllocator());
        }
        detection.AddMember("region",regionValue,d.GetAllocator());
        d.PushBack(detection,d.GetAllocator());
    }

    rapidjson::StringBuffer buffer;

    buffer.Clear();

    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);

    std::ofstream outFile(outPath);
    outFile << buffer.GetString() << std::endl;
    outFile.close();*/
}




RleRegion RleRegions::getRegion(int idx) {
    if (this->regions.size() -1 >= idx)
        return this->regions[idx];
    else
        return RleRegion();
}

void RleRegions::drawRegions(cv::Mat &image) {
  std::cout << regions.size() << '\n';
    for (auto it = regions.begin(), end= regions.end(); it != end; ++it) {
        cv::Mat mask = cv::Mat(it->region.w, it->region.h, CV_8UC1, cv::Scalar(0));
        std::cout << rleToString(&(it->region)) << '\n';
        rleDecode(&(it->region), mask.data , 1);
        std::cout << "Decoding Done" << '\n';
        mask = mask * 255;
        cv::rotate(mask, mask, cv::ROTATE_90_CLOCKWISE);
        cv::flip(mask, mask, 1);
        cv::imshow("mask", mask);
        cv::waitKey(0);

        //std::cout << image.rows << " " << mask.rows << " " << image.cols << " " << mask.cols << '\n';
        cv::Scalar color(255);
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        cv::Mat colorMask(image.size(), CV_8UC1, cv::Scalar(255));
        colorMask.copyTo(channels[0], mask);
        colorMask.copyTo(channels[1], mask);
        cv::Mat image2show;
        cv::merge(channels, image2show);
        image2show.copyTo(image);

    }

}

std::vector<RleRegion> RleRegions::getRegions() {
    return this->regions;
}

void RleRegions::filterSamplesByID(std::vector<std::string> filteredIDS) {
    std::vector<RleRegion> oldRegions(this->regions);
    this->regions.clear();
    for(auto it = oldRegions.begin(), end=oldRegions.end(); it != end; ++it) {
        if (std::find(filteredIDS.begin(), filteredIDS.end(), it->classID) != filteredIDS.end()) {
            this->regions.push_back(*it);
        }
    }
}

bool RleRegions::empty() {
    return (this->regions.size()==0);
}

void RleRegions::print() {
    //todo
    std::cout << "Not yet implemented" << std::endl;
}
