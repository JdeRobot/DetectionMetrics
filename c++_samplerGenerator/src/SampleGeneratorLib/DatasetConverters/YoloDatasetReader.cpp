//
// Created by frivas on 22/01/17.
//

#include <fstream>
#include <Utils/Logger.h>
#include <boost/filesystem/path.hpp>
#include "YoloDatasetReader.h"
#include "ClassTypeVoc.h"


bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

YoloDatasetReader::YoloDatasetReader(const std::string &path) {
    std::ifstream inFile(path);

    std::string line;
    while (getline(inFile,line)){
        Sample sample;
        sample.setSampleID(boost::filesystem::path(line).filename().stem().string());
        sample.setColorImage(line);
        Logger::getInstance()->info("Loading sample: " + line);
        cv::Mat image = cv::imread(line);
        replace(line,"JPEGImages", "labels");
        replace(line,".jpg", ".txt");
        std::ifstream labelFile(line);
        std::string data;
        RectRegions rectRegions;
        while(getline(labelFile,data)) {
            std::istringstream iss(data);
            int class_id;
            double x, y, w,h;
            iss >> class_id >> x >> y >> w >> h;
            cv::Rect bounding(x * image.size().width - (w * image.size().width)/2, y * image.size().height - (h * image.size().height)/2, w * image.size().width, h * image.size().height);

            ClassTypeVoc typeConverter(class_id);
            rectRegions.add(bounding,typeConverter.getClassString());
        }
        labelFile.close();
        sample.setRectRegions(rectRegions);
        this->samples.push_back(sample);
    }
}
