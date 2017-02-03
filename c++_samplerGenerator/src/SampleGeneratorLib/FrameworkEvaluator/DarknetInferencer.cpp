//
// Created by frivas on 31/01/17.
//

#include <Sample.h>
#include <DatasetConverters/ClassTypeVoc.h>
#include "DarknetInferencer.h"

DarknetInferencer::DarknetInferencer(const std::string &netConfig, const std::string &netWeights): netConfig(netConfig),netWeights(netWeights) {
    this->cnn = boost::shared_ptr<DarknetAPI>(new DarknetAPI((char*)this->netConfig.c_str(), (char*)this->netWeights.c_str()));
}

Sample DarknetInferencer::detect(const cv::Mat &image) {
    cv::Mat rgbImage;
    cv::cvtColor(image,rgbImage,CV_RGB2BGR);
    DarknetDetections detections = this->cnn->process(rgbImage);

    Sample sample;
    RectRegions regions;
    for (auto it = detections.data.begin(), end=detections.data.end(); it !=end; ++it){
        ClassTypeVoc typeConverter(it->classId);
        regions.add(it->detectionBox,typeConverter.getClassString());
        std::cout<< (it->classId) << ": " << it->probability << std::endl;
    }
    sample.setColorImage(image);
    sample.setRectRegions(regions);
    return sample;
}
