//
// Created by frivas on 31/01/17.
//

#include <Common/Sample.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include "DarknetInferencer.h"
#include <glog/logging.h>

DarknetInferencer::DarknetInferencer(const std::string &netConfig, const std::string &netWeights,const std::string& classNamesFile): netConfig(netConfig),netWeights(netWeights) {
    LOG(INFO)<< "--Darknet Inferencer creation ---" << std::endl;
    LOG(INFO)<< this->netConfig.c_str()  << std::endl;
    LOG(INFO)<< this->netWeights.c_str()  << std::endl;
    LOG(INFO)<< classNamesFile  << std::endl;
    this->classNamesFile=classNamesFile;
    this->cnn = boost::shared_ptr<DarknetAPI>(new DarknetAPI((char*)this->netConfig.c_str(), (char*)this->netWeights.c_str()));
    LOG(INFO)<< this->cnn  << std::endl;
}

Sample DarknetInferencer::detectImp(const cv::Mat &image, double confidence_threshold) {
    LOG(INFO)<< "--Darknet Inferencer detection ---" << std::endl;
    cv::Mat rgbImage;
    LOG(INFO)<< "--Darknet Inferencer detection 1 ---" << std::endl;
    cv::cvtColor(image,rgbImage,cv::COLOR_RGB2BGR);
    LOG(INFO)<< "--Darknet Inferencer detection 1.5 --- " << (float)confidence_threshold << std::endl;
    DarknetDetections detections = this->cnn->process(rgbImage, (float)confidence_threshold);
    LOG(INFO)<< "--Darknet Inferencer detection 2 ---" << std::endl;
    Sample sample;
    /*RectRegionsPtr regions(new RectRegions());
    ClassTypeGeneric typeConverter(classNamesFile);

    for (auto it = detections.data.begin(), end=detections.data.end(); it !=end; ++it){
        LOG(INFO)<< "--Darknet Inferencer detection 3 ---" << std::endl;
	LOG(INFO)<< it->classId  << std::endl;
	typeConverter.setId(it->classId);
        regions->add(it->detectionBox,typeConverter.getClassString(), it->probability);
        LOG(INFO)<< typeConverter.getClassString() << ": " << it->probability << std::endl;
    }
    sample.setColorImage(image);
    sample.setRectRegions(regions);*/
    return sample;
}
