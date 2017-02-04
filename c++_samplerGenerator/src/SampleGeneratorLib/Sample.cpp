//
// Created by frivas on 22/01/17.
//

#include "Sample.h"
#include "Utils/Logger.h"
#include <iomanip>


Sample::Sample() {

}

Sample::Sample(const cv::Mat &colorImage) {
    colorImage.copyTo(this->colorImage);
}

Sample::Sample(const cv::Mat &colorImage, const RectRegions &rectRegions) {
    this->setColorImage(colorImage);
    this->setRectRegions(rectRegions);
}

Sample::Sample(const cv::Mat &colorImage, const ContourRegions &contourRegions) {
    this->setColorImage(colorImage);
    this->setContourRegions(contourRegions);
}

Sample::Sample(const cv::Mat &colorImage, const RectRegions &rectRegions, const ContourRegions &contourRegions) {
    this->setColorImage(colorImage);
    this->setRectRegions(rectRegions);
    this->setContourRegions(contourRegions);
}

Sample::Sample(const cv::Mat &colorImage, const cv::Mat &depthImage, const RectRegions &rectRegions) {
    this->setColorImage(colorImage);
    this->setDepthImage(depthImage);
    this->setRectRegions(rectRegions);
}

Sample::Sample(const cv::Mat &colorImage, const cv::Mat &depthImage, const ContourRegions &contourRegions) {
    this->setColorImage(colorImage);
    this->setDepthImage(depthImage);
    this->setContourRegions(contourRegions);
}

Sample::Sample(const cv::Mat &colorImage, const cv::Mat &depthImage, const RectRegions &rectRegions,
               const ContourRegions &contourRegions) {
    this->setColorImage(colorImage);
    this->setDepthImage(depthImage);
    this->setRectRegions(rectRegions);
    this->setContourRegions(contourRegions);

}

void Sample::setColorImage(const cv::Mat &image) {
    image.copyTo(this->colorImage);
}

void Sample::setDepthImage(const cv::Mat &image) {
    image.copyTo(this->depthImage);
}

void Sample::setRectRegions(const RectRegions &regions) {
    this->rectRegions=regions;
}

void Sample::setContourRegions(const ContourRegions &regions) {
    this->contourRegions=regions;
}

RectRegions Sample::getRectRegions() {
    return this->rectRegions;
}

ContourRegions Sample::getContourRegions() {
    return this->contourRegions;
}

cv::Mat Sample::getColorImage() {
    if (this->colorImage.empty()) {
        cv::Mat image = cv::imread(this->colorImagePath);
        return image;
    }
    else
        return this->colorImage.clone();
}

cv::Mat Sample::getDepthImage() {
    if (this->depthImage.empty()) {
        cv::Mat image = cv::imread(this->depthImagePath);
        return image;
    }
    else
        return this->depthImage.clone();
}

Sample::Sample(const std::string &path, const std::string &id,bool loadDepth) {
    this->colorImagePath=path + "/"  + id + ".png";

    this->rectRegions= RectRegions(path + "/" + id + ".json");
    this->contourRegions=ContourRegions(path + "/" + id + "-region.json");

    if (loadDepth) {
        this->depthImagePath=path + "/" + id + "-depth.png";
    }
}

cv::Mat Sample::getSampledColorImage() {
    cv::Mat image = this->getColorImage();
    this->rectRegions.drawRegions(image);
    this->contourRegions.drawRegions(image);
    return image;
}

cv::Mat Sample::getSampledDepthImage() {
    cv::Mat image =this->getDepthImage();
    this->rectRegions.drawRegions(image);
    this->contourRegions.drawRegions(image);
    return image;
}

void Sample::save(const std::string &outPath, int id) {
    std::stringstream ss ;
    ss << std::setfill('0') << std::setw(5) << id;
    this->save(outPath,ss.str());

}

void Sample::save(const std::string &outPath, const std::string &filename) {

    cv::imwrite(outPath + "/" + filename + ".png",this->colorImage);
    if (! this->depthImage.empty()) {
        cv::imwrite(outPath + "/" + filename + "-depth.png", depthImage);
    }
    rectRegions.saveJson(outPath + "/" + filename + ".json");
    contourRegions.saveJson(outPath + "/" + filename + "-region.json");
}

void Sample::save(const std::string &outPath) {
    if (this->sampleID.size() != 0 ){
        this->save(outPath,this->sampleID);
    }
    else{
        Logger::getInstance()->error("No sample id is defined, this sample will not be saved");
    }

}




bool Sample::isValid() {
    return !this->colorImage.empty();
}

void Sample::filterSamplesByID(std::vector<std::string> filteredIDS) {
    this->rectRegions.filterSamplesByID(filteredIDS);
    this->contourRegions.filterSamplesByID(filteredIDS);
}

void Sample::setColorImage(const std::string &imagePath) {
    this->colorImagePath=imagePath;
}

void Sample::setDepthImage(const std::string &imagePath) {
    this->depthImagePath=imagePath;
}

void Sample::setSampleID(const std::string &sampleID) {
    this->sampleID=sampleID;
}

std::string Sample::getSampleID() {
    return this->sampleID;
}

Sample::~Sample() {
    if (!this->colorImage.empty()){
        this->colorImage.release();
    }
    if (this->depthImage.empty()){
        this->depthImage.release();
    }

}



