//
// Created by frivas on 22/01/17.
//

#include "Sample.h"
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
    return this->colorImage;
}

cv::Mat Sample::getDepthImage() {
    return this->depthImage;
}

Sample::Sample(const std::string &path, const std::string &id,bool loadDepth) {
    this->colorImage=cv::imread(path + "/"  + id + ".png");

    this->rectRegions= RectRegions(path + "/" + id + ".json");
    this->contourRegions=ContourRegions(path + "/" + id + "-region.json");

    if (loadDepth) {
        this->depthImage=cv::imread(path + "/" + id + "-depth.png");
    }
}

cv::Mat Sample::getSampledColorImage() {
    cv::Mat image;
    this->colorImage.copyTo(image);
    this->rectRegions.drawRegions(image);
    this->contourRegions.drawRegions(image);
    return image;
}

cv::Mat Sample::getSampledDepthImage() {
    cv::Mat image;
    this->depthImage.copyTo(image);
    this->rectRegions.drawRegions(image);
    this->contourRegions.drawRegions(image);
    return image;
}

void Sample::save(const std::string &outPath, int id) {
    std::stringstream ss ;
    ss << std::setfill('0') << std::setw(5) << id;
    cv::Mat imageRGB;
    cv::cvtColor(colorImage,imageRGB,CV_RGB2BGR);
    cv::imwrite(outPath + "/" + ss.str() + ".png",imageRGB);
    cv::imwrite(outPath + "/" + ss.str() + "-depth.png",depthImage);
    rectRegions.saveJson(outPath + "/" + ss.str() + ".json");
    contourRegions.saveJson(outPath + "/" + ss.str() + "-region.json");
}


