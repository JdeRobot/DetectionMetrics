//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_SAMPLE_H
#define SAMPLERGENERATOR_SAMPLE_H

#include <RectRegions.h>
#include <ContourRegions.h>

struct Sample {
    Sample(const std::string& path, const std::string& id, bool loadDepth= true);
    Sample();
    Sample(const cv::Mat& colorImage);
    Sample(const cv::Mat& colorImage, const RectRegions& rectRegions);
    Sample(const cv::Mat& colorImage, const ContourRegions& contourRegions);
    Sample(const cv::Mat& colorImage, const RectRegions& rectRegions, const ContourRegions& contourRegions);
    Sample(const cv::Mat& colorImage, const cv::Mat& depthImage, const RectRegions& rectRegions);
    Sample(const cv::Mat& colorImage, const cv::Mat& depthImage, const ContourRegions& contourRegions);
    Sample(const cv::Mat& colorImage, const cv::Mat& depthImage, const RectRegions& rectRegions, const ContourRegions& contourRegions);
    void setColorImage(const cv::Mat& image);
    void setDepthImage(const cv::Mat& image);
    void setRectRegions(const RectRegions& regions);
    void setContourRegions(const ContourRegions& regions);
    RectRegions getRectRegions();
    ContourRegions getContourRegions();
    cv::Mat getColorImage();
    cv::Mat getDepthImage();
    cv::Mat getSampledColorImage();
    cv::Mat getSampledDepthImage();
    void save(const std::string& outPath, int id);


private:
    RectRegions rectRegions;
    ContourRegions contourRegions;
    cv::Mat colorImage;
    cv::Mat depthImage;
};


#endif //SAMPLERGENERATOR_SAMPLE_H
