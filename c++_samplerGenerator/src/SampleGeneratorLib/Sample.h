//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_SAMPLE_H
#define SAMPLERGENERATOR_SAMPLE_H

#include <Regions/RectRegions.h>
#include <Regions/ContourRegions.h>

struct Sample {
    Sample(const std::string& path, const std::string& id, bool loadDepth=true);
    Sample();
    Sample(const cv::Mat& colorImage);
    Sample(const cv::Mat& colorImage, const RectRegions& rectRegions);
    Sample(const cv::Mat& colorImage, const ContourRegions& contourRegions);
    Sample(const cv::Mat& colorImage, const RectRegions& rectRegions, const ContourRegions& contourRegions);
    Sample(const cv::Mat& colorImage, const cv::Mat& depthImage, const RectRegions& rectRegions);
    Sample(const cv::Mat& colorImage, const cv::Mat& depthImage, const ContourRegions& contourRegions);
    Sample(const cv::Mat& colorImage, const cv::Mat& depthImage, const RectRegions& rectRegions, const ContourRegions& contourRegions);
    ~Sample();
    void setColorImage(const std::string& imagePath);
    void setColorImage(const cv::Mat& image);
    void setDepthImage(const cv::Mat& image);
    void setDepthImage(const std::string& imagePath);
    void setRectRegions(const RectRegions& regions);
    void setContourRegions(const ContourRegions& regions);
    void setSampleID(const std::string& sampleID);

    RectRegions getRectRegions();
    ContourRegions getContourRegions();
    cv::Mat getColorImage();
    cv::Mat getDepthImage();
    cv::Mat getSampledColorImage();
    cv::Mat getSampledDepthImage();
    void save(const std::string& outPath, int id);
    void save(const std::string& outPath, const std::string& filename);
    void save(const std::string& outPath);

    bool isValid();
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    std::string getSampleID();



private:
    RectRegions rectRegions;
    ContourRegions contourRegions;
    cv::Mat colorImage;
    std::string colorImagePath;
    cv::Mat depthImage;
    std::string depthImagePath;
    std::string sampleID;
};


#endif //SAMPLERGENERATOR_SAMPLE_H
