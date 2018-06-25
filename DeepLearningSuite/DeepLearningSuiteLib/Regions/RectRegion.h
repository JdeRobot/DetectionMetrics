//
// Created by frivas on 25/01/17.
//

#ifndef SAMPLERGENERATOR_RECTREGION_H
#define SAMPLERGENERATOR_RECTREGION_H

#include <opencv2/opencv.hpp>

struct RectRegion {

    RectRegion():valid(false){};
    RectRegion(const cv::Rect& region, const std::string& classID):region(region),classID(classID),valid(true){};
    RectRegion(const cv::Rect& region, const std::string& classID, const double confidence_score):region(region),classID(classID),confidence_score(confidence_score),valid(true){};

    bool operator < (const RectRegion &obj) const {

      if (classID.empty() || obj.classID.empty()) {
         throw std::invalid_argument("One of the RectRegions passed for comparision were not initialized");
      }

      if (classID != obj.classID) {
         std::cout << "returning not equal class" << '\n';
         return classID < obj.classID;

      } else {
         std::cout << "came here" << '\n';
         return confidence_score > obj.confidence_score;          //Reverse Sorting of Confidence Scores
      }

   }

    cv::Rect region;
    std::string classID;
    int uniqObjectID;
    double confidence_score = -1;
    bool valid;

};


#endif //SAMPLERGENERATOR_RECTREGION_H
