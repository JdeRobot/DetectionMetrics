//
// Created by frivas on 22/11/16.
//

#ifndef SAMPLERGENERATOR_BOUDINGRECT_H
#define SAMPLERGENERATOR_BOUDINGRECT_H

#include <opencv2/opencv.hpp>

struct BoundingRect {
public:

    BoundingRect(const std::vector<cv::Point>& points);
    BoundingRect(const cv::Rect& rectangle);

    std::vector<cv::Point> getPoints();
    void move(const cv::Point& from, const cv::Point& to);
    cv::Rect getRect();


private:
    std::vector<cv::Point> points;


    void getClosestsLinePoints(const cv::Point &from,unsigned int& p1, unsigned int& p2);
    bool isVertical(const cv::Point& from, const cv::Point& to);

};


#endif //SAMPLERGENERATOR_BOUDINGRECT_H
