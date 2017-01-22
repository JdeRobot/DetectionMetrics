//
// Created by frivas on 22/11/16.
//

#include "BoundingRect.h"

BoundingRect::BoundingRect(const std::vector<cv::Point> &points):points(points) {

}

BoundingRect::BoundingRect(const cv::Rect &rectangle) {
    points.push_back(cv::Point(rectangle.x,rectangle.y));
    points.push_back(cv::Point(rectangle.x + rectangle.width,rectangle.y));
    points.push_back(cv::Point(rectangle.x + rectangle.width,rectangle.y + rectangle.height));
    points.push_back(cv::Point(rectangle.x,rectangle.y + rectangle.height));
}




std::vector<cv::Point> BoundingRect::getPoints() {
    return points;
}

void BoundingRect::move(const cv::Point &from, const cv::Point &to) {
    unsigned int idx1,idx2;

    getClosestsLinePoints(from,idx1,idx2);

    cv::Point& p1 = points[idx1];
    cv::Point& p2 = points[idx2];


//    std::cout << "closest points: " << p1 << ", " << p2<< std::endl;
    cv::Point movement;
    if (isVertical(p1,p2)){
        movement = cv::Point(to.x-from.x,0);
    }
    else{
        movement = cv::Point(0,to.y-from.y);
    }

//    std::cout << "Vertical: " << isVertical(p1,p2) << "  Movement: " << movement << std::endl;
//    std::cout << "Previous : " << p1 << ", " << p2 << std::endl;
    p1= p1 + movement;
    p2 = p2 + movement;


}

void BoundingRect::getClosestsLinePoints(const cv::Point &from, unsigned int &p1, unsigned int &p2) {

    double minDistance=999999999;

//    std::cout << "number of points: " << points.size() << std::endl;

    for (unsigned int i=0; i < points.size(); i++){
        int j=i+1;
        if (j >= points.size())
            j=0;

        cv::Point midpoint = (points[i] + points[j])*0.5;
        double distance = cv::norm(midpoint-from);
//        std::cout << "distance:" << distance  << std::endl;
        if (distance < minDistance){
            minDistance = distance;
            p1 = i;
            p2 = j;
        }
    }
}

bool BoundingRect::isVertical(const cv::Point &from, const cv::Point &to) {
    int xMovement = abs(from.x-to.x);
    int yMovement = abs(from.y-to.y);
//    std::cout << "Xmov: " << xMovement << ", ymov: " << yMovement << std::endl;
    return abs(from.x-to.x) <  abs(from.y-to.y);
}

cv::Rect BoundingRect::getRect(const double scale) {
    return cv::Rect(points[0].x/scale, points[0].y/scale,(points[1].x - points[0].x)/scale,(points[2].y - points[1].y)/scale);
}

