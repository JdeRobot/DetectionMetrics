//
// Created by frivas on 22/11/16.
//

#include "BoundingValidator.h"
#include "BoundingRect.h"

BoundingValidator::BoundingValidator(const cv::Mat &image_in) {
    this->image=image_in.clone();
    this->scale=3;
    cv::cvtColor(this->image, this->image, CV_RGB2BGR);
    cv::resize(this->image,this->image,cv::Size(), scale,scale);


}

bool BoundingValidator::validate(std::vector<cv::Point> &bounding) {

    std::vector<cv::Point> scaledBounding;
    for (auto it = bounding.begin(), end = bounding.end(); it != end; ++it){
        scaledBounding.push_back((*it) * this->scale);
    }
    cv::Rect boundingRectangle = cv::boundingRect(scaledBounding);
    BoundingRect rect(boundingRectangle);
    int key='0';
    while (char(key) != 'n' and char(key) != ' ') {
        cv::Mat image2show= this->image.clone();
        cv::rectangle(image2show, rect.getRect(), cv::Scalar(255, 0, 0), 3);
        std::string windowName="Validation window";
        cv::namedWindow(windowName, 1);
        cv::setMouseCallback(windowName, CallBackFunc, &rect);
        cv::imshow(windowName, image2show);
        key=cv::waitKey(1);
        //std::cout << "key: " << char(key) << std::endl;
    }

    return (char(key)==' ');
}

bool clicked=false;
cv::Point from, to;
cv::Point tempFrom;


void BoundingValidator::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    BoundingRect* rect = (BoundingRect*)userdata;
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        if (!clicked) {
            clicked = true;
            from=cv::Point(x,y);
            tempFrom=cv::Point(x,y);

        }
    }
    else if  ( event == cv::EVENT_LBUTTONUP )
    {
        if (clicked) {
            clicked = false;
            /*to=cv::Point(x,y);
            std::cout << "moving from: " << from << ", to: " << to << std::endl;
            rect->move(from,to);*/
        }
    }
    else if ( event == cv::EVENT_MOUSEMOVE )
    {
        if (clicked) {
            to=cv::Point(x,y);
//            std::cout << "moving from: " << from << ", to: " << to << std::endl;
            rect->move(tempFrom,to);
            tempFrom=cv::Point(x,y);


        }
    }
}