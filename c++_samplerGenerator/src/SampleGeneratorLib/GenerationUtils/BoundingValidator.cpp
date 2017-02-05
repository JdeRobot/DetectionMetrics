//
// Created by frivas on 22/11/16.
//

#include "BoundingValidator.h"
#include "BoundingRectGuiMover.h"

BoundingValidator::BoundingValidator(const cv::Mat &image_in) {
    this->image=image_in.clone();
    this->scale=3;
    cv::cvtColor(this->image, this->image, CV_RGB2BGR);
    cv::resize(this->image,this->image,cv::Size(), scale,scale);


}

bool BoundingValidator::validate(std::vector<cv::Point> &bounding, cv::Rect& validatedBound, int& key) {


    std::vector<char> validationKeys;
    validationKeys.push_back('1');
    validationKeys.push_back('2');
    validationKeys.push_back('3');


    char rejectionKey='n';
    std::vector<cv::Point> scaledBounding;
    for (auto it = bounding.begin(), end = bounding.end(); it != end; ++it){
        scaledBounding.push_back((*it) * this->scale);
    }
    cv::Rect boundingRectangle = cv::boundingRect(scaledBounding);
    BoundingRectGuiMover rect(boundingRectangle);
    key='0';
    while (char(key) != rejectionKey and (std::find(validationKeys.begin(), validationKeys.end(), char(key))== validationKeys.end() )) {
        cv::Mat image2show= this->image.clone();
        cv::rectangle(image2show, rect.getRect(), cv::Scalar(255, 0, 0), 3);
        std::string windowName="Validation window";
        cv::namedWindow(windowName, 1);
        cv::setMouseCallback(windowName, CallBackFunc, &rect);
        cv::imshow(windowName, image2show);
        key=cv::waitKey(1);
        //std::cout << "key: " << char(key) << std::endl;
    }
    validatedBound=rect.getRect(this->scale);


    return std::find(validationKeys.begin(), validationKeys.end(), char(key))!= validationKeys.end();
}

bool clicked=false;
cv::Point from, to;
cv::Point tempFrom;
BoundingRectGuiMover::MovementType movementType = BoundingRectGuiMover::NONE;



void BoundingValidator::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    BoundingRectGuiMover* rect = (BoundingRectGuiMover*)userdata;
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        if (!clicked) {
            movementType = BoundingRectGuiMover::LOCAL_MOVEMENT;
            clicked = true;
            from=cv::Point(x,y);
            tempFrom=cv::Point(x,y);

        }
    }
    else if  ( event == cv::EVENT_LBUTTONUP )
    {
        if (clicked) {
            movementType = BoundingRectGuiMover::NONE;
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
            rect->move(tempFrom,to,movementType);
            tempFrom=cv::Point(x,y);


        }
    }
    else if (event == cv::EVENT_MBUTTONDOWN){
        if (!clicked) {
            movementType = BoundingRectGuiMover::GLOBAL_MOVEMENT;
            clicked = true;
            from=cv::Point(x,y);
            tempFrom=cv::Point(x,y);

        }
    }
    else if (event == cv::EVENT_MBUTTONUP){
        movementType = BoundingRectGuiMover::NONE;
        if (clicked) {
            clicked = false;
        }
    }
}