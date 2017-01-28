//
// Created by frivas on 25/01/17.
//

#ifndef SAMPLERGENERATOR_CLASSTYPE_H
#define SAMPLERGENERATOR_CLASSTYPE_H


#include <opencv2/opencv.hpp>

enum ClassTypeEnum{
    AEROPLANE,
    BICYCLE,
    BIRD,
    BOAT,
    BOTTLE,
    BUS,
    CAR,
    CAT,
    CHAIR,
    COW,
    DININGTABLE,
    DOG,
    HORSE,
    MOTORBIKE,
    PERSON,
    POTTEDPLANT,
    SHEEP,
    SOFA,
    TRAIN,
    TVMONITOR
};


struct ClassType{
    ClassType(ClassTypeEnum classType);
    ClassType(int id);
    cv::Scalar getColor();
    std::string getName();
private:
    int numClases;

    float _get_color(int c, int x, int max);
    ClassTypeEnum classType;

};

#endif //SAMPLERGENERATOR_CLASSTYPE_H
