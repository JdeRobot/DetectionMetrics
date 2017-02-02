//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_CLASSTYPECOCO_H
#define SAMPLERGENERATOR_CLASSTYPECOCO_H


#include <opencv2/opencv.hpp>

enum ClassTypeCocoEnum{
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


struct ClassTypeCoco{
    ClassTypeCoco(ClassTypeCocoEnum classType);
    ClassTypeCoco(int id);
    cv::Scalar getColor();
    std::string getName();
private:
    int numClases;

    float _get_color(int c, int x, int max);
    ClassTypeCocoEnum classType;

};


#endif //SAMPLERGENERATOR_CLASSTYPECOCO_H
