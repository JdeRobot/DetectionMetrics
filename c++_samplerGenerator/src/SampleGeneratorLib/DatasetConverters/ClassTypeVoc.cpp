//
// Created by frivas on 25/01/17.
//


#include "ClassTypeVoc.h"

float colorsClass[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

cv::Scalar ClassTypeVoc::getColor() {

    int offset = classType*123457 % numClases;
    float red = _get_color(2,offset,numClases);
    float green = _get_color(1,offset,numClases);
    float blue = _get_color(0,offset,numClases);
    return cv::Scalar(red*255, green*255,blue*255);
}

float ClassTypeVoc::_get_color(int c, int x, int max) {

    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colorsClass[i][c] + ratio*colorsClass[j][c];
    //printf("%f\n", r);
    return r;
}

ClassTypeVoc::ClassTypeVoc(ClassTypeEnum classType):classType(classType) {
    numClases=20;
}

ClassTypeVoc::ClassTypeVoc(int id){
    classType =ClassTypeEnum (id);
    numClases=20;
}


std::string ClassTypeVoc::getName() {
    switch (classType){
        case AEROPLANE:
            return "AEROPLANE";
            break;
        case BICYCLE:
            return "BICYCLE";
            break;
        case BIRD:
            return "BIRD";
            break;
        case BOAT:
            return "BOAT";
            break;
        case BOTTLE:
            return "BOTTLE";
            break;
        case BUS:
            return "BUS";
            break;
        case CAR:
            return "CAR";
            break;
        case CAT:
            return "CAT";
            break;
        case CHAIR:
            return "CHAIR";
            break;
        case COW:
            return "COW";
            break;
        case DININGTABLE:
            return "DININGTABLE";
            break;
        case DOG:
            return "DOG";
            break;
        case HORSE:
            return "HORSE";
            break;
        case MOTORBIKE:
            return "MOTORBIKE";
            break;
        case PERSON:
            return "PERSON";
            break;
        case POTTEDPLANT:
            return "POTTEDPLANT";
            break;
        case SHEEP:
            return "SHEEP";
            break;
        case SOFA:
            return "SOFA";
            break;
        case TRAIN:
            return "TRAIN";
            break;
        case TVMONITOR:
            return "TVMONITOR";
            break;
    }
}

