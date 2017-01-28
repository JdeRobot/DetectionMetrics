//
// Created by frivas on 25/01/17.
//


#include "ClassType.h"

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

cv::Scalar ClassType::getColor() {

    int offset = classType*123457 % numClases;
    float red = _get_color(2,offset,numClases);
    float green = _get_color(1,offset,numClases);
    float blue = _get_color(0,offset,numClases);
    return cv::Scalar(red*255, green*255,blue*255);
}

float ClassType::_get_color(int c, int x, int max) {

    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}

ClassType::ClassType(ClassTypeEnum classType):classType(classType) {
    numClases=20;
}

ClassType::ClassType(int id){
    classType =ClassTypeEnum (id);
    numClases=20;
}


std::string ClassType::getName() {
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

