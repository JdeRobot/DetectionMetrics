//
// Created by frivas on 25/01/17.
//

#ifndef SAMPLERGENERATOR_CLASSTYPEVOC_H
#define SAMPLERGENERATOR_CLASSTYPEVOC_H


#include "ClassType.h"

struct ClassTypeVoc: public ClassType{
    ClassTypeVoc(int id);
    ClassTypeVoc(const std::string& classID);
    void fillStringClassesVector();
};

#endif //SAMPLERGENERATOR_CLASSTYPE_H
