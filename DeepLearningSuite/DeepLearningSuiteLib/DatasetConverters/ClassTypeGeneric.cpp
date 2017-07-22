//
// Created by frivas on 9/02/17.
//

#include <fstream>
#include "ClassTypeGeneric.h"


ClassTypeGeneric::ClassTypeGeneric(const std::string &classesFile) {
    fillStringClassesVector(classesFile);
}

ClassTypeGeneric::ClassTypeGeneric(const std::string &classesFile, int id) {
    fillStringClassesVector(classesFile);
    this->classID=this->classes[id];
}

void ClassTypeGeneric::fillStringClassesVector(const std::string &classesFile) {
    std::ifstream labelFile(classesFile);
    std::string data;
    while(getline(labelFile,data)) {
        this->classes.push_back(data);
    }
}

void ClassTypeGeneric::setId(int id) {
    this->classID=this->classes[id];
}

