//
// Created by frivas on 25/01/17.
//


#include "ClassTypeVoc.h"

ClassTypeVoc::ClassTypeVoc(int id) {
    fillStringClassesVector();
    this->classID=this->classes[id];

}

ClassTypeVoc::ClassTypeVoc(const std::string &classID) {
    fillStringClassesVector();
    this->classID=classID;

}

void ClassTypeVoc::fillStringClassesVector() {
    classes.push_back("aeroplane");
    classes.push_back("bicycle");
    classes.push_back("bird");
    classes.push_back("boat");
    classes.push_back("bottle");
    classes.push_back("bus");
    classes.push_back("car");
    classes.push_back("cat");
    classes.push_back("chair");
    classes.push_back("cow");
    classes.push_back("diningtable");
    classes.push_back("dog");
    classes.push_back("horse");
    classes.push_back("motorbike");
    classes.push_back("person");
    classes.push_back("pottedplant");
    classes.push_back("sheep");
    classes.push_back("sofa");
    classes.push_back("train");
    classes.push_back("tvmonitor");
}
