//
// Created by frivas on 19/02/17.
//

#include <glog/logging.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include "TabHandler.h"
#include "ListViewConfig.h"


TabHandler::TabHandler() {
    fillContexts();
}



void TabHandler::fillContexts() {
    this->contexts.push_back("viewer");
    this->contexts.push_back("converter");
}

std::string TabHandler::getStringContext(int index) {
    return this->contexts[index];
}

std::vector<std::string> TabHandler::getAllContexts() {
    return this->contexts;
}
