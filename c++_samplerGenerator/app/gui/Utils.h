//
// Created by frivas on 19/02/17.
//

#ifndef SAMPLERGENERATOR_UTILS_H
#define SAMPLERGENERATOR_UTILS_H


#include <QtWidgets/QListView>

class Utils {
public:
    static bool getListViewContent(const QListView* list,std::vector<std::string>& content ,const std::string& prefix);

};


#endif //SAMPLERGENERATOR_UTILS_H
