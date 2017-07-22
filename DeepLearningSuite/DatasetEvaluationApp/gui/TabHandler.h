//
// Created by frivas on 19/02/17.
//

#ifndef SAMPLERGENERATOR_TABHANDLER_H
#define SAMPLERGENERATOR_TABHANDLER_H


#include <boost/shared_ptr.hpp>
#include <Utils/Configuration.h>
#include <QtWidgets/QMainWindow>
#include <mainwindow.h>

class TabHandler {
public:
    TabHandler();
    std::string getStringContext(int index);
    std::vector<std::string> getAllContexts();

private:
    std::vector<std::string> contexts;

    void fillContexts();
};


typedef boost::shared_ptr<TabHandler> TabHandlerPtr;

#endif //SAMPLERGENERATOR_TABHANDLER_H
