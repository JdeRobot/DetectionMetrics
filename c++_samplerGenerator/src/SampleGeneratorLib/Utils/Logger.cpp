//
// Created by frivas on 21/01/17.
//

#include <iostream>
#include "Logger.h"

static boost::shared_ptr<Logger> logger;

Logger::Logger():logLevel(ERROR) {
}

Logger::~Logger() {

}

boost::shared_ptr<Logger> Logger::getInstance() {
    if (! logger){
        logger=boost::shared_ptr<Logger>(new Logger);
    }
    return logger;
}

void Logger::debug(const std::string msg) {
    if (this->logLevel<=DEBUG)
        std::cout << "DEBUG: " << msg << std::endl;
}

void Logger::info(const std::string msg) {
    if (this->logLevel<=INFO)
        std::cout << "INFO: " << msg << std::endl;
}

void Logger::warning(const std::string msg) {
    if (this->logLevel<=WARNING)
        std::cout << "WARNING: " << msg << std::endl;
}

void Logger::error(const std::string msg) {
    if (this->logLevel<=ERROR)
        std::cout << "ERROR: " << msg << std::endl;
}

void Logger::setLevel(Logger::LogLevel level) {
    this->logLevel=level;
}


