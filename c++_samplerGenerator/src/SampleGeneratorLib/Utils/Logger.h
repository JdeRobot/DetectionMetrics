//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_LOGGER_H
#define SAMPLERGENERATOR_LOGGER_H


#include <boost/shared_ptr.hpp>

class Logger {
public:
    enum  LogLevel {DEBUG, INFO, WARNING, ERROR };

    static boost::shared_ptr<Logger> getInstance();
    void debug(const std::string msg);
    void info(const std::string msg);
    void warning(const std::string msg);
    void error(const std::string msg);
    void setLevel(LogLevel level);
    ~Logger();


private:
    Logger();
    LogLevel logLevel;

};


#endif //SAMPLERGENERATOR_LOGGER_H
