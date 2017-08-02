//
// Created by frivas on 22/07/17.
//

#ifndef SAMPLERGENERATOR_GLOBALSTATS_H
#define SAMPLERGENERATOR_GLOBALSTATS_H

#include <map>
#include "ClassStatistics.h"

class GlobalStats {
public:
    GlobalStats();
    void addTruePositive(const std::string &classID);

    void addFalsePositive(const std::string &classID);

    void addFalseNegative(const std::string &classID);

    void addIOU(const std::string &classID, double value);

    void printStats(const std::vector<std::string>& classesToDisplay) const;

private:
    std::map<std::string,ClassStatistics> statsMap;

};


#endif //SAMPLERGENERATOR_GLOBALSTATS_H
