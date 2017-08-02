//
// Created by frivas on 22/07/17.
//

#include "GlobalStats.h"


GlobalStats::GlobalStats() = default;


void GlobalStats::addTruePositive(const std::string &classID) {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].truePositives = this->statsMap[classID].truePositives+1;
    }
    else{
        ClassStatistics s(classID);
        s.truePositives = s.truePositives+1;
        this->statsMap[classID]=s;
    }
}

void GlobalStats::addFalsePositive(const std::string &classID) {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].falsePositives = this->statsMap[classID].falsePositives+1;
    }
    else{
        ClassStatistics s(classID);
        s.falsePositives = s.falsePositives+1;
        this->statsMap[classID]=s;
    }
}

void GlobalStats::addFalseNegative(const std::string &classID)  {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].falseNegatives = this->statsMap[classID].falseNegatives+1;
    }
    else{
        ClassStatistics s(classID);
        s.falseNegatives = s.falseNegatives+1;
        this->statsMap[classID]=s;
    }
}

void GlobalStats::addIOU(const std::string &classID, double value)  {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].iou.push_back(value);
    }
    else{
        ClassStatistics s(classID);
        s.iou.push_back(value);
        this->statsMap[classID]=s;
    }
}

void GlobalStats::printStats(const std::vector<std::string>& classesToDisplay) const{
    if (classesToDisplay.empty()) {
        for (auto it : this->statsMap) {
            it.second.printStats();
        }
    }
    else{
        for (const std::string& it : classesToDisplay){
            if (this->statsMap.count(it))
                auto data =this->statsMap.at(it);
        }
    }
}


