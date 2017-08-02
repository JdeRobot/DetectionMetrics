//
// Created by frivas on 1/02/17.
//

#include <numeric>
#include "ClassStatistics.h"
#include <iostream>

ClassStatistics::ClassStatistics(const std::string& classID):classID(classID),nSamples(0),truePositives(0),falsePositives(0), falseNegatives(0), trueNegatives(0){

}
ClassStatistics::ClassStatistics():classID(""),nSamples(0),truePositives(0),falsePositives(0), falseNegatives(0), trueNegatives(0){

}


double ClassStatistics::getMeanIOU() const{
   return std::accumulate( this->iou.begin(), this->iou.end(), 0.0)/this->iou.size();
}

double ClassStatistics::getPrecision() const{
    return (double)this->truePositives/ (double)(this->truePositives + this->falsePositives);
}

double ClassStatistics::getRecall() const{
    return  (double)this->truePositives/ (double)(this->truePositives + this->falseNegatives);
}

void ClassStatistics::printStats() const{
    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "  Class id: " << classID << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "TP: " << this->truePositives  << std::endl;
    std::cout << "FP: " << this->falsePositives << std::endl;
    std::cout << "FN: " << this->falseNegatives << std::endl;
    std::cout << "Mean IOU: " << getMeanIOU() << std::endl;
    std::cout << "Precision: " << getPrecision() << std::endl;
    std::cout << "Recall: " << getRecall() << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;
}


