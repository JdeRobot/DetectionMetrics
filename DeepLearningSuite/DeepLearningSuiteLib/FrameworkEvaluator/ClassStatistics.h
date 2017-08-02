//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_STATISTICS_H
#define SAMPLERGENERATOR_STATISTICS_H


#include <vector>
#include <string>

struct ClassStatistics {
    ClassStatistics();
    ClassStatistics(const std::string& classID);
    double getMeanIOU() const;
    double getPrecision() const;
    double getRecall() const;
    void printStats() const;


    std::string classID;
    std::vector<double> iou;
    int nSamples;
    int truePositives;
    int falsePositives;
    int falseNegatives;
    int trueNegatives; //???? evaluar muestra negativa??


};


#endif //SAMPLERGENERATOR_STATISTICS_H
