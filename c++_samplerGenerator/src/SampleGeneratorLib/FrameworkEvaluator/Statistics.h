//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_STATISTICS_H
#define SAMPLERGENERATOR_STATISTICS_H


#include <vector>

class Statistics {
public:
    Statistics();
private:
    std::vector<double> iou;
    int nSamples;
    int truePositives;
    int falsePositives;
    int falseNegatives;
    int trueNegatives; //???? evaluar muestra negativa??
};


#endif //SAMPLERGENERATOR_STATISTICS_H
