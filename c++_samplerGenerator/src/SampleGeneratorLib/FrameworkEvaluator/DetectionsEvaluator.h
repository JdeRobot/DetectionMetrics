//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_DETECTIONSEVALUATOR_H
#define SAMPLERGENERATOR_DETECTIONSEVALUATOR_H

#include <DatasetConverters/DatasetReader.h>
#include <boost/shared_ptr.hpp>
#include "ClassStatistics.h"

class DetectionsEvaluator {
public:
    DetectionsEvaluator(DatasetReaderPtr gt, DatasetReaderPtr detections, bool debug=true);
    void evaluate();
    void addValidMixClass(const std::string classA, const std::string classB);
    void addClassToDisplay(const std::string& classID);

private:
    DatasetReaderPtr gt;
    DatasetReaderPtr detections;
    bool debug;
    std::map<std::string,ClassStatistics> statsMap;
    std::vector<std::pair<std::string, std::string>> validMixClass;

    void evaluateSamples(Sample gt, Sample detection);
    void printStats();

    bool sameClass(const std::string class1, const std::string class2);

    void addTruePositive(const std::string& classID);
    void addFalsePositive(const std::string& classID);
    void addFalseNegative(const std::string& classID);
    void addIOU(const std::string& classID, double value);

    std::vector<std::string> classesToDisplay;
};


typedef boost::shared_ptr<DetectionsEvaluator> DetectionsEvaluatorPtr;


#endif //SAMPLERGENERATOR_DETECTIONSEVALUATOR_H
