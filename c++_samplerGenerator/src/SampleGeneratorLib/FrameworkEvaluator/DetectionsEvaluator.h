//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_DETECTIONSEVALUATOR_H
#define SAMPLERGENERATOR_DETECTIONSEVALUATOR_H

#include <DatasetConverters/DatasetReader.h>
#include <boost/shared_ptr.hpp>

class DetectionsEvaluator {
public:
    DetectionsEvaluator(DatasetReaderPtr gt, DatasetReaderPtr detections, bool debug=true);
    void evaluate();

private:
    DatasetReaderPtr gt;
    DatasetReaderPtr detections;
    bool debug;

    std::vector<double> iou;
    int nSamples;
    int truePositives;
    int falsePositives;
    int falseNegatives;
    int trueNegatives;


    void evaluateSamples(Sample gt, Sample detection);
    void printStats();
    double getIOU(const cv::Rect& gt, const cv::Rect& detection,const cv::Size& imageSize);
};


typedef boost::shared_ptr<DetectionsEvaluator> DetectionsEvaluatorPtr;


#endif //SAMPLERGENERATOR_DETECTIONSEVALUATOR_H
