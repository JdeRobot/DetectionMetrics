//
// Created by frivas on 1/02/17.
//

#include <glog/logging.h>
#include <Utils/StatsUtils.h>
#include "DetectionsEvaluator.h"


DetectionsEvaluator::DetectionsEvaluator(DatasetReaderPtr gt, DatasetReaderPtr detections,bool debug):gt(gt),detections(detections),debug(debug) {

}

void DetectionsEvaluator::evaluate() {
    int counter=0;
    int gtSamples = this->gt->getNumberOfElements();
    int detectionSamples = this->detections->getNumberOfElements();

    if (gtSamples != detectionSamples){
        LOG(WARNING) << "Both dataset has not the same number of elements";
    }

    Sample gtSample;
    Sample detectionSample;

    while (this->gt->getNextSample(gtSample)){
        counter++;
        std::cout << "Evaluating: " << gtSample.getSampleID() << "(" << counter << "/" << gtSamples << ")" << std::endl;
        this->detections->getNextSample(detectionSample);
        if (gtSample.getSampleID().compare(detectionSample.getSampleID()) != 0){
            const std::string error="Both dataset has not the same structure ids mismatch from:" + gtSample.getSampleID() + " to " + detectionSample.getSampleID();
            LOG(WARNING) << error;
            throw error;
        }

        evaluateSamples(gtSample,detectionSample);
        printStats();

        if (this->debug){
            cv::imshow("GT", gtSample.getSampledColorImage());
            Sample detectionWithImage=detectionSample;
            detectionWithImage.setColorImage(gtSample.getColorImage());
            cv::imshow("Detection", detectionWithImage.getSampledColorImage());
            cv::waitKey(100);
        }

    }
}


void DetectionsEvaluator::evaluateSamples(Sample gt, Sample detection) {
    double thIOU = 0.5;

    auto detectionRegions = detection.getRectRegions()->getRegions();
    for (auto itDetection = detectionRegions.begin(), end = detectionRegions.end(); itDetection != end; ++itDetection) {
        bool matched = false;
        auto gtRegions = gt.getRectRegions()->getRegions();
        for (auto itGT = gtRegions.begin(), endGT = gtRegions.end(); itGT != endGT; ++itGT) {
            if (sameClass(itDetection->classID,itGT->classID )){
                double iouValue = StatsUtils::getIOU(itGT->region, itDetection->region, gt.getColorImage().size());
                if (iouValue > thIOU) {
                    if (itDetection->classID.compare(itGT->classID) ==0) {
                        addTruePositive(itDetection->classID);
                        addIOU(itDetection->classID, iouValue);
                    }
                    else{
                        addTruePositive(itGT->classID);
                        addIOU(itGT->classID,iouValue );
                    }
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            addFalsePositive(itDetection->classID);
        }

    }

    auto gtRegions = gt.getRectRegions()->getRegions();
    for (auto itGT = gtRegions.begin(), endGT = gtRegions.end(); itGT != endGT; ++itGT) {
        bool matched = false;
        for (auto itDetection = detectionRegions.begin(), end = detectionRegions.end();
             itDetection != end; ++itDetection) {
                if (sameClass(itGT->classID,itDetection->classID )){
                double iouValue = StatsUtils::getIOU(itGT->region, itDetection->region, gt.getColorImage().size());

                if (iouValue > thIOU) {
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            addFalseNegative(itGT->classID);
        }
    }

}

void DetectionsEvaluator::printStats() {
    if (classesToDisplay.size()==0) {
        for (auto it = this->statsMap.begin(), end = this->statsMap.end(); it != end; ++it) {
            it->second.printStats();
        }
    }
    else{
        for (auto it =this->classesToDisplay.begin(),end= this->classesToDisplay.end(); it != end; ++it){
            if (this->statsMap.count(*it))
                this->statsMap[*it].printStats();
        }
    }
}

bool DetectionsEvaluator::sameClass(const std::string class1, const std::string class2) {

    if (class1.compare(class2)==0)
        return true;
    else{
        if (std::find(validMixClass.begin(), validMixClass.end(), std::make_pair(class1,class2)) != validMixClass.end())
            return true;
        if (std::find(validMixClass.begin(), validMixClass.end(), std::make_pair(class2,class1)) != validMixClass.end())
            return true;
    }
    return false;
}

void DetectionsEvaluator::addTruePositive(const std::string &classID) {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].truePositives = this->statsMap[classID].truePositives+1;
    }
    else{
        ClassStatistics s(classID);
        s.truePositives = s.truePositives+1;
        this->statsMap[classID]=s;
    }
}

void DetectionsEvaluator::addFalsePositive(const std::string &classID) {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].falsePositives = this->statsMap[classID].falsePositives+1;
    }
    else{
        ClassStatistics s(classID);
        s.falsePositives = s.falsePositives+1;
        this->statsMap[classID]=s;
    }
}

void DetectionsEvaluator::addFalseNegative(const std::string &classID) {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].falseNegatives = this->statsMap[classID].falseNegatives+1;
    }
    else{
        ClassStatistics s(classID);
        s.falseNegatives = s.falseNegatives+1;
        this->statsMap[classID]=s;
    }
}




void DetectionsEvaluator::addIOU(const std::string &classID, double value) {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].iou.push_back(value);
    }
    else{
        ClassStatistics s(classID);
        s.iou.push_back(value);
        this->statsMap[classID]=s;
    }
}


void DetectionsEvaluator::addValidMixClass(const std::string classA, const std::string classB){
    //B is valid by detecting object as A
    this->validMixClass.push_back(std::make_pair(classA,classB));
}

void DetectionsEvaluator::addClassToDisplay(const std::string &classID) {
    this->classesToDisplay.push_back(classID);
}
