//
// Created by frivas on 1/02/17.
//

#include <glog/logging.h>
#include <Utils/StatsUtils.h>
#include "DetectionsEvaluator.h"


DetectionsEvaluator::DetectionsEvaluator(DatasetReaderPtr gt, DatasetReaderPtr detections,bool debug):
        gt(gt),detections(detections),debug(debug) {
    thIOU = 0.5;

}

void DetectionsEvaluator::evaluate() {
    int counter=0;
    int gtSamples = this->gt->getNumberOfElements();
    int detectionSamples = this->detections->getNumberOfElements();

    Sample testSample;
    int count = 0;
    while (this->gt->getNextSample(testSample)){
        testSample.print();
        if (count == 500) {
            break;
        }
        count++;
    }

    /*while (this->detections->getNextSample(testSample)){
        testSample.print();
    }*/

    if (gtSamples != detectionSamples){
        LOG(WARNING) << "Both dataset has not the same number of elements";
    }

    ClassTypeMapper classMapper(this->gt->getClassNamesFile());
    this->classMapping = classMapper.mapFile(this->detections->getClassNamesFile());


    Sample gtSample;
    Sample detectionSample;
    Sample* gtSamplePtr;

    Eval::EvalMatrix evalmatrix;

    while (this->detections->getNextSample(detectionSample)){
        counter++;
        std::cout << "Evaluating: " << detectionSample.getSampleID() << "(" << counter << "/" << gtSamples << ")" << std::endl;


        //this->detections->getNextSample(detectionSample);
        if(!this->gt->getSampleBySampleID(&gtSamplePtr, detectionSample.getSampleID())) {
            std::cout << "Can't Find Sample" << '\n';
            continue;
        }

        if (gtSamplePtr->getSampleID().compare(detectionSample.getSampleID()) != 0){
            const std::string error="Both dataset has not the same structure ids mismatch from:" + gtSample.getSampleID() + " to " + detectionSample.getSampleID();
            LOG(WARNING) << error;
            throw error;
        }

        detectionSample.print();

        evaluateSample(*gtSamplePtr,detectionSample, evalmatrix);
        std::cout << "Size: " << gtSamplePtr->getColorImage().size() << '\n';
        //Eval::printMatrix(evalmatrix);
        //printStats();

        if (this->debug){
            cv::imshow("GT", gtSample.getSampledColorImage());
            Sample detectionWithImage=detectionSample;
            detectionWithImage.setColorImage(gtSample.getColorImage());
            cv::imshow("Detection", detectionWithImage.getSampledColorImage());
            cv::waitKey(100);
        }

    }



    cv::destroyAllWindows();
    std::cout << "Evaluated Successfully" << '\n';
}


void DetectionsEvaluator::evaluateSamples(Sample gt, Sample detection) {
    GlobalStats currentStats;

    auto detectionRegions = detection.getRectRegions()->getRegions();
    for (auto itDetection = detectionRegions.begin(), end = detectionRegions.end(); itDetection != end; ++itDetection) {
        std::string detectionClass = this->classMapping[itDetection->classID];
        if (detectionClass.empty())
            continue;

        bool matched = false;
        auto gtRegions = gt.getRectRegions()->getRegions();
        for (auto itGT = gtRegions.begin(), endGT = gtRegions.end(); itGT != endGT; ++itGT) {
            if (sameClass(detectionClass,itGT->classID )){
                double iouValue = StatsUtils::getIOU(itGT->region, itDetection->region);
                if (iouValue > thIOU) {
                    if (detectionClass.compare(itGT->classID) ==0) {
                        stats.addTruePositive(detectionClass);
                        currentStats.addTruePositive(detectionClass);
                        stats.addIOU(detectionClass, iouValue);
                        currentStats.addIOU(detectionClass, iouValue);
                    }
                    else{
                        stats.addTruePositive(itGT->classID);
                        currentStats.addTruePositive(itGT->classID);
                        stats.addIOU(itGT->classID,iouValue );
                        currentStats.addIOU(itGT->classID,iouValue );

                    }
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            stats.addFalsePositive(detectionClass);
            currentStats.addFalsePositive(detectionClass);
        }

    }

    auto gtRegions = gt.getRectRegions()->getRegions();
    for (auto itGT = gtRegions.begin(), endGT = gtRegions.end(); itGT != endGT; ++itGT) {
        bool matched = false;
        for (auto itDetection = detectionRegions.begin(), end = detectionRegions.end();
             itDetection != end; ++itDetection) {
                std::string detectionClass = this->classMapping[itDetection->classID];

                if (sameClass(itGT->classID,itDetection->classID )){
                double iouValue = StatsUtils::getIOU(itGT->region, itDetection->region);

                if (iouValue > thIOU) {
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            stats.addFalseNegative(itGT->classID);
            currentStats.addFalseNegative(itGT->classID);
        }
    }

}

void DetectionsEvaluator::evaluateSample(Sample gt, Sample detection, Eval::EvalMatrix& evalmatrix) {

    StatsUtils::computeIOUMatrix(gt, detection, evalmatrix);

    std::string sampleID = gt.getSampleID();

    auto detectionRegions = detection.getRectRegions()->getRegions();

    /*for (auto itDetection = detectionRegions.begin(); itDetection != detectionRegions.end(); itDetection++) {


    }*/
    for (int i = 0; i < 10; i++) {
        std::string current_class, previous_class;
        int count = 0;
        for (auto itDetection = detectionRegions.begin(); itDetection != detectionRegions.end(); itDetection++) {
            previous_class = current_class;
            current_class = itDetection->classID;
            if (!previous_class.empty()) {
                if (current_class != previous_class) {
                    count++;
                }
            }

            evalmatrix[sampleID][classID]

        }
    }

}

void DetectionsEvaluator::printStats() {
    this->stats.printStats(classesToDisplay);
}

GlobalStats DetectionsEvaluator::getStats() {
    return this->stats;
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


void DetectionsEvaluator::addValidMixClass(const std::string classA, const std::string classB){
    //B is valid by detecting object as A
    this->validMixClass.push_back(std::make_pair(classA,classB));
}

void DetectionsEvaluator::addClassToDisplay(const std::string &classID) {
    this->classesToDisplay.push_back(classID);
}
