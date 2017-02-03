//
// Created by frivas on 1/02/17.
//

#include <Logger.h>
#include "DetectionsEvaluator.h"

DetectionsEvaluator::DetectionsEvaluator(DatasetReaderPtr gt, DatasetReaderPtr detections,bool debug):gt(gt),detections(detections),debug(debug) {
    nSamples=0;
    truePositives=0;
    falsePositives=0;
    falseNegatives=0;
    trueNegatives=0;
}

void DetectionsEvaluator::evaluate() {
    int counter=0;
    int gtSamples = this->gt->getNumberOfElements();
    int detectionSamples = this->detections->getNumberOfElements();

    if (gtSamples != detectionSamples){
        Logger::getInstance()->error("Both dataset has not the same number of elements");
        //exit(1);
    }

    Sample gtSample;
    Sample detectionSample;

    while (this->gt->getNetxSample(gtSample)){
        counter++;
        std::cout << "Evaluating: " << gtSample.getSampleID() << "(" << counter << "/" << gtSamples << ")" << std::endl;
        this->detections->getNetxSample(detectionSample);
        if (gtSample.getSampleID().compare(detectionSample.getSampleID()) != 0){
            Logger::getInstance()->error("Both dataset has not the same structure ids mismatch");
            exit(1);
        }

        evaluateSamples(gtSample,detectionSample);
        printStats();

        if (this->debug){
            cv::imshow("GT", gtSample.getSampledColorImage());
            cv::imshow("Detection", detectionSample.getSampledColorImage());
            cv::waitKey(100);
        }

    }
}

void DetectionsEvaluator::evaluateSamples(Sample gt, Sample detection) {
    double thIOU = 0.5;

    auto detectionRegions = detection.getRectRegions().getRegions();
    for (auto itDetection = detectionRegions.begin(), end = detectionRegions.end(); itDetection != end; ++itDetection) {
        if (itDetection->classID.compare("person")== 0) {
            bool matched = false;
            auto gtRegions = gt.getRectRegions().getRegions();
            for (auto itGT = gtRegions.begin(), endGT = gtRegions.end(); itGT != endGT; ++itGT) {
                if (itDetection->classID.compare("person")== 0 && itDetection->classID.compare(itGT->classID) == 0) {

                    double iouValue = getIOU(itGT->region, itDetection->region, gt.getColorImage().size());

                    if (iouValue > thIOU) {
                        truePositives++;
                        this->iou.push_back(iouValue);
                        matched = true;
                        break;
                    }
//                std::cout << "Intersection: " << interSectionArea << std::endl;
//                std::cout << "Union: " << unionArea << std::endl;
//                std::cout << "IOU: " << iouValue << std::endl;
//                cv::imshow("maskGT",maskGT);
//                cv::imshow("maskDetection",maskDetection);
//                cv::imshow("unionMask",unionMask);
//                cv::imshow("interSection",interSection);
//                cv::waitKey(0);
                }
            }
            if (!matched) {
                this->falsePositives++;
            }
        }
    }

    auto gtRegions = gt.getRectRegions().getRegions();
    for (auto itGT = gtRegions.begin(), endGT = gtRegions.end(); itGT != endGT; ++itGT) {
        if (itGT->classID.compare("person")== 0) {
            bool matched = false;
            for (auto itDetection = detectionRegions.begin(), end = detectionRegions.end();
                 itDetection != end; ++itDetection) {
                if (itGT->classID.compare("person")== 0 && itDetection->classID.compare(itGT->classID) == 0) {

                    double iouValue = getIOU(itGT->region, itDetection->region, gt.getColorImage().size());

                    if (iouValue > thIOU) {
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) {
                this->falseNegatives++;
            }
        }
    }
}

void DetectionsEvaluator::printStats() {
    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "TP: " << this->truePositives  << std::endl;
    std::cout << "FP: " << this->falsePositives << std::endl;
    std::cout << "FN: " << this->falseNegatives << std::endl;
    double average = std::accumulate( this->iou.begin(), this->iou.end(), 0.0)/this->iou.size();
    std::cout << "Mean IOU: " << average << std::endl;
    double precision = (double)this->truePositives/ (double)(this->truePositives + this->falsePositives);
    std::cout << "Precision: " <<precision << std::endl;
    double recall = (double)this->truePositives/ (double)(this->truePositives + this->falseNegatives);
    std::cout << "Recall: " <<recall << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;

}

double DetectionsEvaluator::getIOU(const cv::Rect &gt, const cv::Rect &detection, const cv::Size& imageSize) {
    //compute iou
    cv::Mat maskGT(imageSize, CV_8UC1, cv::Scalar(0));
    cv::Mat maskDetection(imageSize, CV_8UC1, cv::Scalar(0));

    cv::rectangle(maskGT, gt, cv::Scalar(255), -1);
    cv::rectangle(maskDetection, detection, cv::Scalar(255), -1);

    cv::Mat unionMask(imageSize, CV_8UC1, cv::Scalar(0));
    cv::rectangle(unionMask, gt, cv::Scalar(150), -1);
    cv::rectangle(unionMask, detection, cv::Scalar(255), -1);

    cv::Mat interSection = maskGT & maskDetection;

    int interSectionArea = cv::countNonZero(interSection);
    int unionArea = cv::countNonZero(unionMask);
    double iouValue = double(interSectionArea) / double(unionArea);
    return iouValue;
}
