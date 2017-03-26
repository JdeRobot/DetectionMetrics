//
// Created by frivas on 24/02/17.
//


#include "FrameworkInferencer.h"



Sample FrameworkInferencer::detect(const cv::Mat &image) {
    boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::local_time();
    Sample s = detectImp(image);
    boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration duration = endTime-startTime;
    long elapsedTime = duration.total_milliseconds();
    durationVector.push_back(elapsedTime);
    return s;
}

int FrameworkInferencer::getMeanDurationTime() {
    int accumulate=0;
    for (auto it = this->durationVector.begin(), end = this->durationVector.end(); it != end; ++it)
        accumulate+=(int)(*it);
    return accumulate/(int)this->durationVector.size();
}

FrameworkInferencer::FrameworkInferencer() {

}

FrameworkInferencer::~FrameworkInferencer() {
    std::cout << "Mean inference time: " << this->getMeanDurationTime() << "(ms)" <<  std::endl;
}
