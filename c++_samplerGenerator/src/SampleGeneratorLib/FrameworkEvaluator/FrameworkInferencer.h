//
// Created by frivas on 29/01/17.
//

#ifndef SAMPLERGENERATOR_FRAMEWORKEVALUATOR_H
#define SAMPLERGENERATOR_FRAMEWORKEVALUATOR_H

#include <cv.h>
#include <boost/shared_ptr.hpp>


class FrameworkInferencer{
public:
    virtual Sample detect(const cv::Mat& image) =0;
};


typedef boost::shared_ptr<FrameworkInferencer> FrameworkInferencerPtr;


#endif //SAMPLERGENERATOR_FRAMEWORKEVALUATOR_H
