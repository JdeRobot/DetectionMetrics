//
// Created by frivas on 1/02/17.
//

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <Logger.h>
#include "MassInferencer.h"

MassInferencer::MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer,
                               const std::string &resultsPath,bool debug): reader(reader), inferencer(inferencer), resultsPath(resultsPath),debug(debug)
{
    auto boostPath= boost::filesystem::path(this->resultsPath);
    if (!boost::filesystem::exists(boostPath)){
        boost::filesystem::create_directories(boostPath);
    }
    else{
        Logger::getInstance()->error("Output directory already exists");
        //exit(-1);
    }
}

void MassInferencer::process() {


    Sample sample;
    int counter=0;
    int nsamples = this->reader->getNumberOfElements();
    while (this->reader->getNetxSample(sample)){
        counter++;
        std::cout << "Evaluating: " << sample.getSampleID() << "(" << counter << "/" << nsamples << ")" << std::endl;
        cv::Mat image =sample.getSampledColorImage();
        Sample detection=this->inferencer->detect(sample.getColorImage());
        detection.setSampleID(sample.getSampleID());
        detection.save(this->resultsPath);
        if (this->debug) {
            cv::imshow("Viewer", image);
            cv::imshow("Detection", detection.getSampledColorImage());
            cv::waitKey(100);
        }
    }
}
