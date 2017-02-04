//
// Created by frivas on 1/02/17.
//

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <Utils/Logger.h>
#include "MassInferencer.h"

MassInferencer::MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer,
                               const std::string &resultsPath,bool debug): reader(reader), inferencer(inferencer), resultsPath(resultsPath),debug(debug)
{
    alreadyProcessed=0;
    auto boostPath= boost::filesystem::path(this->resultsPath);
    if (!boost::filesystem::exists(boostPath)){
        boost::filesystem::create_directories(boostPath);
    }
    else{
        Logger::getInstance()->error("Output directory already exists");
        Logger::getInstance()->error("Continuing detecting");
        boost::filesystem::directory_iterator end_itr;

        for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
        {
            if ((is_regular_file(itr->status()) && itr->path().extension()==".png") && (itr->path().string().find("-depth") == std::string::npos)) {
                alreadyProcessed++;
            }

        }
        //exit(-1);
    }
}

void MassInferencer::process() {

    Sample sample;
    int counter=0;
    int nsamples = this->reader->getNumberOfElements();
    while (alreadyProcessed>0){
        std::cout << "Already evaluated: " << sample.getSampleID() << "(" << counter << "/" << nsamples << ")" << std::endl;
        this->reader->getNetxSample(sample);
        counter++;
        alreadyProcessed--;
    }


    while (this->reader->getNetxSample(sample)){
        counter++;
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
