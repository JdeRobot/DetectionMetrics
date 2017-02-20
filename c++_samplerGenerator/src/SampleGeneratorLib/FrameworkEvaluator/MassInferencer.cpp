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
        Logger::getInstance()->warning("Output directory already exists");
        Logger::getInstance()->warning("Continuing detecting");
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

void MassInferencer::process(bool useDepthImages) {

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
        std::cout << "Evaluating : " << sample.getSampleID() << "(" << counter << "/" << nsamples << ")" << std::endl;
        counter++;
        cv::Mat image =sample.getSampledColorImage();
        cv::Mat image2detect;
        if (useDepthImages)
            image2detect = sample.getColorImage();
        else
            image2detect=sample.getDepthImage();

        Sample detection=this->inferencer->detect(image2detect);
        detection.setSampleID(sample.getSampleID());
        detection.save(this->resultsPath);
        if (this->debug) {
            Sample detectionWithImage;
            detectionWithImage=detection;
            if (useDepthImages)
                detectionWithImage.setColorImage(sample.getDepthImage());
            else
                detectionWithImage.setColorImage(sample.getColorImage());
            cv::imshow("Viewer", image);
            cv::imshow("Detection", detectionWithImage.getSampledColorImage());
            cv::waitKey(100);
        }
    }
}
