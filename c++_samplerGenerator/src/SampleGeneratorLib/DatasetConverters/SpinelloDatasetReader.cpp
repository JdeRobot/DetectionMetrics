//
// Created by frivas on 29/01/17.
//

#include <boost/filesystem/operations.hpp>
#include <fstream>
#include "SpinelloDatasetReader.h"
#include "ClassTypeVoc.h"
#include <Utils/StringHandler.h>
#include <Utils/Normalizations.h>

SpinelloDatasetReader::SpinelloDatasetReader(const std::string &path) {
    appendDataset(path);
}

SpinelloDatasetReader::SpinelloDatasetReader() {

}

bool SpinelloDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    auto boostPath= boost::filesystem::path(datasetPath + "/track_annotations/");



    boost::filesystem::directory_iterator end_itr;

    std::cout << "Path: " << boostPath.string() << std::endl;


    std::vector<std::string> labelFileNames;

    for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
    {
        if (is_regular_file(itr->status()) && itr->path().extension()==".txt") {
            labelFileNames.push_back(itr->path().string());
        }
    }

    std::sort(labelFileNames.begin(), labelFileNames.end());

    for (auto it = labelFileNames.begin(), end=labelFileNames.end(); it != end; ++it){
        std::ifstream labelFile(*it);
        std::string data;
        while(getline(labelFile,data)) {
            Sample sample;
            if (data[0] == '#')
                continue;
            std::vector<std::string> tokens = split(data, ' ');


            std::string imageID=tokens[0];

            for (auto it=tokens.begin(), end=tokens.end(); it != end; ++it){
            }

            std::string colorImagePath=datasetPath + "/"  + "rgb" + "/" + imageID + ".ppm";
            std::string depthImagePath=datasetPath + "/"  + "depth" + "/" + imageID + ".pgm";
            cv::Mat colorImage= cv::imread(colorImagePath);
            cv::Mat depthImage= cv::imread(depthImagePath);



            cv::Rect colorRect;
            cv::Rect depthRect;

            std::istringstream iss(tokens[2]);
            iss >> colorRect.x;
            iss=std::istringstream(tokens[3]);
            iss >> colorRect.y;
            iss=std::istringstream(tokens[4]);
            iss >> colorRect.width;
            iss=std::istringstream(tokens[5]);
            iss >> colorRect.height;
            iss=std::istringstream(tokens[6]);
            iss >> depthRect.x;
            iss=std::istringstream(tokens[7]);
            iss >> depthRect.y;
            iss=std::istringstream(tokens[8]);
            iss >> depthRect.width;
            iss=std::istringstream(tokens[9]);
            iss >> depthRect.height;

            Normalizations::normalizeRect(colorRect,colorImage.size());
            Normalizations::normalizeRect(depthRect,depthImage.size());

            Sample* samplePointer;
            if (this->getSampleBySampleID(&samplePointer,imageID)){
//                std::cout << "Image already exits" << std::endl;
                RectRegionsPtr regions=samplePointer->getRectRegions();
                regions->add(colorRect,"person");
                samplePointer->setRectRegions(regions);
            }
            else{
//                std::cout << "Image does not exits" << std::endl;
                sample.setSampleID(datasetPrefix + imageID);
                sample.setColorImage(colorImagePath);
                sample.setDepthImage(depthImagePath);
                RectRegionsPtr colorRegions(new RectRegions());
                colorRegions->add(colorRect,"person");
                sample.setRectRegions(colorRegions);
                samples.push_back(sample);
            }
        }
    }
    printDatasetStats();
}
