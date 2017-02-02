//
// Created by frivas on 29/01/17.
//

#include <boost/filesystem/operations.hpp>
#include <fstream>
#include "SpinelloDatasetReader.h"
#include "ClassTypeVoc.h"
#include <Utils/StringHandler.h>
#include <Utils/Normalizations.h>

SpinelloDatasetReader::SpinelloDatasetReader(const std::string &path):path(path) {

    auto boostPath= boost::filesystem::path(this->path + "/track_annotations/");



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
            std::cout << "----------------------------------" << std::endl;
            if (data[0] == '#')
                continue;
            std::vector<std::string> tokens = split(data, ' ');


            std::string imageID=tokens[0];

            for (auto it=tokens.begin(), end=tokens.end(); it != end; ++it){
                std::cout << "aaa: " <<  *it << std::endl;
            }

            std::string colorImagePath=this->path + "/"  + "rgb" + "/" + imageID + ".ppm";
            std::string depthImagePath=this->path + "/"  + "depth" + "/" + imageID + ".pgm";


            std::cout << "Image path: " << colorImagePath << std::endl;
            std::cout << "Image path: " << depthImagePath << std::endl;

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
                std::cout << "Image already exits" << std::endl;
                RectRegions regions=samplePointer->getRectRegions();
                regions.add(colorRect,PERSON);
                samplePointer->setRectRegions(regions);
            }
            else{
                std::cout << "Image does not exits" << std::endl;
                sample.setSampleID(imageID);
                sample.setColorImage(colorImagePath);
                sample.setDepthImage(depthImagePath);
                RectRegions colorRegions;
                colorRegions.add(colorRect,PERSON);
                sample.setRectRegions(colorRegions);
                samples.push_back(sample);
            }



            std::cout << "Number of samples: " << this->getNumberOfElements() << std::endl;

//            std::istringstream iss(data);
//            std::cout << "DAta: " << data << std::endl;
//            std::string n1, imageName;
//            double timeStamp, X_tl_dpt, Y_tl_dpt, WDT_dpt, HGT_dpt, X_tl_rgb, Y_tl_rgb, WDT_rgb, HGT_rgb;
//            int VSB;
//            iss >> n1 >> imageName >> timeStamp >> X_tl_dpt >> Y_tl_dpt >> WDT_dpt >> HGT_dpt >> X_tl_rgb >> Y_tl_rgb >> WDT_rgb >> HGT_rgb >> VSB;
//            std::cout << "Image path: " << this->path + "/"  + "rgb" + "/" + imageName + ".ppm";
        }
    }
}
