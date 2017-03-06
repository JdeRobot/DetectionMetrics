//
// Created by frivas on 22/01/17.
//

#include "YoloDatasetWriter.h"
#include "DatasetConverters/ClassTypeOwn.h"
#include <iomanip>
#include <fstream>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <Utils/Logger.h>


YoloDatasetWriter::YoloDatasetWriter(const std::string &outPath, DatasetReaderPtr &reader,bool overWriteclassWithZero):DatasetWriter(outPath,reader),overWriteclassWithZero(overWriteclassWithZero){

    this->fullImagesPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/JPEGImages")).string();
    this->fullLabelsPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/labels")).string();


    auto boostImages= boost::filesystem::path(fullImagesPath);
    if (!boost::filesystem::exists(boostImages)){
        boost::filesystem::create_directories(boostImages);
    }
    auto boostLabels= boost::filesystem::path(fullLabelsPath);
    if (!boost::filesystem::exists(boostLabels)){
        boost::filesystem::create_directories(boostLabels);
    }


    std::cout << "FullImagesPath: " << this->fullImagesPath << std::endl;
    std::cout << "FullLabelsPath: " << this->fullImagesPath << std::endl;

}

void YoloDatasetWriter::process(bool usedColorImage) {
    Sample sample;
    int id=0;

    std::ofstream sampleFile(this->outPath + "/sample.txt");

    while (reader->getNextSample(sample)){
        auto boundingBoxes = sample.getRectRegions()->getRegions();
        std::stringstream ssID ;
        ssID << std::setfill('0') << std::setw(5) << id;
        std::string imageFilePath= this->fullImagesPath + "/" + ssID.str() + ".jpg";
        sampleFile << imageFilePath << std::endl;

        std::string labelFilePath= this->fullLabelsPath + "/" + ssID.str() + ".txt";
        std::ofstream out(labelFilePath);

        cv::Mat image;
        if (usedColorImage)
            image= sample.getColorImage();
        else {
            image = sample.getDepthColorMapImage();

        }


        for (auto it = boundingBoxes.begin(), end=boundingBoxes.end(); it != end; ++it){
            double x = it->region.x;
            double y = it->region.y;
            double w = it->region.width;
            double h = it->region.height;
            if ((w + x) > image.size().width){
                w =  image.size().width - 1 - x;
            }
            if ((h + y) > image.size().height){
                h = image.size().height - 1 - y;
            }

            int classId;
            if (overWriteclassWithZero)
                classId=0;
            else {
                ClassTypeOwn typeConverter(it->classID);
                classId = typeConverter.getClassID();
            }
            std::stringstream boundSrt;
            boundSrt << classId <<" " <<  (it->region.x + w/2.0) / (double)image.size().width << " " << (it->region.y + h/2.0) / (double)image.size().height << " " << w / image.size().width << " " << h / image.size().height;
//            std::cout << boundSrt.str() << std::endl;
            out << boundSrt.str() << std::endl;
        }
        out.close();
        id++;
        cv::imwrite(imageFilePath,image);
    }
    sampleFile.close();
}

