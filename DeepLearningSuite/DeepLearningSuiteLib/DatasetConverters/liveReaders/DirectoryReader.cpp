//
// Created by frivas on 26/03/17.
//

#include "DirectoryReader.h"

DirectoryReader::DirectoryReader(const std::string& directoryPath):DatasetReader(true) {


    std::unordered_set<std::string> supported_image_formats ( {".bmp",".dib",".jpeg", ".jpg", ".jpe", ".jp2",
                                                                ".png", ".webp", ".pbm", ".pgm", ".ppm", ".sr",
                                                                 ".ras", ".tiff", ".tif"} );


    boost::filesystem::path boostDirectoryPath(directoryPath);

    if (boost::filesystem::exists(boostDirectoryPath)) {
        if (!boost::filesystem::is_directory(boostDirectoryPath)) {
            throw std::invalid_argument("Please Provide a folder containing all the images, not just a single file");
        }
    } else {
        throw std::invalid_argument("Provided Directory Path doesn't exist");
    }

    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator itr(boostDirectoryPath); itr!=end_itr; ++itr)  {
        if (boost::filesystem::is_directory(*itr))
            continue;

        if (supported_image_formats.find(boost::filesystem::extension(*itr)) != supported_image_formats.end()) {
            (this->listOfImages).push_back(itr->path().string());
        }


    }


}


bool DirectoryReader::getNextSample(Sample &sample) {

        cv::Mat image = cv::imread(this->listOfImages[this->sample_offset]);

        sample.setSampleID(std::to_string(++this->sample_offset));
        sample.setColorImage(image);
        return true;
}


int DirectoryReader::getNumberOfElements() {
    return this->listOfImages.size();
}
