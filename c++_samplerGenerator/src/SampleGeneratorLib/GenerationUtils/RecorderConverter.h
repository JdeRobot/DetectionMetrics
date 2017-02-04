// Created by frivas on 16/11/16.
//

#ifndef SAMPLERGENERATOR_RECORDERCONVERTER_H
#define SAMPLERGENERATOR_RECORDERCONVERTER_H

#include <string>
#include <vector>


class RecorderConverter {
public:
    RecorderConverter(const std::string& colorImagesPath, const std::string& depthImagesPath);
    bool getNext(std::string& colorImagePath, std::string& depthImagePath);
    int getNumSamples();

private:
    const  std::string depthPath;
    const std::string colorPath;
    int currentIndex;
    std::vector<int> depthIndexes;
    std::vector<int> colorIndexes;

    void getImagesByIndexes(const std::string& path, std::vector<int>& indexes);
    std::string getPathByIndex(const std::string& path, const int id);
    int closest(std::vector<int> const& vec, int value);


        };


#endif //SAMPLERGENERATOR_RECORDERCONVERTER_H
