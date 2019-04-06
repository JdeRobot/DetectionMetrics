#include <iostream>
#include "RecorderReader.h"
#include <boost/filesystem.hpp>

int main()
{
    const std::string colorImagePath = boost::filesystem::current_path().string()+"/color_images/";
    const std::string depthImagePath = boost::filesystem::current_path().string()+"/depth_images/";
    RecorderReader r(colorImagePath,depthImagePath);
    std::cout<<r.getNumSamples()<<std::endl;
    return 0;
}
