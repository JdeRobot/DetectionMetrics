//
// Created by frivas on 24/02/17.
//

#include <Ice/CommunicatorF.h>
#include "JderobotReader.h"
#include <jderobot/easyiceconfig/EasyIce.h>

JderobotReader::JderobotReader(const std::string &IceConfigFile) {
    Ice::CommunicatorPtr ic;

    //todo hack
    int argc=2;
    char* argv[2];
    argv[0] = (char*)std::string("myFakeApp").c_str();
    argv[1] = (char*)IceConfigFile.c_str();


    ic = EasyIce::initialize(argc,argv);
    Ice::ObjectPrx base = ic->propertyToProxy("Cameraview.Camera.Proxy");
    Ice::PropertiesPtr prop = ic->getProperties();

    if (0==base)
        throw "Could not create proxy";


    this->camera = jderobot::CameraClientPtr (new jderobot::cameraClient(ic,"Cameraview.Camera."));

    if (! this->camera){
        throw "Invalid proxy";
    }
    this->camera->start();

}

bool JderobotReader::getNextSample(Sample &sample) {
    cv::Mat image;
    this->camera->getImage(image);
    if (!image.empty()){
        sample.setColorImage(image);
        sample.setDepthImage(image);
    }


}
