//
// Created by frivas on 24/02/17.
//

#ifndef SAMPLERGENERATOR_JDEROBOTREADER_H
#define SAMPLERGENERATOR_JDEROBOTREADER_H


#include <jderobot/parallelIce/cameraClient.h>
#include <DatasetConverters/readers/DatasetReader.h>


class JderobotReader: public DatasetReader {
public:
    JderobotReader(const std::string& IceConfigFile);

    bool getNextSample(Sample &sample);
private:
    jderobot::CameraClientPtr camera;

};

typedef boost::shared_ptr<JderobotReader> JderobotReaderPtr;


#endif //SAMPLERGENERATOR_JDEROBOTREADER_H
