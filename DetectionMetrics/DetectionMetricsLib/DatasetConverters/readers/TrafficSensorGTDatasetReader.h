#ifndef SAMPLERGENERATOR_TRAFFICSENSORGTDATASETREADER_H
#define SAMPLERGENERATOR_TRAFFICSENSORGTDATASETREADER_H

#include <DatasetConverters/readers/DatasetReader.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

class TrafficSensorGTDatasetReader:public DatasetReader {
    public:
        TrafficSensorGTDatasetReader(const std::string& path, const std::string& classNamesFile, const bool imagesRequired);
        TrafficSensorGTDatasetReader(const bool imagesRequired);
        bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
    private:
};


typedef boost::shared_ptr<TrafficSensorGTDatasetReader> TrafficSensorGTDatasetReaderPtr;

#endif //SAMPLERGENERATOR_TRAFFICSENSORGTREADER_H
