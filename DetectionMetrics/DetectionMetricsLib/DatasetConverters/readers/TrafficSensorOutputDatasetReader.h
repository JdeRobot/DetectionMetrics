#ifndef SAMPLERGENERATOR_TRAFFICSENSOROUTPUTDATASETREADER_H
#define SAMPLERGENERATOR_TRAFFICSENSOROUTPUTDATASETREADER_H

#include <DatasetConverters/readers/DatasetReader.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

class TrafficSensorOutputDatasetReader:public DatasetReader {
    public:
        TrafficSensorOutputDatasetReader(const std::string& path, const std::string& classNamesFile, const bool imagesRequired);
        TrafficSensorOutputDatasetReader(const bool imagesRequired);
        bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
    private:
};


typedef boost::shared_ptr<TrafficSensorOutputDatasetReader> TrafficSensorOutputDatasetReaderPtr;

#endif //SAMPLERGENERATOR_TRAFFICSENSOROUTPUTREADER_H
