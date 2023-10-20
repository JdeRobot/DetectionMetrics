#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <boost/lexical_cast.hpp>
#include "DatasetConverters/readers/TrafficSensorGTDatasetReader.h"

TrafficSensorGTDatasetReader::TrafficSensorGTDatasetReader(const std::string &path,const std::string& classNamesFile,const bool imagesRequired):DatasetReader(imagesRequired){
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}

TrafficSensorGTDatasetReader::TrafficSensorGTDatasetReader(const bool imagesRequired):DatasetReader(imagesRequired) {}

bool TrafficSensorGTDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    boost::filesystem::directory_iterator end_itr;
    boost::filesystem::path boostPath(datasetPath);
    std::vector<std::string> filesID;


    for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
    {
        if ((is_regular_file(itr->status()) && itr->path().extension()==".xml") && (itr->path().string().find("-region") == std::string::npos)) {
            filesID.push_back(itr->path().string());
        }
    }

    std::sort(filesID.begin(),filesID.end());

    for (auto it = filesID.begin(), end=filesID.end(); it != end; ++it) {

        Sample sample;
        sample.setSampleID(datasetPrefix + boost::filesystem::path(*it).filename().stem().string());

        boost::property_tree::ptree tree;
        boost::property_tree::read_xml(boost::filesystem::path(*it).string(), tree);

        RectRegionsPtr rectRegions(new RectRegions());

        std::string m_folder = tree.get<std::string>("annotation.folder");
        std::string m_filename = tree.get<std::string>("annotation.filename");
        std::string m_path = tree.get<std::string>("annotation.path");

        BOOST_FOREACH(boost::property_tree::ptree::value_type &v, tree.get_child("annotation")) {
            // The data function is used to access the data stored in a node.
            if (v.first == "object") {
                std::string object_name = v.second.get<std::string>("name");
                int xmin = v.second.get<int>("bndbox.xmin");
                int xmax = v.second.get<int>("bndbox.xmax");
                int ymin = v.second.get<int>("bndbox.ymin");
                int ymax = v.second.get<int>("bndbox.ymax");

                cv::Rect bounding(xmin, ymin, xmax - xmin, ymax - ymin);
                rectRegions->add(bounding, object_name);
                sample.setRectRegions(rectRegions);
            }
        }

        this->samples.push_back(sample);
    }

    LOG(INFO) << "Loaded: " + boost::lexical_cast<std::string>(this->samples.size()) + " samples";
    printDatasetStats();
    return true;
}