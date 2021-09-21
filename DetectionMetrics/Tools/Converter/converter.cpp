//
// Created by frivas on 21/01/17.
//


#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <glog/logging.h>
#include <DatasetConverters/readers/OwnDatasetReader.h>
#include <DatasetConverters/writers/YoloDatasetWriter.h>
#include <DatasetConverters/readers/YoloDatasetReader.h>


#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <DatasetConverters/writers/GenericDatasetWriter.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("writerImplementation");
        this->requiredArguments.push_back("readerNames");
        this->requiredArguments.push_back("writeImages");
    };

    void operator()(){
        YAML::Node inputPathNode=this->config.getNode("inputPath");
        YAML::Node readerImplementationNode = this->config.getNode("readerImplementation");
        YAML::Node writerImplementationNode = this->config.getNode("writerImplementation");
        YAML::Node outputPathNode = this->config.getNode("outputPath");
        YAML::Node readerNamesNode = this->config.getNode("readerNames");
        YAML::Node writeImages = this->config.getNode("writeImages");

        GenericDatasetReaderPtr reader;
        if (inputPathNode.IsSequence()) {
            reader = GenericDatasetReaderPtr(new GenericDatasetReader(inputPathNode.as<std::vector<std::string>>(), readerNamesNode.as<std::string>(), readerImplementationNode.as<std::string>(), writeImages.as<bool>()));
        } else {
            reader = GenericDatasetReaderPtr(new GenericDatasetReader(inputPathNode.as<std::string>(),readerNamesNode.as<std::string>(), readerImplementationNode.as<std::string>(), writeImages.as<bool>()));
        }

        auto readerPtr = reader->getReader();

        GenericDatasetWriterPtr writer( new GenericDatasetWriter(outputPathNode.as<std::string>(),readerPtr,writerImplementationNode.as<std::string>()));
        writer->getWriter()->process(writeImages.as<bool>());
    };
};

int main (int argc, char* argv[])
{
    MyApp myApp(argc,argv);
    myApp.process();
}

