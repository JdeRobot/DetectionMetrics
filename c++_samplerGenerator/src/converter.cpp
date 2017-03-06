//
// Created by frivas on 21/01/17.
//


#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <SampleGeneratorLib/Utils/Logger.h>
#include <highgui.h>
#include <SampleGeneratorLib/DatasetConverters/readers/OwnDatasetReader.h>
#include <SampleGeneratorLib/DatasetConverters/writers/YoloDatasetWriter.h>
#include <SampleGeneratorLib/DatasetConverters/readers/YoloDatasetReader.h>


#include <iostream>
#include <string>
#include <SampleGeneratorLib/Utils/SampleGenerationApp.h>
#include <SampleGeneratorLib/DatasetConverters/readers/GenericDatasetReader.h>
#include <SampleGeneratorLib/DatasetConverters/writers/GenericDatasetWriter.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("writerImplementation");
        this->requiredArguments.push_back("readerNames");


    };
    void operator()(){
        Key inputPathKey=this->config->getKey("inputPath");
        Key readerImplementationKey = this->config->getKey("readerImplementation");
        Key writerImplementationKey = this->config->getKey("writerImplementation");
        Key outputPathKey = this->config->getKey("outputPath");
        Key readerNamesKey = this->config->getKey("readerNames");


        GenericDatasetReaderPtr reader;
        if (inputPathKey.isVector()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathKey.getValues(), readerNamesKey.getValue(), readerImplementationKey.getValue()));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathKey.getValue(),readerNamesKey.getValue(), readerImplementationKey.getValue()));
        }

        auto readerPtr = reader->getReader();

//        std::vector<std::string> idsToFilter;
//        idsToFilter.push_back("person");
//        idsToFilter.push_back("person-falling");
//        idsToFilter.push_back("person-fall");
//        readerPtr->filterSamplesByID(idsToFilter);
//        readerPtr->printDatasetStats();


        GenericDatasetWriterPtr writer( new GenericDatasetWriter(outputPathKey.getValue(),readerPtr,writerImplementationKey.getValue()));
        writer->getWriter()->process();
    };
};

int main (int argc, char* argv[])
{

    MyApp myApp(argc,argv);
    myApp.process();
}

/*void extractPersonsFromYolo(const std::string& dataSetPath){
    YoloDatasetReader reader(dataSetPath);

    std::vector<std::string> idsToFilter;
    idsToFilter.push_back("person");


    std::cout << "Samples before: " << reader.getNumberOfElements() << std::endl;
    reader.filterSamplesByID(idsToFilter);
    std::cout << "Samples after: " << reader.getNumberOfElements() << std::endl;
    YoloDatasetWriter converter("converter_output", reader);
    converter.process(true);
}



int main (int argc, char* argv[]) {

    ViewerAguments args;
    parse_arguments(argc,argv,args);


    Logger::getInstance()->setLevel(Logger::INFO);
    Logger::getInstance()->info("Reviewing " + args.path);


    extractPersonsFromYolo(args.path);

    OwnDatasetReader reader(args.path);

    YoloDatasetWriter converter("converter_output", reader);
    converter.process(true);
}
*/