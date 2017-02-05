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
#include <SampleGeneratorLib/DatasetConverters/OwnDatasetReader.h>
#include <SampleGeneratorLib/DatasetConverters/YoloDatasetWriter.h>
#include <SampleGeneratorLib/DatasetConverters/YoloDatasetReader.h>


#include <iostream>
#include <string>
#include <SampleGeneratorLib/Utils/SampleGenerationApp.h>
#include <SampleGeneratorLib/DatasetConverters/GenericDatasetReader.h>
#include <SampleGeneratorLib/FrameworkEvaluator/DetectionsEvaluator.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("readerImplementation");


    };
    void operator()(){
        Key inputPathKey=this->config.getKey("inputPath");
        Key readerImplementationKey = this->config.getKey("readerImplementation");


        GenericDatasetReaderPtr reader;
        if (inputPathKey.isVector()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathKey.getValues(), readerImplementationKey.getValue()));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathKey.getValue(), readerImplementationKey.getValue()));
        }



        Sample sample;
        while (reader->getReader()->getNetxSample(sample)){
            std::cout << "number of elements: " << sample.getRectRegions()->getRegions().size() << std::endl;
            cv::Mat image =sample.getSampledColorImage();
            cv::imshow("Viewer", image);
            cv::waitKey(0);
        }

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