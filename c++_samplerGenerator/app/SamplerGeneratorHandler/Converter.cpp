//
// Created by frivas on 19/02/17.
//

#include <SampleGeneratorLib/DatasetConverters/GenericDatasetWriter.h>
#include <gui/Utils.h>
#include "Converter.h"
#include "SamplerGenerationHandler.h"

void SampleGeneratorHandler::Converter::process(QListView *datasetList, QListView *namesList, QListView *readerImpList,
                                                QListView *filterClasses, QListView *writerImpList,
                                                const std::string& datasetPath, const std::string& namesPath, const std::string &outputPath,
                                                bool splitActive, double splitRatio,bool useColorImage) {


    GenericDatasetReaderPtr reader = SamplerGenerationHandler::createReaderPtr(datasetList,namesList,readerImpList,filterClasses,datasetPath,namesPath);
    std::vector<std::string> writerImp;
    Utils::getListViewContent(writerImpList,writerImp,"");


    if (splitActive){
        DatasetReaderPtr readerTest(new DatasetReader());
        DatasetReaderPtr readerTrain(new DatasetReader());

        std::string testPath = outputPath + "/test";
        std::string trainPath = outputPath + "/train";


        int ratio=int(splitRatio*10);

        Sample sample;
        auto readerPtr = reader->getReader();
        int counter=0;
        while (readerPtr->getNetxSample(sample)){
            if (counter <ratio){
                readerTrain->addSample(sample);
            }
            else{
                readerTest->addSample(sample);
            }
            counter++;
            counter= counter % 10;
        }

        std::cout << "Train: " << std::endl;
        readerTrain->printDatasetStats();
        std::cout << "Test: " << std::endl;
        readerTest->printDatasetStats();


        GenericDatasetWriterPtr writerTest( new GenericDatasetWriter(testPath,readerTest,writerImp[0]));
        writerTest->getWriter()->process(useColorImage);

        GenericDatasetWriterPtr writerTrain( new GenericDatasetWriter(trainPath,readerTrain,writerImp[0]));
        writerTrain->getWriter()->process(useColorImage);
    }
    else{
        auto readerPtr = reader->getReader();
        GenericDatasetWriterPtr writer( new GenericDatasetWriter(outputPath,readerPtr,writerImp[0]));
        writer->getWriter()->process(useColorImage);
    }


}
