//
// Created by frivas on 18/02/17.
//

#include <iostream>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <Common/Sample.h>
#include "Viewer.h"
#include "SamplerGenerationHandler.h"
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace SampleGeneratorHandler {

    void Viewer::process(QListView* datasetList,QListView* namesList,QListView* readerImpList,QListView* filterClasses, const std::string& datasetPath, const std::string& namesPath) {

        GenericDatasetReaderPtr reader = SamplerGenerationHandler::createDatasetReaderPtr(datasetList, namesList,
                                                                                          readerImpList, filterClasses,
                                                                                          datasetPath, namesPath);
        if (!reader){
            return;
        }

        std::string windowName="viewer";
        Sample sample;
        while (reader->getReader()->getNextSample(sample)){
            std::cout << "number of elements: " << sample.getRectRegions()->getRegions().size() << std::endl;
            cv::Mat image =sample.getSampledColorImage();
            cv::imshow(windowName, image);
            int key = cv::waitKey(0);
            if (char(key) == 'q'){
                break;
            }
        }
        cv::destroyWindow(windowName);

    }

}