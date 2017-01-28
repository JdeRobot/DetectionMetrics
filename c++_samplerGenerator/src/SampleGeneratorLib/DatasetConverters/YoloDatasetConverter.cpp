//
// Created by frivas on 22/01/17.
//

#include "YoloDatasetConverter.h"

YoloDatasetConverter::YoloDatasetConverter(const std::string &outPath, DatasetReader &reader):outPath(outPath),reader(reader) {

}

void YoloDatasetConverter::process(bool overWriteclassWithZero) {
    Sample sample;
    while (reader.getNetxSample(sample)){
        auto boundingBoxes = sample.getRectRegions().getRegions();
        for (auto it = boundingBoxes.begin(), end=boundingBoxes.end(); it != end; ++it){
            cv::Mat image = sample.getSampledColorImage();
            double x = it->region.x;
            double y = it->region.y;
            double w = it->region.width;
            double h = it->region.height;
            if ((w + x) > image.size().width){
                w =  image.size().width - 1 - x;
            }
            if ((h + y) > image.size().height){
                h = image.size().height - 1 - y;
            }

            int classId;
            if (overWriteclassWithZero)
                classId=0;
            else
                classId=it->id;
            std::stringstream boundSrt;
            boundSrt << classId <<" " <<  (it->region.x + w/2.0) / (double)image.size().width << " " << (it->region.y + h/2.0) / (double)image.size().height << " " << w / image.size().width << " " << h / image.size().height;
            std::cout << boundSrt.str() << std::endl;
        }
        cv::imshow("color", sample.getSampledColorImage());
        cv::waitKey(0);
    }
}

