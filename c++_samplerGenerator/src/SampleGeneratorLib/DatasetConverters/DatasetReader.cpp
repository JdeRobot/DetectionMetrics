//
// Created by frivas on 22/01/17.
//

#include "DatasetReader.h"


DatasetReader::DatasetReader():readerCounter(0) {

}

void DatasetReader::filterSamplesByID(std::vector<int> filteredIDS) {
    std::vector<Sample> old_samples(this->samples);
    this->samples.clear();

    for (auto it=old_samples.begin(), end=old_samples.end(); it != end; ++it){
        Sample& sample =*it;
        sample.filterSamplesByID(filteredIDS);
        if (sample.getContourRegions().empty() && sample.getRectRegions().empty()){

        }
        else{
            this->samples.push_back(sample);
        }
    }
}

int DatasetReader::getNumberOfElements() {
    return this->samples.size();
}

void DatasetReader::resetReaderCounter() {
    this->readerCounter=0;

}

bool DatasetReader::getNetxSample(Sample &sample) {
    std::cout << "readCounter: " << this->readerCounter << ", size: " << this->samples.size() << std::endl;
    if (this->readerCounter < this->samples.size()){
        sample=this->samples[this->readerCounter];
        this->readerCounter++;
        return true;
    }
    else{
        return false;
    }

}

