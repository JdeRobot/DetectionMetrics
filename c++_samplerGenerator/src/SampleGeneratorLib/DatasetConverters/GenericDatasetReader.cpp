//
// Created by frivas on 4/02/17.
//

#include <Utils/Logger.h>
#include "GenericDatasetReader.h"


GenericDatasetReader::GenericDatasetReader(const std::string &path, const std::string &readerImplementation) {
    configureAvailablesImplementations();
    if (std::find(this->availableImplementations.begin(), this->availableImplementations.end(), readerImplementation) != this->availableImplementations.end()){
        imp = getImplementation(readerImplementation);
        switch (imp) {
            case YOLO:
                this->yoloDatasetReaderPtr = YoloDatasetReaderPtr( new YoloDatasetReader(path));
                break;
            case SPINELLO:
                this->spinelloDatasetReaderPtr = SpinelloDatasetReaderPtr( new SpinelloDatasetReader(path));
                break;
            case OWN:
                this->ownDatasetReaderPtr = OwnDatasetReaderPtr( new OwnDatasetReader(path));
                break;
            default:
                Logger::getInstance()->error(readerImplementation + " is not a valid reader implementation");
                break;
        }
    }
    else{
        Logger::getInstance()->error(readerImplementation + " is not a valid reader implementation");
    }
}



void GenericDatasetReader::configureAvailablesImplementations() {
    this->availableImplementations.push_back("yolo");
    this->availableImplementations.push_back("spinello");
    this->availableImplementations.push_back("own");
}

READER_IMPLEMENTATIONS GenericDatasetReader::getImplementation(const std::string& readerImplementation) {
    if (readerImplementation.compare("yolo")==0){
        return YOLO;
    }
    if (readerImplementation.compare("spinello")==0){
        return SPINELLO;
    }
    if (readerImplementation.compare("own")==0){
        return OWN;
    }
}

DatasetReaderPtr GenericDatasetReader::getReader() {
    switch (imp) {
        case YOLO:
            return this->yoloDatasetReaderPtr;
        case SPINELLO:
            return this->spinelloDatasetReaderPtr;
        case OWN:
            return this->ownDatasetReaderPtr;
        default:
//            Logger::getInstance()->error(imp + " is not a valid reader implementation");
            break;
    }
}
