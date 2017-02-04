//
// Created by frivas on 4/02/17.
//

#include <Utils/Logger.h>
#include "GenericInferencer.h"

GenericInferencer::GenericInferencer(const std::string &netConfig, const std::string &netWeights,
                                     const std::string &implementation) {

    configureAvailablesImplementations();
    if (std::find(this->availableImplementations.begin(), this->availableImplementations.end(), implementation) != this->availableImplementations.end()){
        imp = getImplementation(implementation);
        switch (imp) {
            case INF_YOLO:
                this->darknetInferencerPtr = DarknetInferencerPtr( new DarknetInferencer(netConfig, netWeights));
                break;
            default:
                Logger::getInstance()->error(implementation + " is not a valid inferencer implementation");
                break;
        }
    }
    else{
        Logger::getInstance()->error(implementation + " is not a valid inferencer implementation");
    }

}

void GenericInferencer::configureAvailablesImplementations() {
    this->availableImplementations.push_back("yolo");
}

INFERENCER_IMPLEMENTATIONS GenericInferencer::getImplementation(const std::string &inferencerImplementation) {
    if (inferencerImplementation.compare("yolo")==0){
        return INF_YOLO;
    }
}

FrameworkInferencerPtr GenericInferencer::getInferencer() {
    switch (imp) {
        case INF_YOLO:
            return this->darknetInferencerPtr;
        default:
//            Logger::getInstance()->error(imp + " is not a valid reader implementation");
            break;
    }
}

