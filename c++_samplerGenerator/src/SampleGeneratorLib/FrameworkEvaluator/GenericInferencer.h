//
// Created by frivas on 4/02/17.
//

#ifndef SAMPLERGENERATOR_GENERICINFERENCER_H
#define SAMPLERGENERATOR_GENERICINFERENCER_H

#include <FrameworkEvaluator/DarknetInferencer.h>

enum INFERENCER_IMPLEMENTATIONS{INF_YOLO};


class GenericInferencer {
public:
    GenericInferencer(const std::string& netConfig, const std::string& netWeights, const std::string& classNames, const std::string& implementation);
    FrameworkInferencerPtr getInferencer();

private:
    INFERENCER_IMPLEMENTATIONS imp;

    DarknetInferencerPtr darknetInferencerPtr;
    std::vector<std::string> availableImplementations;

    void configureAvailablesImplementations();
    INFERENCER_IMPLEMENTATIONS getImplementation(const std::string& inferencerImplementation);
};

typedef  boost::shared_ptr<GenericInferencer> GenericInferencerPtr;


#endif //SAMPLERGENERATOR_GENERICINFERENCER_H
