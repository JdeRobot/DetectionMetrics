//
// Created by frivas on 4/02/17.
//

#include "Key.h"
#include "Logger.h"

Key::Key(const std::string &key):key(key) {

}


bool Key::isVector() {
    return this->values.size()==1;
}

void Key::addValue(const std::string &value) {
    this->values.push_back(value);
}

std::string Key::getValue() {
    if (this->values.size()==1)
        return this->values[0];
    else {
        const std::string ErrorMsg="Key [" + this->key + "] is an array not value";
        Logger::getInstance()->error(ErrorMsg);
        throw ErrorMsg;
    }
}

std::string Key::getKey() {
    return this->key;
}

std::string Key::getValue(int id) {
    if (this->values.size()> id)
    {
        return this->values[id];
    }
}

std::vector<std::string> Key::getValues() {
    return this->values;
}

Key::Key() {

}

int Key::getNValues() {
    return this->values.size();
}

