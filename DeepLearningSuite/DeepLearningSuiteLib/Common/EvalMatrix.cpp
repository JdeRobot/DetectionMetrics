#include "EvalMatrix.h"
#include <iostream>

void Eval::printMatrix(Eval::EvalMatrix matrix) {

    std::cout << "Printing Matrix" << '\n';
    for (auto it = matrix.begin(); it != matrix.end(); it++) {
        std::cout << "ID: " << it->first << '\n';
        for (auto itr = it->second.begin(); itr != it->second.end(); itr++) {
            std::cout << "ClassID: " << itr->first <<'\n';
            for (auto iter = itr->second.begin(); iter != itr->second.end(); iter++ ) {
                for (auto iterate = iter->begin(); iterate != iter->end(); iterate++) {
                    std::cout << *iterate << " ";
                }
                std::cout << '\n';
            }
        }
    }

}
