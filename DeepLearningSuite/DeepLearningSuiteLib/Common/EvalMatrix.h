#ifndef SAMPLERGENERATOR_EVALMATRIX_H
#define SAMPLERGENERATOR_EVALMATRIX_H

#include "Matrix.h"
#include <map>

namespace Eval {

using EvalMatrix = std::map<std::string, std::map<std::string, Matrix<double>> >;

using DetectionsMatcher = std::map<double thresh, std::map<std::string sampleID, std::map<int dt, int gt>>>;

void printMatrix(EvalMatrix matrix);

}
#endif //SAMPLERGENERATOR_MATRIX_H
