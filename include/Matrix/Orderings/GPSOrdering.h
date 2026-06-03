// GPSOrdering.h
#pragma once
#include "MatrixOrdering.h"   // brings SparseMatrix + config.h (vType)
#include "SparseMatrix.h"
#include <string>
#include <vector>

class GPSOrdering : public MatrixOrdering {
public:
    GPSOrdering(SparseMatrix& m, std::string name, std::string /*unused*/)
    : MatrixOrdering(m, std::move(name),
                     /*rectangularSupport=*/false,
                     /*patternUnsymmetricSupport=*/false,
                     /*price=*/3.0) {}

private:
    void orderingFunction() override;
};
