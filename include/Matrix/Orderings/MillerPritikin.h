#pragma once

#include "MatrixOrdering.h"

class MillerPritikin: public MatrixOrdering
{
public:
    MillerPritikin(SparseMatrix& matrix, std::string orderingName): MatrixOrdering(matrix, orderingName, false, false) {}

private:
    virtual void orderingFunction() override;
};
