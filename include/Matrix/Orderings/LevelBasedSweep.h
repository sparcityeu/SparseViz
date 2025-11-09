#pragma once

#include "MatrixOrdering.h"

class LevelBasedSweep: public MatrixOrdering
{
public:
    LevelBasedSweep(SparseMatrix& matrix, std::string orderingName): MatrixOrdering(matrix, orderingName, false, false) {}

private:
    virtual void orderingFunction() override;
};
