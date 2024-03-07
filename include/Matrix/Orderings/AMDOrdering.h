//
// Created by Kamer Kaya on 25.12.2023.
//

#ifndef SPARSEVIZ_AMDORDERING_H
#define SPARSEVIZ_AMDORDERING_H

#include "MatrixOrdering.h"
#include <string>

class AMDOrdering : public MatrixOrdering
{
public:
    AMDOrdering(SparseMatrix& matrix, std::string orderingName): MatrixOrdering(matrix, orderingName, false, false) {}

private:
    virtual void orderingFunction() override;
};


#endif //SPARSEVIZ_AMDORDERING_H
