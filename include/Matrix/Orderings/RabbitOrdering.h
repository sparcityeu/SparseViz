//
// Created by Kamer Kaya on 25.12.2023.
//

#ifndef SPARSEVIZ_RABBITORDERING_H
#define SPARSEVIZ_RABBITORDERING_H

#include "MatrixOrdering.h"
#include <string>

class RabbitOrdering : public MatrixOrdering
{
public:
    RabbitOrdering(SparseMatrix& matrix, std::string orderingName): MatrixOrdering(matrix, orderingName, false, false) {}

private:
    virtual void orderingFunction() override;
};


#endif //SPARSEVIZ_RABBITORDERING_H
