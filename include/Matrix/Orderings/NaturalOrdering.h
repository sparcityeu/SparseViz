#ifndef SPARSEVIZ_NATURALORDERING_H
#define SPARSEVIZ_NATURALORDERING_H

#include "MatrixOrdering.h"
#include <string>


class NaturalOrdering : public MatrixOrdering
{
public:
    NaturalOrdering(SparseMatrix& matrix, std::string orderingName): MatrixOrdering(matrix, orderingName, true, true) {}

private:
    virtual void orderingFunction() override;
};


#endif //SPARSEVIZ_NATURALORDERING_H
