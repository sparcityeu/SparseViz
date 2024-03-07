#ifndef SPARSEVIZ_TRIVIALORDERING_H
#define SPARSEVIZ_TRIVIALORDERING_H

#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include <vector>
#include <algorithm>
#include "config.h"


class TrivialOrdering: public MatrixOrdering
{
public:
    TrivialOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters);

private:
    virtual void orderingFunction() override;

private:
    // COMPARATORS TO BE GIVEN TO THE SORTING ALGORITHMS
    static int ascendingComparator(const void* a, const void* b);
    static int descendingComparator(const void* a, const void* b);

private:
    std::string m_OrderingCriteria;
    std::string m_OrderingOrientation;
    std::string m_SortingDirection;
};


#endif //SPARSEVIZ_TRIVIALORDERING_H
