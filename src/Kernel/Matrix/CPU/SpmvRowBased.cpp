//
// Created on 1/18/24.
//

#include "SpmvRowBased.h"


bool SPMVRowBased::init(const SparseMatrix &A)
{
    m = A.getRowCount();
    n = A.getColCount();
    ptrs = A.getPtr();
    ids = A.getInd();
    vals = A.getValues();
    x = new valType[n];
    y = new valType[m];

    return true;
}

SPMVRowBased::~SPMVRowBased()
{
    delete[] x;
    delete[] y;
}

void SPMVRowBased::functionBody(const SparseMatrix& A, int iterNumber)
{
#pragma omp parallel for
    for (vType i = 0; i < m; ++i)  {
        valType temp = 0;
        for (vType ptr = ptrs[i]; ptr < ptrs[i + 1]; ++ptr) {
            temp += vals[ptr] * x[ids[ptr]];
        }
        y[i] = temp;
    }
}
