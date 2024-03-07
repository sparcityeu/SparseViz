//
// Created on 1/18/24.
//

#include "SpmmRowBased.h"

bool SPMMRowBased::init(const SparseMatrix &A)
{
    m = A.getRowCount();
    n = A.getColCount();
    ptrs = A.getPtr();
    ids = A.getInd();
    vals = A.getValues();
    x = new valType[n * latent_dimension];
    y = new valType[m * latent_dimension];

    return true;
}

SPMMRowBased::~SPMMRowBased()
{
    delete[] x;
    delete[] y;
}

void SPMMRowBased::functionBody(const SparseMatrix& A, int iterNumber)
{
#pragma omp parallel for
    for (vType i = 0; i < m; ++i)  {
        valType y_vec[latent_dimension];
        memset(y_vec, 0, sizeof(valType) * latent_dimension); 
        for (vType ptr = ptrs[i]; ptr < ptrs[i + 1]; ++ptr) {
            vType a_id = ids[ptr];
            valType a_val = vals[ptr];
            valType* x_vec = x + (latent_dimension * a_id);
            
            #pragma unroll
            for(int j = 0; j < latent_dimension; j++) {
                y_vec[j] += a_val * x_vec[j];
            }
        }
        memcpy(y + (latent_dimension * i), y_vec, sizeof(valType) * latent_dimension);
    }
}
