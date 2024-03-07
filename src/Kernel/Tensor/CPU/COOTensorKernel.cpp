//
// Created on 1/18/24.
//

#include "COOTensorKernel.h"

bool COOTensorKernel::init(const SparseTensor &T)
{
    nonzeros = T.getStorage();
    dims = T.getDims();
    nnzcount = T.getNNZCount();
    order = T.getOrder();
    
    arrays = new float*[order];
    for(int i = 0; i < order; i++) {
        arrays[i] = new float[dims[i]];
        for(vType j = 0; j < dims[i]; j++) {
            arrays[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }
    
    return true;
}

COOTensorKernel::~COOTensorKernel()
{
    for(int i = 0; i < order; i++) {
        delete [] arrays[i];
    }
    delete [] arrays;
}

void COOTensorKernel::functionBody(const SparseTensor& A, int iterNumber)
{
    float sum = 0, temp;
    for(int i = 0; i < 1; i++) {
#pragma omp parallel for reduction(+:sum)
        for (eType e = 0; e < nnzcount; ++e)  {
            temp = 1;
            vType* nnz = nonzeros + (e * order);
            for (int m = 0; m < order; m++) {
                temp *= arrays[m][nnz[m]];
            } 
            sum += temp;
        }
    }
}
