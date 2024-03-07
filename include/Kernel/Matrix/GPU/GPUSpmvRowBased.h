//
// Created on 2/12/24.
//

#ifndef SPARSEVIZ_GPUSPMVROWBASED_H
#define SPARSEVIZ_GPUSPMVROWBASED_H

#include "MatrixGPUKernel.h"


class GPUSpmvRowBased: public MatrixGPUKernel
{
public:
    GPUSpmvRowBased(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore)
    : MatrixGPUKernel(kernelName, gridSizes, blockSizes, nRun, nIgnore) {}

    virtual ~GPUSpmvRowBased() override;
    virtual bool init(const SparseMatrix& A) override;
    virtual void preprocess(const SparseMatrix& A) override {}
    virtual void hostFunction(const SparseMatrix& A, int iterNumber, int gridSize, int blockSize) override;

private:
    valType* inputVector;
    valType* outputVector;
    valType* deviceInput;
    valType* deviceOutput;
    vType* deviceM;
    vType* ptrs;
    vType* inds;
    valType* vals;
};


#endif //SPARSEVIZ_GPUSPMVROWBASED_H
