//
// Created on 2/12/24.
//

#ifndef SPARSEVIZ_COOGPUTENSORKERNELSOA_H
#define SPARSEVIZ_COOGPUTENSORKERNELSOA_H

#include "TensorGPUKernel.h"

class COOTensorGPUKernelSoA: public TensorGPUKernel
{
public:
    COOTensorGPUKernelSoA(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore)
    : TensorGPUKernel(kernelName, gridSizes, blockSizes, nRun, nIgnore) {}

    virtual ~COOTensorGPUKernelSoA() override;
    virtual bool init(const SparseTensor& A) override;
    virtual void preprocess(const SparseTensor& A) override {}
    virtual void hostFunction(const SparseTensor& A, int iterNumber, int gridSize, int blockSize) override;

private:
    float* h_arrays;
    float* d_arrays;
    float* d_output;
    float h_output;
    vType* h_nonzeros;
    vType* d_nonzeros;
    vType* dims;
    eType nnzcount;
    int order;
};

#endif //SPARSEVIZ_COOGPUTENSORKERNELSOA_H
