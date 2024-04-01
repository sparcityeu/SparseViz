//
// Created on 2/4/24.
//

#ifndef SPARSEVIZ_TENSORGPUKERNEL_H
#define SPARSEVIZ_TENSORGPUKERNEL_H

#include "config.h"
#include "GPUKernelResult.h"
#include "SparseTensor.h"


class TensorGPUKernel
{
public:
    TensorGPUKernel(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore)
    :   kernelName(kernelName), gridSizes(gridSizes), blockSizes(blockSizes), nRun(nRun), nIgnore(nIgnore) {}

    virtual ~TensorGPUKernel() = default;
    GPUKernelResult operator()(const SparseTensor& A);
    virtual bool init(const SparseTensor& A) = 0;
    virtual void preprocess(const SparseTensor& A) = 0;
    virtual void postprocess(const SparseTensor& A, int r);
    virtual void hostFunction(const SparseTensor& A, int iterNumber, int gridSize, int blockSize) = 0;

private:
    std::string kernelName;
    std::vector<int> gridSizes;
    std::vector<int> blockSizes;
    int nRun;
    int nIgnore;
};


#endif //SPARSEVIZ_TENSORGPUKERNEL_H
