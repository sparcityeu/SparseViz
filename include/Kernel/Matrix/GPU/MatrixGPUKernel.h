//
// Created on 2/2/24.
//

#ifndef SPARSEVIZ_MATRIXGPUKERNEL_H
#define SPARSEVIZ_MATRIXGPUKERNEL_H

#include "config.h"
#include "GPUKernelResult.h"
#include "SparseMatrix.h"


class MatrixGPUKernel
{
public:
    MatrixGPUKernel(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore)
            :   kernelName(kernelName), gridSizes(gridSizes), blockSizes(blockSizes), nRun(nRun), nIgnore(nIgnore) {}

    virtual ~MatrixGPUKernel() = default;
    GPUKernelResult operator()(const SparseMatrix& A);
    virtual bool init(const SparseMatrix& A) = 0;
    virtual void preprocess(const SparseMatrix& A) = 0;
    virtual void postprocess(const SparseMatrix& A, int r);
    virtual void hostFunction(const SparseMatrix& A, int iterNumber, int gridSize, int blockSize) = 0;

private:
    std::string kernelName;
    std::vector<int> gridSizes;
    std::vector<int> blockSizes;
    int nRun;
    int nIgnore;
};




#endif //SPARSEVIZ_MATRIXGPUKERNEL_H
