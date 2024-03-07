//
// Created on 2/2/24.
//

#ifndef SPARSEVIZ_CUDAHELLOWORLD_H
#define SPARSEVIZ_CUDAHELLOWORLD_H

#include "MatrixGPUKernel.h"


class CudaHelloWorld: public MatrixGPUKernel
{
public:
    CudaHelloWorld(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore)
            : MatrixGPUKernel(kernelName, gridSizes, blockSizes, nRun, nIgnore), N(128) {}

    virtual ~CudaHelloWorld() override;
    virtual bool init(const SparseMatrix& A);
    virtual void preprocess(const SparseMatrix& A) {}
    virtual void hostFunction(const SparseMatrix& A, int iterNumber, int gridSize, int blockSize);

private:
    const int N;
    char* hostOutput;
};


#endif //SPARSEVIZ_CUDAHELLOWORLD_H
