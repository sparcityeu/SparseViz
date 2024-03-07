//
// Created on 1/18/24.
//

#ifndef SPARSEVIZ_MATRIXKERNELFUNCTION_H
#define SPARSEVIZ_MATRIXKERNELFUNCTION_H

#include "config.h"
#include "KernelResult.h"
#include "SparseMatrix.h"


class MatrixKernelFunction
{
public:
    MatrixKernelFunction(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
            :   kernelName(kernelName), threadCounts(threadCounts), schedulingPolicy(schedulingPolicy), chunkSize(chunkSize), nRun(nRun), nIgnore(nIgnore) {}

    virtual ~MatrixKernelFunction() = default;
    KernelResult operator()(const SparseMatrix& A);
    virtual bool init(const SparseMatrix& A) = 0;
    virtual void preprocess(const SparseMatrix& A) = 0;
    virtual void postprocess(const SparseMatrix& A, int r);
    virtual void functionBody(const SparseMatrix& A, int iterNumber) = 0;

private:
    std::string kernelName;
    std::vector<int> threadCounts;
    std::string schedulingPolicy;
    int chunkSize;
    int nRun;
    int nIgnore;

private:
    void determineOMPSchedule();
};


#endif //SPARSEVIZ_MATRIXKERNELFUNCTION_H
