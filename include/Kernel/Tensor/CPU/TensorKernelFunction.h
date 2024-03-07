//
// Created on 1/18/24.
//

#ifndef SPARSEVIZ_TENSORKERNELFUNCTION_H
#define SPARSEVIZ_TENSORKERNELFUNCTION_H

#include "config.h"
#include "KernelResult.h"
#include "SparseTensor.h"


class TensorKernelFunction
{
public:
    TensorKernelFunction(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
            :   kernelName(kernelName), threadCounts(threadCounts), schedulingPolicy(schedulingPolicy), chunkSize(chunkSize), nRun(nRun), nIgnore(nIgnore) {}

    virtual ~TensorKernelFunction() = default;
    KernelResult operator()(const SparseTensor& A);
    virtual bool init(const SparseTensor& A) = 0;
    virtual void preprocess(const SparseTensor& A) = 0;
    virtual void functionBody(const SparseTensor& A, int iterNumber) = 0;

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


#endif //SPARSEVIZ_TENSORKERNELFUNCTION_H
