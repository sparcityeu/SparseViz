//
// Created on 2/11/24.
//

#ifndef SPARSEVIZ_BELLMAN_FORD_H
#define SPARSEVIZ_BELLMAN_FORD_H

#include "MatrixKernelFunction.h"


class Bellman_Ford: public MatrixKernelFunction
{
public:
    Bellman_Ford(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
            :   MatrixKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore) {}
    virtual ~Bellman_Ford() override;
    virtual bool init(const SparseMatrix& A) override;
    virtual void preprocess(const SparseMatrix& A) override;
    virtual void functionBody(const SparseMatrix& A, int iterNumber) override;

public:
    vType singleSource;
    valType* dists;

private:
    const vType* ptrs;
    const vType* inds;
    const valType* vals;
};


#endif //SPARSEVIZ_BELLMAN_FORD_H
