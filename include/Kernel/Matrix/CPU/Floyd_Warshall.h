//
// Created on 2/11/24.
//

#ifndef SPARSEVIZ_FLOYD_WARSHALL_H
#define SPARSEVIZ_FLOYD_WARSHALL_H

#include "MatrixKernelFunction.h"


class Floyd_Warshall: public MatrixKernelFunction
{
public:
    Floyd_Warshall(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
            :   MatrixKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore) {}
    virtual ~Floyd_Warshall() override;
    virtual bool init(const SparseMatrix& A) override;
    virtual void preprocess(const SparseMatrix& A) override;
    virtual void functionBody(const SparseMatrix& A, int iterNumber) override;

public:
    unsigned** dists;

private:
    vType vertexCount;
    const vType* ptrs;
    const vType* inds;
    const valType* vals;
};


#endif //SPARSEVIZ_FLOYD_WARSHALL_H
