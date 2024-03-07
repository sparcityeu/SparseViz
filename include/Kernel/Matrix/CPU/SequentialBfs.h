//
// Created on 1/19/24.
//

#ifndef SPARSEVIZ_SEQUENTIALBFS_H
#define SPARSEVIZ_SEQUENTIALBFS_H

#include "MatrixKernelFunction.h"


class SequentialBFS: public MatrixKernelFunction
{
public:
    SequentialBFS(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
            :   MatrixKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore) {
                level = nullptr;
                queue = nullptr;
            }

    virtual ~SequentialBFS() override;
    virtual bool init(const SparseMatrix& A) override;
    virtual void preprocess(const SparseMatrix& A);
    virtual void postprocess(const SparseMatrix& A, int iter) override;
    virtual void functionBody(const SparseMatrix& A, int iterNumber) override;

private:
    vType m;
    vType* ptrs;
    vType* ids;
    valType* level;
    valType* queue;
};


#endif //SPARSEVIZ_SEQUENTIALBFS_H
