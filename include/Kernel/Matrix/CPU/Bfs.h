//
// Created on 1/18/24.
//

#ifndef SPARSEVIZ_BFS_H
#define SPARSEVIZ_BFS_H

#include "MatrixKernelFunction.h"


class BFS: public MatrixKernelFunction
{
public:
    BFS(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
            :   MatrixKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore) {
                level = nullptr;
            }

    virtual ~BFS() override;
    virtual bool init(const SparseMatrix& A) override;
    virtual void preprocess(const SparseMatrix& A);
    virtual void postprocess(const SparseMatrix& A, int iter) override;
    virtual void functionBody(const SparseMatrix& A, int iterNumber) override;

private:
    vType m;
    vType* ptrs;
    vType* ids;
    valType* level;
};


#endif //SPARSEVIZ_BFS_H
