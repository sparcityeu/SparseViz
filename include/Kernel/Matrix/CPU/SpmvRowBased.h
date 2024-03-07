//
// Created on 1/18/24.
//

#ifndef SPARSEVIZ_SPMVROWBASED_H
#define SPARSEVIZ_SPMVROWBASED_H

#include "MatrixKernelFunction.h"


class SPMVRowBased: public MatrixKernelFunction
{
public:
    SPMVRowBased(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
            :   MatrixKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore) {
                x = nullptr;
                y = nullptr;
            }

    virtual ~SPMVRowBased() override;
    virtual bool init(const SparseMatrix& A) override;
    virtual void preprocess(const SparseMatrix& A) {}
    virtual void functionBody(const SparseMatrix& A, int iterNumber) override;

private:
    vType m, n;
    vType* ptrs;
    vType* ids;
    valType* vals;
    valType* x;
    valType* y;
};


#endif //SPARSEVIZ_SPMVROWBASED_H
