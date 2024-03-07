#ifndef SPARSEVIZ_SPMMROWBASED_H
#define SPARSEVIZ_SPMMROWBASED_H

#include "MatrixKernelFunction.h"

class SPMMRowBased: public MatrixKernelFunction
{
public:
    SPMMRowBased(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
            :   MatrixKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore) {
                x = nullptr;
                y = nullptr;
            }

    virtual ~SPMMRowBased() override;
    virtual bool init(const SparseMatrix& A) override;
    virtual void preprocess(const SparseMatrix& A) {}
    virtual void functionBody(const SparseMatrix& A, int iterNumber) override;

private:
    const int latent_dimension = 8;
    vType m, n;
    vType* ptrs;
    vType* ids;
    valType* vals;
    valType* x;
    valType* y;
};

#endif //SPARSEVIZ_SPMMROWBASED_H
