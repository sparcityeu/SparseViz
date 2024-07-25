#ifndef SPARSEVIZ_COO_TENSOR_KERNEL_H
#define SPARSEVIZ_COO_TENSOR_KERNEL_H

#include "TensorKernelFunction.h"


class COOTensorKernel: public TensorKernelFunction
{
public:
    COOTensorKernel(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
    :   TensorKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore) {}

    virtual ~COOTensorKernel() override;
    virtual bool init(const SparseTensor& A) override;
    virtual void preprocess(const SparseTensor& A) {};
    virtual void functionBody(const SparseTensor& A, int iterNumber) override;

private:
    float** arrays;
    vType* nonzeros;
    vType* dims;
    eType nnzcount;
    int order;
};

#endif //SPARSEVIZ_COO_TENSOR_KERNEL_H
