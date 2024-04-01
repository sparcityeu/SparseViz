//
// Created by deniz on 3/30/24.
//

#ifndef SPARSEVIZ_MTTKRP_H
#define SPARSEVIZ_MTTKRP_H

#include "TensorKernelFunction.h"


class SparseMatrix;

class MTTKRP: public TensorKernelFunction
{
public:
    MTTKRP(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore, const std::string& orderingParameters);

    virtual ~MTTKRP() override;
    virtual bool init(const SparseTensor& A) override;
    virtual void preprocess(const SparseTensor& A) {};
    virtual void functionBody(const SparseTensor& A, int iterNumber) override;

private:
    vType mode; // Mode for unfolding the tensor

    SparseMatrix* matricizedTensor;
    SparseMatrix* khatriRaoProduct;

    const vType* tPtrs;
    const vType* tInds;
    const valType* tVals;

    const vType* kPtrs;
    const vType* kInds;
    const valType* kVals;

    valType** denseMat;
};


#endif //SPARSEVIZ_MTTKRP_H
