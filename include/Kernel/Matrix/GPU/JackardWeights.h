//
// Created by deniz on 5/8/24.
//

#ifndef SPARSEVIZ_JACKARDWEIGHTS_H
#define SPARSEVIZ_JACKARDWEIGHTS_H

#include "MatrixGPUKernel.h"


class JackardWeights: public MatrixGPUKernel
{
public:
    JackardWeights(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, const std::vector<int>& sharedMemorySizes, std::string kernelParameters, int nRun, int nIgnore);

    virtual ~JackardWeights() override;
    virtual bool init(const SparseMatrix& A) override;
    virtual void preprocess(const SparseMatrix& A) override {}
    virtual void hostFunction(const SparseMatrix& A, int iterNumber, int gridSize, int blockSize, int sharedMemorySize) override;

private:
    unsigned assemblySize;
    unsigned searchGroupSize;
    vType* ptrs;
    vType* inds;
};


#endif //SPARSEVIZ_JACKARDWEIGHTS_H
