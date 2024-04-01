// Created by deniz on 3/30/24.
//

#include "MTTKRP.h"
#include "SparseMatrix.h"
#include "helpers.h"
#include <vector>


MTTKRP::MTTKRP(const std::string &kernelName, const std::vector<int> &threadCounts, const std::string &schedulingPolicy, int chunkSize, int nRun, int nIgnore, const std::string &orderingParameters)
:   TensorKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore)
{
    std::vector<std::string> parametersSplitted = split(orderingParameters, '/');
    if (parametersSplitted.empty())
    {
        throw std::runtime_error("MTTKRP tensor kernel requires mode parameter to be passed onto it as an argument in the config file. Terminating...");
    }
    try
    {
        mode = std::stoi(parametersSplitted[0]);
    }
    catch (const std::invalid_argument&)
    {
        throw std::runtime_error("Invalid argument format for mode parameter of the MTTKRP tensor kernel: " + parametersSplitted[0]);
    }
}

MTTKRP::~MTTKRP()
{
    delete matricizedTensor;
    delete khatriRaoProduct;
    for (vType i = 0; i < matricizedTensor->getRowCount(); ++i)
    {
        delete[] denseMat[i];
    }
    delete[] denseMat;
}

bool MTTKRP::init(const SparseTensor &A)
{
    matricizedTensor = A.matricizeTensor(mode);
    tPtrs = matricizedTensor->getPtr();
    tInds = matricizedTensor->getInd();
    tVals = matricizedTensor->getValues();

    khatriRaoProduct = new SparseMatrix("khatri_rao", matricizedTensor->getColCount(), matricizedTensor->getRowCount(), matricizedTensor->getNNZCount());
    kPtrs = khatriRaoProduct->getPtr();
    kInds = khatriRaoProduct->getInd();
    kVals = khatriRaoProduct->getValues();

    denseMat = new valType*[matricizedTensor->getRowCount()];
    for (vType i = 0; i < matricizedTensor->getRowCount(); ++i)
    {
        denseMat[i] = new valType[khatriRaoProduct->getColCount()]; // how come this line gives an error
        memset(denseMat[i], 0, sizeof(valType) * khatriRaoProduct->getColCount());
    }

    return true;
}

void MTTKRP::functionBody(const SparseTensor &A, int iterNumber)
{
    #pragma omp parallel for
    for(vType i = 0; i < matricizedTensor->getRowCount(); ++i)
    {
        for(vType tPtr = tPtrs[i]; tPtr < tPtrs[i + 1]; ++tPtr)
        {
            vType k = tInds[tPtr];
            valType tVal = tVals[tPtr];

            for(vType kPtr = kPtrs[k]; kPtr < kPtrs[k + 1]; ++kPtr)
            {
                vType j = kInds[kPtr];
                valType kVal = kVals[kPtr];

                denseMat[i][j] += tVal * kVal;
            }
        }
    }
}
