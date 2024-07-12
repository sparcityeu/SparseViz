//
// Created by on 5.12.2023.
//

#include "MatrixOrdering.h"
#include <cmath>
#include "helpers.h"
#include "SparseVizIO.h"


MatrixOrdering::MatrixOrdering(SparseMatrix &matrix, std::string orderingName, bool rectangularSupport, bool patternUnsymmetricSupport, unsigned int orderingFunctionPrice)
:   matrix(matrix),
    rowIPermutation(nullptr),
    colIPermutation(nullptr),
    orderingName(orderingName),
    rectangularSupport(rectangularSupport),
    patternUnsymmetricSupport(patternUnsymmetricSupport),
    isOrderingSupportedMatrixGenerated(false)
{
    this->calculateOrderingPrice(orderingFunctionPrice);
#ifdef CONFIG_LOG
    sparsevizlog("Ordering price of the " + orderingName ++ " on " ++ this->getMatrix().getName() ++ " is determined to be " + std::to_string(orderingPrice));
#endif
}

const SparseMatrix &MatrixOrdering::getMatrix()
{
    if (isOrderingSupportedMatrixGenerated)
    {
        //std::cout << isOrderingSupportedMatrixGenerated << std::endl;
        return *matrix.getOrderingSupportedMatrix();
    }
    return matrix;
}

MatrixOrdering::~MatrixOrdering()
{
    delete[] rowIPermutation;
    delete[] colIPermutation;
}

const vType* MatrixOrdering::getRowIPermutation() const
{
    if (!rowIPermutation)
    {
        throw std::runtime_error("The generate ordering method should have been called for one time before this method returns the ordered rows!");
    }
    else
    {
        return rowIPermutation;
    }
}

const vType* MatrixOrdering::getColIPermutation() const
{
    if (!colIPermutation)
    {
        throw std::runtime_error("The generate ordering method should have been called for one time before this method returns the ordered columns!");
    }
    else
    {
        return colIPermutation;
    }
}

void MatrixOrdering::pushKernelResult(const KernelResult &kernelResult)
{
    kernelResults.push_back(kernelResult);
}

void MatrixOrdering::pushGPUKernelResult(const GPUKernelResult& gpuKernelResult)
{
    gpuKernelResults.push_back(gpuKernelResult);
}

void MatrixOrdering::generateOrdering(bool writeOrdering)
{
    if (!USE_EXISTING_ORDERINGS || !this->readOrdering())
    {
        if (ORDERING_PERFORMANCE_LOG)
        {
            std::vector<CPUBenchmarkSettings> settings = getSettings();
            sparseVizPerformance->activatePerf(settings.data(), settings.size());
        }

        double start_time = omp_get_wtime();
        this->checkOrderingSupportingStatus();
        this->orderingFunction();
        this->transformGeneratedPermutation();
        this->checkPermutationCorrectness();
        double end_time = omp_get_wtime();

        SparseVizPerformance::OperationResults results;
        if (ORDERING_PERFORMANCE_LOG)
        {
            results = sparseVizPerformance->deactivatePerf();
        }
        logger->logMatrixOrdering(this, end_time - start_time, results);

        if(writeOrdering && EXPORT_ORDERINGS)
        {
            this->writeOrdering();
        }
    }
}

void MatrixOrdering::transformGeneratedPermutation()
{
    if (!isOrderingSupportedMatrixGenerated)
    {
        return;
    }

    int maxmn = matrix.getOrderingSupportedMatrix()->getColCount();
    int minmn = matrix.getRowCount() + matrix.getColCount() - maxmn;
    vType *tempPermutation = new vType[maxmn], *workingPtr;

    if (matrix.getRowCount() == maxmn)
    {
        workingPtr = colIPermutation;
    }
    else
    {
        workingPtr = rowIPermutation;
    }
    
    for (int i = 0; i != maxmn; ++i) tempPermutation[workingPtr[i]] = i;
    int count = 0;
    for (int i = 0; i < maxmn; ++i)
    {
        if(tempPermutation[i] < minmn)
        {
            workingPtr[tempPermutation[i]] = count++;
        }
    }
    delete [] tempPermutation;
    isOrderingSupportedMatrixGenerated = false;
}

bool MatrixOrdering::readOrdering()
{
    std::string binaryFileName = MATRIX_ORDERING_FILES_DIR + orderingName + "_" + this->getMatrix().getName() + ".bin";

    double start_time = omp_get_wtime();
    bool ret = SparseVizIO::readMatrixOrderingFromBinaryFile(binaryFileName, &rowIPermutation, &colIPermutation, this->getMatrix().getRowCount(), this->getMatrix().getColCount());
    double end_time = omp_get_wtime();

    if (ret) {
        logger->logReadingMatrixOrdering(this, end_time - start_time);
    }

    return ret;
}

void MatrixOrdering::writeOrdering()
{
    std::string binaryFileName = MATRIX_ORDERING_FILES_DIR + orderingName + "_" + this->getMatrix().getName() + ".bin";
    //std::cout << "kamer " << binaryFileName << std::endl;

    double start_time = omp_get_wtime();
    SparseVizIO::writeMatrixOrderingToBinaryFile(binaryFileName, rowIPermutation, colIPermutation, this->getMatrix().getRowCount(), this->getMatrix().getColCount());
    double end_time = omp_get_wtime();

    logger->logWritingMatrixOrdering(this, end_time - start_time);
}

void MatrixOrdering::checkPermutationCorrectness()
{
    std::vector<bool> visited(this->getMatrix().getRowCount(), false);
    for (int i = 0; i != this->getMatrix().getRowCount(); ++i)
    {
        if (rowIPermutation[i] < 0 || (rowIPermutation[i] >= this->getMatrix().getRowCount()))
        {
            throw std::logic_error("The row permutation for " + orderingName + " is not generated correctly: row " + std::to_string(i) + " has an ordering value " + std::to_string(rowIPermutation[i]));
        }
        visited[rowIPermutation[i]] = true;
    }
    for (int i = 0; i != this->getMatrix().getRowCount(); ++i)
    {
        if (!visited[rowIPermutation[i]])
        {
            throw std::logic_error("The row permutation for " + orderingName + " is not generated correctly: " + std::to_string(visited[rowIPermutation[i]]) + " appears twice at least!") ;
        }
    }

    visited.clear();
    visited.resize(this->getMatrix().getColCount(), false);
    for (int i = 0; i != this->getMatrix().getColCount(); ++i)
    {
        if (colIPermutation[i] < 0 || (colIPermutation[i] >= this->getMatrix().getColCount()))
        {
            throw std::logic_error("The column permutation for " + orderingName + " is not generated correctly: column " + std::to_string(i) + " has an ordering value " + std::to_string(colIPermutation[i]));
        }
        visited[colIPermutation[i]] = true;
    }
    for (int i = 0; i != this->getMatrix().getColCount(); ++i)
    {
        if (!visited[colIPermutation[i]])
        {
            throw std::logic_error("The column permutation for " + orderingName + " is not generated correctly: " + std::to_string(visited[colIPermutation[i]]) + " appears twice at least!") ;
        }
    }
}

void MatrixOrdering::calculateOrderingPrice(unsigned int orderingFunctionPrice)
{
    const double nnzBaseFactor = 0.05;
    const double rowColBaseFactor = 0.001;

    double nnzPrice = log(this->getMatrix().getNNZCount() + 1) * nnzBaseFactor;
    double rowPrice = log(this->getMatrix().getRowCount() + 1) * rowColBaseFactor;
    double colPrice = log(this->getMatrix().getColCount() + 1) * rowColBaseFactor;

    orderingPrice = orderingFunctionPrice + nnzPrice + rowPrice + colPrice;
}

void MatrixOrdering::checkOrderingSupportingStatus()
{
#pragma omp critical (orderingSupportingLock)
    {
        if (!this->getMatrix().isPatternSymmetric() && !patternUnsymmetricSupport)
        {
            if (matrix.getOrderingSupportedMatrix())
            {
                std::cout << orderingName << " dictates matrix to be a pattern symmetric matrix. Fortunately, " << this->getMatrix().getName() << " has already been converted into one." << std::endl;
            }
            else
            {
                std::cout << orderingName << " dictates matrix to be a pattern symmetric matrix. " << this->getMatrix().getName() << " is being converted into one." << std::endl;
                this->matrix.generateOrderingSupportedMatrix();
            }
            isOrderingSupportedMatrixGenerated = true;
        }
    }
}
