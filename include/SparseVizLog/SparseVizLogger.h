#ifndef SPARSEVIZLOGGER_SPARSEVIZLOGGER_H
#define SPARSEVIZLOGGER_SPARSEVIZLOGGER_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <set>

class SparseMatrix;
class MatrixOrdering;
class SparseTensor;
class TensorOrdering;
struct KernelResult;
struct GPUKernelResult;
struct Statistic;
struct TStatistic;

class SparseVizLogger
{
public:
    void createCSVFile(const std::string& filePath);

    void logReadingMatrixMarket(SparseMatrix* matrix, double duration);
    void logReadingMatrixBinary(SparseMatrix* matrix, double duration);
    void logWritingMatrixBinary(SparseMatrix* matrix, double duration);
    void logMatrixOrdering(MatrixOrdering* ordering, double duration);
    void logReadingMatrixOrdering(MatrixOrdering* ordering, double duration);
    void logWritingMatrixOrdering(MatrixOrdering* ordering, double duration);
    void logConstructingOrderedMatrix(MatrixOrdering* ordering, double duration);

    void logReadingTensorMarket(SparseTensor* tensor, double duration);
    void logReadingTensorBinary(SparseTensor* tensor, double duration);
    void logWritingTensorBinary(SparseTensor* tensor, double duration);
    void logTensorOrdering(TensorOrdering* ordering, double duration);
    void logReadingTensorOrdering(TensorOrdering* ordering, double duration);
    void logWritingTensorOrdering(TensorOrdering* ordering, double duration);
    void logConstructingOrderedTensor(TensorOrdering* ordering, double duration);

    void logRunningMatrixKernel(const KernelResult& kernelResult, MatrixOrdering* ordering);
    void logRunningGPUMatrixKernel(const GPUKernelResult& kernelResult, MatrixOrdering* ordering);
    void logRunningTensorKernel(const KernelResult& kernelResult, TensorOrdering* ordering);
    void logRunningGPUTensorKernel(const GPUKernelResult& kernelResult, TensorOrdering* ordering);

    void logMatrixProcessing(const std::string& filename, const Statistic& stat, double duration);
    void logTensorProcessing(const std::string& filename, const TStatistic& stat, double duration);

    void makeSilentLog(std::string message, double time = 0);
    void makeExplicitLog(std::string message, double time = 0);

private:
    std::string boolToString(bool boolean);

    std::string determineCompiler();
    std::string determineArchitecture();
    std::string determineOS();
    bool isOMPEnabled();
    bool isCudaEnabled();

    void logTest(std::string line);

    std::vector<std::string> m_ReadingMatrixMarket;
    std::vector<std::string> m_ReadingMatrixBinary;
    std::vector<std::string> m_WritingMatrixBinary;
    std::vector<std::string> m_MatrixOrdering;
    std::vector<std::string> m_ReadingMatrixOrdering;
    std::vector<std::string> m_WritingMatrixOrdering;
    std::vector<std::string> m_ConstructingOrderedMatrix;

    std::vector<std::string> m_ReadingTensorMarket;
    std::vector<std::string> m_ReadingTensorBinary;
    std::vector<std::string> m_WritingTensorBinary;
    std::vector<std::string> m_TensorOrdering;
    std::vector<std::string> m_ReadingTensorOrdering;
    std::vector<std::string> m_WritingTensorOrdering;
    std::vector<std::string> m_ConstructingOrderedTensor;

    std::vector<std::string> m_RunningMatrixKernel;
    std::set<int> m_MatrixKernelThreadCounts;
    std::vector<std::vector<int>> m_ThreadCountsForEachMatrixKernel;
    std::vector<std::vector<double>> m_DurationsForEachMatrixKernel;

    std::vector<std::string> m_RunningTensorKernel;
    std::set<int> m_TensorKernelThreadCounts;
    std::vector<std::vector<int>> m_ThreadCountsForEachTensorKernel;
    std::vector<std::vector<double>> m_DurationsForEachTensorKernel;

    std::vector<std::string> m_MatrixProcessing;
    std::vector<std::string> m_TensorProcessing;

    std::vector<std::string> m_Others;
};


#endif //SPARSEVIZLOGGER_SPARSEVIZLOGGER_H
