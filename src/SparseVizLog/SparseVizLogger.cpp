#include "SparseVizLogger.h"
#include "SparseMatrix.h"
#include "MatrixOrdering.h"
#include "SparseTensor.h"
#include "TensorOrdering.h"
#include "KernelResult.h"
#include "MatrixVisualizer.h"
#include <iomanip>
#include "TensorVisualizer.h"
#include "SparseVizTest.h"


void SparseVizLogger::createCSVFile(const std::string& filePath)
{
    std::ofstream csvFile(filePath);

    if (!csvFile.is_open()) {
        throw std::runtime_error("Unable to open csv file for logging.");
    }

    csvFile << std::fixed << std::setprecision(3);

    csvFile << "SparseViz Performance Logger\n\n";

    // Section for Machine Details
    csvFile << "Compiler\tOS\tArchitecture\tOMP Enabled\tCuda Enabled\n";
    std::string machineDetails = determineCompiler() + '\t' + determineOS() + '\t' + determineArchitecture() + '\t' +
                                 boolToString(isOMPEnabled()) + '\t' + boolToString(isCudaEnabled());
    csvFile << machineDetails << std::endl;
    csvFile << std::endl;

    if (!m_ReadingMatrixMarket.empty())
    {
        csvFile << "Operation: Reading Matrix From The Market File\n";
        csvFile << "Name\tRow Count\tColumn Count\tNonzero Count\tSymmetric\tPattern Symmetric\tSquare\tDuration\n";
        for (const auto& logEntry : m_ReadingMatrixMarket)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_ReadingMatrixBinary.empty())
    {
        csvFile << "Operation: Reading Matrix From The Binary File\n";
        csvFile << "Name\tRow Count\tColumn Count\tNonzero Count\tSymmetric\tPattern Symmetric\tSquare\tDuration\n";
        for (const auto& logEntry : m_ReadingMatrixBinary)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_WritingMatrixBinary.empty())
    {
        csvFile << "Operation: Writing Matrix To A Binary File\n";
        csvFile << "Name\tRow Count\tColumn Count\tNonzero Count\tSymmetric\tPattern Symmetric\tSquare\tDuration\n";
        for (const auto& logEntry : m_WritingMatrixBinary)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_MatrixOrdering.empty())
    {
        // Metrics
        csvFile << "Operation: Matrix Ordering\n";
        csvFile << "Ordering Name\tMatrix Name\tOrdering Price\tOrdering Rectangular Support\tOrdering Pattern Unsymmetric Support\tMatrix Row Count\tMatrix Column Count\tMatrix NNZ Count\tMatrix Symmetric\tMatrix Pattern "
                   "Symmetric\tMatrix Square\tDuration\t";
        if (!m_MatrixOrderingResults.empty())
        {
            for (const auto& result: m_MatrixOrderingResults[0])
            {
                csvFile << result.first << '\t';
            }
        }
        csvFile << '\n';

        // Results
        for (size_t i = 0; i < m_MatrixOrdering.size(); ++i)
        {
            csvFile << m_MatrixOrdering[i] << '\t';
            if (!m_MatrixOrderingResults.empty())
            {
                for (const auto& result: m_MatrixOrderingResults[i])
                {
                    csvFile << result.second.second << '\t';
                }
            }
            csvFile << '\n';
        }

        csvFile << "\n";
    }

    if (!m_ReadingMatrixOrdering.empty())
    {
        csvFile << "Operation: Reading Matrix Ordering From The Binary File\n";
        csvFile << "Ordering Name\tMatrix Name\tOrdering Price\tOrdering Rectangular Support\tOrdering Pattern Unsymmetric Support\tMatrix Row Count\tMatrix Column Count\tMatrix NNZ Count\tMatrix Symmetric\tMatrix Pattern "
                   "Symmetric\tMatrix Square\tDuration\n";
        for (const auto& logEntry : m_ReadingMatrixOrdering)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_WritingMatrixOrdering.empty())
    {
        csvFile << "Operation: Writing Matrix Ordering To A Binary File\n";
        csvFile << "Ordering Name\tMatrix Name\tOrdering Price\tOrdering Rectangular Support\tOrdering Pattern Unsymmetric Support\tMatrix Row Count\tMatrix Column Count\tMatrix NNZ Count\tMatrix Symmetric\tMatrix Pattern "
                   "Symmetric\tMatrix Square\tDuration\n";
        for (const auto& logEntry : m_WritingMatrixOrdering)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_ConstructingOrderedMatrix.empty())
    {
        csvFile << "Operation: Construction of Ordered Matrix\n";
        csvFile << "Ordering Name\tMatrix Name\tOrdering Price\tOrdering Rectangular Support\tOrdering Pattern Unsymmetric Support\tMatrix Row Count\tMatrix Column Count\tMatrix NNZ Count\tMatrix Symmetric\tMatrix Pattern "
                   "Symmetric\tMatrix Square\tDuration\n";
        for (const auto& logEntry : m_ConstructingOrderedMatrix)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_ReadingTensorMarket.empty())
    {
        csvFile << "Operation: Reading Tensor From The Market File\n";
        unsigned long maxDim = 0;
        for (auto& logEntry : m_ReadingTensorMarket)
        {
            std::vector<std::string> splittedString = split(logEntry, '|');
            maxDim = std::max(maxDim, std::stoul(splittedString[0]));
            logEntry = splittedString[1];
        }
        csvFile << "Name\t";
        for (int i = 1; i <= maxDim; ++i)
        {
            csvFile << "Dimension: " << i << '\t';
        }
        csvFile << "Nonzero Count\tDuration\n";
        for (const auto& logEntry : m_ReadingTensorMarket)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_ReadingTensorBinary.empty())
    {
        csvFile << "Operation: Reading Tensor From The Binary File\n";
        unsigned long maxDim = 0;
        for (auto& logEntry : m_ReadingTensorBinary)
        {
            std::vector<std::string> splittedString = split(logEntry, '|');
            maxDim = std::max(maxDim, std::stoul(splittedString[0]));
            logEntry = splittedString[1];
        }
        csvFile << "Name\t";
        for (int i = 1; i <= maxDim; ++i)
        {
            csvFile << "Dimension: " << i << '\t';
        }
        csvFile << "Nonzero Count\tDuration\n";
        for (const auto& logEntry : m_ReadingTensorBinary)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_WritingTensorBinary.empty())
    {
        csvFile << "Operation: Writing Tensor To A Binary File\n";
        unsigned long maxDim = 0;
        for (auto& logEntry : m_WritingTensorBinary)
        {
            std::vector<std::string> splittedString = split(logEntry, '|');
            maxDim = std::max(maxDim, std::stoul(splittedString[0]));
            logEntry = splittedString[1];
        }
        csvFile << "Name\t";
        for (int i = 1; i <= maxDim; ++i)
        {
            csvFile << "Dimension: " << i << '\t';
        }
        csvFile << "Nonzero Count\tDuration\n";
        for (const auto& logEntry : m_WritingTensorBinary)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_TensorOrdering.empty())
    {
        // Metrics
        csvFile << "Operation: Tensor Ordering\n";
        unsigned long maxDim = 0;
        for (auto& logEntry : m_TensorOrdering)
        {
            std::vector<std::string> splittedString = split(logEntry, '|');
            maxDim = std::max(maxDim, std::stoul(splittedString[0]));
            logEntry = splittedString[1];
        }
        csvFile << "Ordering Name\tTensor Name\tOrdering Price\t";
        for (int i = 1; i <= maxDim; ++i)
        {
            csvFile << "Dimension: " << i << '\t';
        }
        csvFile << "Tensor NNZ Count\tDuration\t";
        if (!m_TensorOrderingResults.empty())
        {
            for (const auto& result: m_TensorOrderingResults[0])
            {
                csvFile << result.first << '\t';
            }
        }
        csvFile << '\n';

        // Results
        for (size_t i = 0; i < m_TensorOrdering.size(); ++i)
        {
            csvFile << m_TensorOrdering[i] << '\t';
            if (!m_TensorOrderingResults.empty())
            {
                for (const auto& result: m_TensorOrderingResults[i])
                {
                    csvFile << result.second.second << '\t';
                }
            }
            csvFile << '\n';
        }

        csvFile << "\n";
    }

    if (!m_ReadingTensorOrdering.empty())
    {
        csvFile << "Operation: Reading Tensor Ordering From The Binary File\n";
        unsigned long maxDim = 0;
        for (auto& logEntry : m_ReadingTensorOrdering)
        {
            std::vector<std::string> splittedString = split(logEntry, '|');
            maxDim = std::max(maxDim, std::stoul(splittedString[0]));
            logEntry = splittedString[1];
        }
        csvFile << "Ordering Name\tTensor Name\tOrdering Price\t";
        for (int i = 1; i <= maxDim; ++i)
        {
            csvFile << "Dimension: " << i << '\t';
        }
        csvFile << "Tensor NNZ Count\tDuration\n";
        for (const auto& logEntry : m_ReadingTensorOrdering)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_WritingTensorOrdering.empty())
    {
        csvFile << "Operation: Writing Tensor Ordering To A Binary File\n";
        unsigned long maxDim = 0;
        for (auto& logEntry : m_WritingTensorOrdering)
        {
            std::vector<std::string> splittedString = split(logEntry, '|');
            maxDim = std::max(maxDim, std::stoul(splittedString[0]));
            logEntry = splittedString[1];
        }
        csvFile << "Ordering Name\tTensor Name\tOrdering Price\t";
        for (int i = 1; i <= maxDim; ++i)
        {
            csvFile << "Dimension: " << i << '\t';
        }
        csvFile << "Tensor NNZ Count\tDuration\n";
        for (const auto& logEntry : m_WritingTensorOrdering)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_ConstructingOrderedTensor.empty())
    {
        csvFile << "Operation: Construction of Ordered Tensor\n";
        unsigned long maxDim = 0;
        for (auto& logEntry : m_ConstructingOrderedTensor)
        {
            std::vector<std::string> splittedString = split(logEntry, '|');
            maxDim = std::max(maxDim, std::stoul(splittedString[0]));
            logEntry = splittedString[1];
        }
        csvFile << "Ordering Name\tTensor Name\tOrdering Price\t";
        for (int i = 1; i <= maxDim; ++i)
        {
            csvFile << "Dimension: " << i << '\t';
        }
        csvFile << "Tensor NNZ Count\tDuration\n";
        for (const auto& logEntry : m_ConstructingOrderedTensor)
        {
            csvFile << logEntry << "\n";
        }
        csvFile << "\n";
    }

    if (!m_RunningMatrixKernel.empty())
    {
        csvFile << "Operation: Running Matrix Kernel\n";
        csvFile << "Kernel Name\tMatrix Name\tOrdering Name\tScheduling Policy\tChunk Size\t";

        for (auto i = m_MatrixKernelThreadCounts.begin(); i != m_MatrixKernelThreadCounts.end(); ++i)
        {
            csvFile << "Thread Number: " << std::to_string(*i) << '\t';
        }
        if (!m_MatrixOrderingResults.empty())
        {
            for (const auto& result: m_MatrixOrderingResults[0])
            {
                csvFile << result.first << '\t';
            }
        }
        csvFile << '\n';
        for (int a = 0; a != m_RunningMatrixKernel.size(); ++a)
        {
            csvFile << m_RunningMatrixKernel[a] << "\t";
            for (auto i = m_MatrixKernelThreadCounts.begin(); i != m_MatrixKernelThreadCounts.end(); ++i)
            {
                bool found = false;
                for (int j = 0; j != m_ThreadCountsForEachMatrixKernel[a].size(); ++j)
                {
                    if (*i == m_ThreadCountsForEachMatrixKernel[a][j])
                    {
                        csvFile << std::to_string(m_DurationsForEachMatrixKernel[a][j]) << '\t';
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    csvFile << "-\t";
                }
            }
            if (!m_MatrixOrderingResults.empty())
            {
                for (const auto& result: m_MatrixOrderingResults[a])
                {
                    csvFile << result.second.second << '\t';
                }
            }
            csvFile << '\n';
        }
        csvFile << "\n";
    }

    if (!m_RunningTensorKernel.empty())
    {
        csvFile << "Operation: Running Tensor Kernel\n";
        csvFile << "Kernel Name\tTensor Name\tOrdering Name\tScheduling Policy\tChunk Size\t";

        for (auto i = m_TensorKernelThreadCounts.begin(); i != m_TensorKernelThreadCounts.end(); ++i)
        {
            csvFile << "Thread Number: " << std::to_string(*i) << '\t';
        }
        if (!m_TensorOrderingResults.empty())
        {
            for (const auto& result: m_TensorOrderingResults[0])
            {
                csvFile << result.first << '\t';
            }
        }
        csvFile << '\n';
        for (int a = 0; a != m_RunningTensorKernel.size(); ++a)
        {
            csvFile << m_RunningTensorKernel[a] << "\t";
            for (auto i = m_TensorKernelThreadCounts.begin(); i != m_TensorKernelThreadCounts.end(); ++i)
            {
                bool found = false;
                for (int j = 0; j != m_ThreadCountsForEachTensorKernel[a].size(); ++j)
                {
                    if (*i == m_ThreadCountsForEachTensorKernel[a][j])
                    {
                        csvFile << std::to_string(m_DurationsForEachTensorKernel[a][j]) << '\t';
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    csvFile << "-\t";
                }
            }
            if (!m_TensorOrderingResults.empty())
            {
                for (const auto& result: m_TensorOrderingResults[a])
                {
                    csvFile << result.second.second << '\t';
                }
            }
            csvFile << '\n';
        }
        csvFile << "\n";
    }

    if (!m_MatrixProcessing.empty())
    {
        csvFile << "Operation: Processing Matrix\n";
        csvFile << "Filename\tMatrix Name\tOrdering Name\tNo Bins\tNo Empty Bins\tMean NNZ\tMedian NNZ\tGeo Mean NNZ\tMax Bandwidth\tAvg Bandwidth\tNormalize Bandwidth\tMax Row Span\t"
                   "Average Row Span\tNormalized Row Span\tMax Col Span\tAverage Col Span\tNormalized Col Span\tDuration\n";
        for (const auto& processing: m_MatrixProcessing)
        {
            csvFile << processing << '\n';
        }
        csvFile << '\n';
    }

    if (!m_TensorProcessing.empty())
    {
        csvFile << "Operation: Processing Tensor\n";
        csvFile << "Filename\tTensor Name\tOrdering Name\tNo Bins\tNo Empty Bins\tMean NNZ\tMedian NNZ\tGeo Mean NNZ\tMax X Span\tMax Y Span\tMax Z Span\t"
                   "Average X Span\tAverage Y Span\tAverage Z Span\tNormalized X Span\tNormalized Y Span\tNormalized Z Span\t"
                   "X Fiber Counts\tY Fiber Counts\tZ Fiber Counts\tX Single NNZ Fiber Counts\tY Single NNZ Fiber Counts\tZ Single NNZFiber Counts\tDuration\n";
        for (const auto& processing: m_TensorProcessing)
        {
            csvFile << processing << '\n';
        }
        csvFile << '\n';
    }

    if (!m_Others.empty())
    {
        csvFile << "Section devoted for explicit logs\n";
        for (const auto& other: m_Others)
        {
            csvFile << other << '\n';
        }
        csvFile << '\n';
    }

    csvFile << "©SparCity 2024\n";

    csvFile.close();
}

void SparseVizLogger::logReadingMatrixMarket(SparseMatrix *matrix, double duration)
{
#pragma omp critical
        {
            std::cout   << matrix->getName() << " with dimensions " << matrix->getRowCount() 
                        <<  " x " << matrix->getColCount() << " and " << matrix->getNNZCount() 
                        << " nnzs : reading matrix from the market file took " << duration << " seconds." << std::endl;
            m_ReadingMatrixMarket.push_back(matrix->getName() + '\t' + std::to_string(matrix->getRowCount()) + '\t' + std::to_string(matrix->getColCount()) + '\t' + std::to_string(matrix->getNNZCount()) + '\t' + boolToString(matrix->isSymmetric())
                                            + '\t' + boolToString(matrix->isPatternSymmetric()) + '\t' + boolToString(matrix->isSquare()) + '\t' + std::to_string(duration));
        }
}

void SparseVizLogger::logReadingMatrixBinary(SparseMatrix *matrix, double duration)
{
#pragma omp critical
    {
        std::cout   << matrix->getName() << " with dimensions " << matrix->getRowCount() 
                    <<  " x " << matrix->getColCount() << " and " << matrix->getNNZCount() 
                    << " nnzs : reading matrix from the binary file took " << duration << " seconds." << std::endl;        m_ReadingMatrixBinary.push_back(matrix->getName() + '\t' + std::to_string(matrix->getRowCount()) + '\t' + std::to_string(matrix->getColCount()) + '\t' + std::to_string(matrix->getNNZCount()) + '\t' + boolToString(matrix->isSymmetric())
                                        + '\t' + boolToString(matrix->isPatternSymmetric()) + '\t' + boolToString(matrix->isSquare()) + '\t' + std::to_string(duration));
    }
}

void SparseVizLogger::logWritingMatrixBinary(SparseMatrix *matrix, double duration)
{
#pragma omp critical
    {
        std::cout << matrix->getName() << ": writing matrix to a binary file took " << duration << " seconds." << std::endl;
        m_WritingMatrixBinary.push_back(matrix->getName() + '\t' + std::to_string(matrix->getRowCount()) + '\t' + std::to_string(matrix->getColCount()) + '\t' + std::to_string(matrix->getNNZCount()) + '\t' + boolToString(matrix->isSymmetric())
                                        + '\t' + boolToString(matrix->isPatternSymmetric()) + '\t' + boolToString(matrix->isSquare()) + '\t' + std::to_string(duration));
    }
}

void SparseVizLogger::logMatrixOrdering(MatrixOrdering *ordering, double duration, SparseVizPerformance::OperationResults operationResults)
{
#pragma omp critical
    {
        std::cout << ordering->getOrderingName() << " on " << ordering->getMatrix().getName() << " took " << duration << " seconds." << std::endl;
        m_MatrixOrdering.push_back(ordering->getOrderingName() + '\t' + ordering->getMatrix().getName() + '\t' + std::to_string(ordering->getOrderingPrice()) + '\t' +
                                          boolToString(ordering->hasRectangularSupport()) + '\t' + boolToString(ordering->hasPatternUnsymmetricSupport())
                                          + '\t' + std::to_string(ordering->getMatrix().getRowCount()) + '\t' + std::to_string(ordering->getMatrix().getColCount()) + '\t' + std::to_string(ordering->getMatrix().getNNZCount()) + '\t' +
                                          boolToString(ordering->getMatrix().isSymmetric()) + '\t' + boolToString(ordering->getMatrix().isPatternSymmetric()) + '\t' + boolToString(ordering->getMatrix().isSquare()) + '\t' + std::to_string(duration));
        if (!operationResults.empty())
        {
            m_MatrixOrderingResults.push_back(operationResults);
        }
    }
}

void SparseVizLogger::logReadingMatrixOrdering(MatrixOrdering *ordering, double duration)
{
#pragma omp critical
    {
        std::cout << "RMO: " << ordering->getOrderingName() << " on " << ordering->getMatrix().getName() << " has been read from the binary file in " << duration << " seconds." << std::endl;
        m_ReadingMatrixOrdering.push_back(ordering->getOrderingName() + '\t' + ordering->getMatrix().getName() + '\t' + std::to_string(ordering->getOrderingPrice()) + '\t' +
                                          boolToString(ordering->hasRectangularSupport()) + '\t' + boolToString(ordering->hasPatternUnsymmetricSupport())
                                          + '\t' + std::to_string(ordering->getMatrix().getRowCount()) + '\t' + std::to_string(ordering->getMatrix().getColCount()) + '\t' + std::to_string(ordering->getMatrix().getNNZCount()) + '\t' +
                                          boolToString(ordering->getMatrix().isSymmetric()) + '\t' + boolToString(ordering->getMatrix().isPatternSymmetric()) + '\t' + boolToString(ordering->getMatrix().isSquare()) + '\t' + std::to_string(duration));
    }
}

void SparseVizLogger::logWritingMatrixOrdering(MatrixOrdering *ordering, double duration)
{
#pragma omp critical
    {
        std::cout << "WMO: " << ordering->getOrderingName() << " on " << ordering->getMatrix().getName() << " has been written to a binary file in " << duration << " seconds." << std::endl;
        m_WritingMatrixOrdering.push_back(ordering->getOrderingName() + '\t' + ordering->getMatrix().getName() + '\t' + std::to_string(ordering->getOrderingPrice()) + '\t' +
                                          boolToString(ordering->hasRectangularSupport()) + '\t' + boolToString(ordering->hasPatternUnsymmetricSupport())
                                          + '\t' + std::to_string(ordering->getMatrix().getRowCount()) + '\t' + std::to_string(ordering->getMatrix().getColCount()) + '\t' + std::to_string(ordering->getMatrix().getNNZCount()) + '\t' +
                                          boolToString(ordering->getMatrix().isSymmetric()) + '\t' + boolToString(ordering->getMatrix().isPatternSymmetric()) + '\t' + boolToString(ordering->getMatrix().isSquare()) + '\t' + std::to_string(duration));
    }
}

void SparseVizLogger::logConstructingOrderedMatrix(MatrixOrdering *ordering, double duration)
{
#pragma omp critical
    {
        std::cout << "Constructing a " << ordering->getOrderingName() << " ordered " << ordering->getMatrix().getName() << " took " << duration << " seconds." << std::endl;
        m_ConstructingOrderedMatrix.push_back(ordering->getOrderingName() + '\t' + ordering->getMatrix().getName() + '\t' + std::to_string(ordering->getOrderingPrice()) + '\t' +
                                   boolToString(ordering->hasRectangularSupport()) + '\t' + boolToString(ordering->hasPatternUnsymmetricSupport())
                                   + '\t' + std::to_string(ordering->getMatrix().getRowCount()) + '\t' + std::to_string(ordering->getMatrix().getColCount()) + '\t' + std::to_string(ordering->getMatrix().getNNZCount()) + '\t' +
                                   boolToString(ordering->getMatrix().isSymmetric()) + '\t' + boolToString(ordering->getMatrix().isPatternSymmetric()) + '\t' + boolToString(ordering->getMatrix().isSquare()) + '\t' + std::to_string(duration));
    }
}

void SparseVizLogger::logReadingTensorMarket(SparseTensor *tensor, double duration)
{
#pragma omp critical
    {
        std::string logString;
        std::cout << tensor->getName() + ": reading ";
        logString += tensor->getName() + '\t';

        const vType* dims = tensor->getDims();
        for (int i = 0; i != tensor->getOrder() - 1; ++i)
        {
            std::cout << dims[i] << " x ";
            logString += std::to_string(dims[i]) + '\t';
        }
        if (tensor->getOrder() != 0)
        {
            std::cout << dims[tensor->getOrder() - 1];
            logString += std::to_string(dims[tensor->getOrder() - 1]) + '\t';
        }
        std::cout << " tensor with " << tensor->getNNZ() << " nonzeros from the market file took " << duration << " seconds." << std::endl;
        logString += std::to_string(tensor->getNNZ()) + '\t' + std::to_string(duration);

        m_ReadingTensorMarket.push_back(std::to_string(tensor->getOrder()) + '|' + logString);
    }
}

void SparseVizLogger::logReadingTensorBinary(SparseTensor *tensor, double duration)
{
#pragma omp critical
    {
        std::string logString;
        std::cout << tensor->getName() + ": reading ";
        logString += tensor->getName() + '\t';

        const vType* dims = tensor->getDims();
        for (int i = 0; i != tensor->getOrder() - 1; ++i)
        {
            std::cout << dims[i] << " x ";
            logString += std::to_string(dims[i]) + '\t';
        }
        if (tensor->getOrder() != 0)
        {
            std::cout << dims[tensor->getOrder() - 1];
            logString += std::to_string(dims[tensor->getOrder() - 1]) + '\t';
        }
        std::cout << " tensor with " << tensor->getNNZ() << " nonzeros from the binary file took " << duration << " seconds." << std::endl;
        logString += std::to_string(tensor->getNNZ()) + '\t' + std::to_string(duration);

        m_ReadingTensorBinary.push_back(std::to_string(tensor->getOrder()) + '|' + logString);
    }
}

void SparseVizLogger::logWritingTensorBinary(SparseTensor *tensor, double duration)
{
#pragma omp critical
    {
        std::string logString;
        std::cout << tensor->getName() + ": writing ";
        logString += tensor->getName() + '\t';

        const vType* dims = tensor->getDims();
        for (int i = 0; i != tensor->getOrder() - 1; ++i)
        {
            std::cout << dims[i] << " x ";
            logString += std::to_string(dims[i]) + '\t';
        }
        if (tensor->getOrder() != 0)
        {
            std::cout << dims[tensor->getOrder() - 1];
            logString += std::to_string(dims[tensor->getOrder() - 1]) + '\t';
        }
        std::cout << " tensor with " << tensor->getNNZ() << " nonzeros to a binary file took " << duration << " seconds." << std::endl;
        logString += std::to_string(tensor->getNNZ()) + '\t' + std::to_string(duration);

        m_WritingTensorBinary.push_back(std::to_string(tensor->getOrder()) + '|' + logString);
    }
}

void SparseVizLogger::logTensorOrdering(TensorOrdering *ordering, double duration, SparseVizPerformance::OperationResults operationResults)
{
#pragma omp critical
    {
        std::cout << ordering->getOrderingName() << " on " << ordering->getTensor().getName() << " took " << duration << " seconds." << std::endl;

        std::string logString;
        logString += ordering->getOrderingName() + '\t' + ordering->getTensor().getName() + '\t' + std::to_string(ordering->getOrderingPrice()) + '\t';

        const vType* dims = ordering->getTensor().getDims();
        for (int i = 0; i != ordering->getTensor().getOrder(); ++i)
        {
            logString += std::to_string(dims[i]) + '\t';
        }
        logString += std::to_string(ordering->getTensor().getNNZ()) + '\t' + std::to_string(duration);

        m_TensorOrdering.push_back(std::to_string(ordering->getTensor().getOrder()) + '|' + logString);

        if (!operationResults.empty())
        {
            m_TensorOrderingResults.push_back(operationResults);
        }
    }
}

void SparseVizLogger::logReadingTensorOrdering(TensorOrdering *ordering, double duration)
{
#pragma omp critical
    {
        std::cout <<  "RTO: " << ordering->getOrderingName() << " on " << ordering->getTensor().getName() << " has been read from the binary file in " << duration << " seconds." << std::endl;

        std::string logString;
        logString += ordering->getOrderingName() + '\t' + ordering->getTensor().getName() + '\t' + std::to_string(ordering->getOrderingPrice()) + '\t';

        const vType* dims = ordering->getTensor().getDims();
        for (int i = 0; i != ordering->getTensor().getOrder(); ++i)
        {
            logString += std::to_string(dims[i]) + '\t';
        }
        logString += std::to_string(ordering->getTensor().getNNZ()) + '\t' + std::to_string(duration);

        m_ReadingTensorOrdering.push_back(std::to_string(ordering->getTensor().getOrder()) + '|' + logString);
    }
}

void SparseVizLogger::logWritingTensorOrdering(TensorOrdering *ordering, double duration)
{
#pragma omp critical
    {
        std::cout << "WTO: " << ordering->getOrderingName() << " on " << ordering->getTensor().getName() << " has been written to a binary file in " << duration << " seconds." << std::endl;

        std::string logString;
        logString += ordering->getOrderingName() + '\t' + ordering->getTensor().getName() + '\t' + std::to_string(ordering->getOrderingPrice()) + '\t';

        const vType* dims = ordering->getTensor().getDims();
        for (int i = 0; i != ordering->getTensor().getOrder(); ++i)
        {
            logString += std::to_string(dims[i]) + '\t';
        }
        logString += std::to_string(ordering->getTensor().getNNZ()) + '\t' + std::to_string(duration);

        m_WritingTensorOrdering.push_back(std::to_string(ordering->getTensor().getOrder()) + '|' + logString);
    }
}

void SparseVizLogger::logConstructingOrderedTensor(TensorOrdering* ordering, double duration)
{
#pragma omp critical
    {
        std::cout << "Constructing a " << ordering->getOrderingName() << " ordered " << ordering->getTensor().getName() << " took " << duration << " seconds." << std::endl;

        std::string logString;
        logString += ordering->getOrderingName() + '\t' + ordering->getTensor().getName() + '\t' + std::to_string(ordering->getOrderingPrice()) + '\t';

        const vType* dims = ordering->getTensor().getDims();
        for (int i = 0; i != ordering->getTensor().getOrder(); ++i)
        {
            logString += std::to_string(dims[i]) + '\t';
        }
        logString += std::to_string(ordering->getTensor().getNNZ()) + '\t' + std::to_string(duration);

        m_ConstructingOrderedTensor.push_back(std::to_string(ordering->getTensor().getOrder()) + '|' + logString);
    }
}

void SparseVizLogger::logRunningMatrixKernel(const KernelResult& kernelResult, MatrixOrdering* ordering, SparseVizPerformance::OperationResults operationResults)
{
#pragma omp critical
    {
        double averageTimeTook = 0;
        for (int i = 0; i != kernelResult.durations.size(); ++i)
        {
            averageTimeTook += kernelResult.durations[i];
        }
        averageTimeTook /= kernelResult.durations.size();

        std::string threadNosForTerminal;
        for (int i = 0; i != kernelResult.threadCounts.size() - 1; ++i)
        {
            threadNosForTerminal += std::to_string(kernelResult.threadCounts[i]) + '-';
            m_MatrixKernelThreadCounts.insert(kernelResult.threadCounts[i]);
        }
        if (kernelResult.threadCounts.size() > 0)
        {
            threadNosForTerminal += std::to_string(kernelResult.threadCounts.back());
            m_MatrixKernelThreadCounts.insert(kernelResult.threadCounts.back());
        }

        std::cout << kernelResult.kernelName << " on " << ordering->getMatrix().getName() << " with " << ordering->getOrderingName() << ": took " << averageTimeTook << " seconds on average with thread no's: " << threadNosForTerminal << std::endl;

        m_RunningMatrixKernel.push_back(kernelResult.kernelName + '\t' + ordering->getMatrix().getName() + '\t' + ordering->getOrderingName() + '\t' + kernelResult.schedulingPolicy + '\t' + std::to_string(kernelResult.chunkSize));
        m_ThreadCountsForEachMatrixKernel.push_back(kernelResult.threadCounts);
        m_DurationsForEachMatrixKernel.push_back(kernelResult.durations);
        if (!operationResults.empty())
        {
            m_MatrixKernelResults.push_back(operationResults);
        }
    }
}

void SparseVizLogger::logRunningGPUMatrixKernel(const GPUKernelResult &kernelResult, MatrixOrdering *ordering)
{
#pragma omp critical
    {
        double averageTimeTook = 0;
        for (int i = 0; i != kernelResult.durations.size(); ++i)
        {
            averageTimeTook += kernelResult.durations[i];
        }
        averageTimeTook /= kernelResult.durations.size();

        std::cout << "[GPU] " << kernelResult.kernelName << " on " << ordering->getMatrix().getName() << " with " << ordering->getOrderingName() << ": took " << averageTimeTook << " seconds on average." << std::endl;
    }
}

void SparseVizLogger::logRunningTensorKernel(const KernelResult& kernelResult, TensorOrdering* ordering, SparseVizPerformance::OperationResults operationResults)
{
#pragma omp critical
    {
        double averageTimeTook = 0;
        for (int i = 0; i != kernelResult.durations.size(); ++i)
        {
            averageTimeTook += kernelResult.durations[i];
        }
        averageTimeTook /= kernelResult.durations.size();

        std::string threadNosForTerminal;
        for (int i = 0; i != kernelResult.threadCounts.size() - 1; ++i)
        {
            threadNosForTerminal += std::to_string(kernelResult.threadCounts[i]) + '-';
            m_TensorKernelThreadCounts.insert(kernelResult.threadCounts[i]);
        }
        if (kernelResult.threadCounts.size() > 0)
        {
            threadNosForTerminal += std::to_string(kernelResult.threadCounts.back());
            m_TensorKernelThreadCounts.insert(kernelResult.threadCounts.back());
        }

        std::cout << kernelResult.kernelName << " on " << ordering->getTensor().getName() << " with " << ordering->getOrderingName() << ": took " << averageTimeTook << " seconds on average with thread no's: " << threadNosForTerminal << std::endl;

        m_RunningTensorKernel.push_back(kernelResult.kernelName + '\t' + ordering->getTensor().getName() + '\t' + ordering->getOrderingName() + '\t' + kernelResult.schedulingPolicy + '\t' + std::to_string(kernelResult.chunkSize));
        m_ThreadCountsForEachTensorKernel.push_back(kernelResult.threadCounts);
        m_DurationsForEachTensorKernel.push_back(kernelResult.durations);
        if (!operationResults.empty())
        {
            m_TensorKernelResults.push_back(operationResults);
        }
    }
}

void SparseVizLogger::logRunningGPUTensorKernel(const GPUKernelResult &kernelResult, TensorOrdering *ordering)
{
#pragma omp critical
    {
        double averageTimeTook = 0;
        for (int i = 0; i != kernelResult.durations.size(); ++i)
        {
            averageTimeTook += kernelResult.durations[i];
        }
        averageTimeTook /= kernelResult.durations.size();

        std::cout << "[GPU] " << kernelResult.kernelName << " on " << ordering->getTensor().getName() << " with " << ordering->getOrderingName() << ": took " << averageTimeTook << " seconds on average." << std::endl;
    }
}

void SparseVizLogger::logMatrixProcessing(const std::string &filename, const Statistic &stat, double duration)
{
#pragma omp critical
    {
        m_MatrixProcessing.push_back(filename + '\t' + stat.matrixName + '\t' + stat.orderingName + '\t' + std::to_string(stat.no_bins) + '\t' + std::to_string(stat.no_empty_bins) + '\t' + std::to_string(stat.mean_nnz) + '\t' + std::to_string(stat.median_nnz) + '\t'
                                     + std::to_string(stat.geo_mean_nnz) + '\t' + std::to_string(stat.maxBandwidth) + '\t' + std::to_string(stat.avgBandwidth) + '\t' + std::to_string(stat.normalizedBandwidth) + '\t' + std::to_string(stat.maxRowSpan) + '\t' + std::to_string(stat.avgRowSpan) + '\t'
                                     + std::to_string(stat.normalizedRowSpan) + '\t' + std::to_string(stat.maxColSpan) + '\t' + std::to_string(stat.avgColSpan) + '\t' + std::to_string(stat.normalizedColSpan) + '\t' + std::to_string(duration)
        );
    }
}

void SparseVizLogger::logTensorProcessing(const std::string &filename, const TStatistic &stat, double duration)
{
#pragma omp critical
    {
        m_TensorProcessing.push_back(filename + '\t' + stat.tensorName + '\t' + stat.orderingName + '\t' + std::to_string(stat.no_bins) + '\t' + std::to_string(stat.no_empty_bins) + '\t' + std::to_string(stat.mean_nnz) + '\t' + std::to_string(stat.median_nnz) + '\t'
                                     + std::to_string(stat.geo_mean_nnz) + '\t' + std::to_string(stat.maxSpan[0]) + '\t' + std::to_string(stat.maxSpan[1]) + '\t' + std::to_string(stat.maxSpan[2]) + '\t' + std::to_string(stat.avgSpan[0]) + '\t' + std::to_string(stat.avgSpan[1]) + '\t'
                                     + std::to_string(stat.avgSpan[2]) + '\t' + std::to_string(stat.normSpan[0]) + '\t' + std::to_string(stat.normSpan[1]) + '\t' + std::to_string(stat.normSpan[2]) + '\t' + std::to_string(stat.fiberCounts[0]) + '\t' + std::to_string(stat.fiberCounts[1]) + '\t'
                                     + std::to_string(stat.fiberCounts[2]) + '\t' + std::to_string(stat.singleNNZfiberCounts[0]) + '\t' + std::to_string(stat.singleNNZfiberCounts[1]) + '\t' + std::to_string(stat.singleNNZfiberCounts[2]) + '\t' + std::to_string(duration)
        );
    }
}

void SparseVizLogger::makeSilentLog(std::string message, double time)
{
#pragma omp critical
    {
        if(time > 0)
        {
            std::cout << message << ": " << std::setprecision(3) << time << " secs" << std::endl;
        }
        else
        {
            std::cout << message << std::endl;
        }
    }
}

void SparseVizLogger::makeExplicitLog(std::string message, double time)
{
#pragma omp critical
    {
        if(time > 0)
        {
            std::cout << message << ": " << std::setprecision(3) << time << " secs" << std::endl;
            m_Others.push_back(message + '\t' + std::to_string(time));
        }
        else
        {
            std::cout << message << std::endl;
            m_Others.push_back(message);
        }
    }
}

std::string SparseVizLogger::boolToString(bool boolean)
{
    if (boolean)
    {
        return "True";
    }
    else
    {
        return "False";
    }
}

std::string SparseVizLogger::determineArchitecture()
{
#if defined(__x86_64__) || defined(__i386__)
    return "x86";
#elif defined(__arm__) || defined(__aarch64__)
    return "ARM";
#else
    return "UNKNOWN";
#endif
}

std::string SparseVizLogger::determineCompiler()
{
#if defined(__clang__)
    return "Clang/LLVM";
#elif defined(__GNUC__)
    return "GNU GCC";
#elif defined(_MSC_VER)
    return "Microsoft Visual C++";
#else
    return "UNKNOWN";
#endif
}

std::string SparseVizLogger::determineOS()
{
#if defined(_WIN32) || defined(_WIN64)
    return "Windows";
#elif defined(__APPLE__) || defined(__MACH__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#elif defined(__unix__)
    return "Unix";
#elif defined(__FreeBSD__)
    return "FreeBSD";
#else
    return "UNKNOWN";
#endif
}

bool SparseVizLogger::isOMPEnabled()
{
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

bool SparseVizLogger::isCudaEnabled()
{
#ifdef CUDA_ENABLED
    return true;
#else
    return false;
#endif
}

void SparseVizLogger::logTest(std::string line)
{
    SparseVizTest::getSparseVizTester()->logTest(line);
}
