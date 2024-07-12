#include "SparseVizEngine.h"
#include "MatrixVisualizer.h"
#include "TensorVisualizer.h"
#include "SparseVizLogger.h"
#include "SparseVizIO.h"


SparseVizEngine::~SparseVizEngine()
{
    for (const auto& orderingPtr: m_MatrixOrderings)
    {
        delete orderingPtr;
    }

    for (const auto& orderingPtr: m_TensorOrderings)
    {
        delete orderingPtr;
    }

    for (const auto& matrixPtr: m_Matrices)
    {
        delete matrixPtr;
    }

    for (const auto& tensorPtr: m_Tensors)
    {
        delete tensorPtr.tensor;
    }

    for (const auto& kernelPointer: m_MatrixKernels)
    {
        delete kernelPointer;
    }

    for (const auto& kernelPointer: m_TensorKernels)
    {
        delete kernelPointer;
    }
}

void SparseVizEngine::runEngine()
{
    if (ZOO_TYPE == "MAT")
    {
        // Orderings are listed for each matrix.

        std::unordered_map<std::string, std::vector<MatrixOrdering*>> matrixOrderingPermutations = this->getMatrixOrderingPermutations();
        this->runMatrixKernels();
#ifdef CUDA_ENABLED
        this->runGPUMatrixKernels();
#endif
        #pragma omp parallel for schedule(dynamic,1)
        for (int i = 0; i < matrixOrderingPermutations.size(); ++i)
        {
            auto iter = matrixOrderingPermutations.begin();
            advance(iter, i);
            visualizeMatrixOrderings(iter->second.data(), iter->second.size());
        }
    }
    else if (ZOO_TYPE == "MATORD")
    {
        // Matrices are listed for each ordering.

        std::unordered_map<std::string, std::vector<MatrixOrdering*>> matrixPermutations = this->getMatrixPermutations();
        this->runMatrixKernels();
#ifdef CUDA_ENABLED
        this->runGPUMatrixKernels();
#endif
        #pragma omp parallel for schedule(dynamic,1)
        for (int i = 0; i < matrixPermutations.size(); ++i)
        {
            auto iter = matrixPermutations.begin();
            advance(iter, i);
            visualizeMatrices(iter->second.data(), iter->second.size());
        }
    }
    else if (ZOO_TYPE == "TENS")
    {
        // Orderings are listed for each tensor.

        std::unordered_map<std::string, std::vector<TensorOrdering*>> tensorOrderingPermutations = this->getTensorOrderingPermutations();
        this->runTensorKernels();
#ifdef CUDA_ENABLED
        this->runGPUTensorKernels();
#endif

        #pragma omp parallel for schedule(dynamic,1)
        for (int i = 0; i < tensorOrderingPermutations.size(); ++i)
        {
            auto iter = tensorOrderingPermutations.begin();
            advance(iter, i);
            visualizeTensorOrderings(iter->second.data(), iter->second.size());
        }
    }
    else if (ZOO_TYPE == "TENSORD")
    {
        std::unordered_map<std::string, std::vector<TensorOrdering*>> tensorPermutations = this->getTensorPermutations();
        this->runTensorKernels();
#ifdef CUDA_ENABLED
        this->runGPUTensorKernels();
#endif

        if (ZOO_TYPE == "TENSORD")
        {
            // Tensors are listed for each ordering.
            #pragma omp parallel for schedule(dynamic,1)
            for (int i = 0; i < tensorPermutations.size(); ++i)
            {
                auto iter = tensorPermutations.begin();
                advance(iter, i);
                visualizeTensors(iter->second.data(), iter->second.size());
            }
        }
    }
    else if (ZOO_TYPE == "FULLTENSOR")
    {
        std::unordered_map<std::string, std::vector<TensorOrdering*>> tensorPermutations = this->getTensorPermutations();
        this->runTensorKernels();
#ifdef CUDA_ENABLED
        this->runGPUTensorKernels();
#endif
    if (ZOO_TYPE == "FULLTENSOR")
        {
            #pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < tensorPermutations.size(); ++i)
            {
                auto iter = tensorPermutations.begin();
                advance(iter, i);
                #pragma omp parallel for schedule(dynamic, 1)
                for(int j = 0; j < iter->second.size(); j++)
                {
                    visualizeFullSparseTensor(iter->second.data()[j]);    
                }      
            }
        }
    }
}

void SparseVizEngine::addMatrix(const std::string& matrixFilePath)
{
    for (const auto& matrixPath: m_MatrixFilePaths)
    {
        if (matrixPath == matrixFilePath)
        {
            return;
        }
    }

    m_MatrixFilePaths.push_back(matrixFilePath);
}

void SparseVizEngine::addTensor(const std::string &tensorFilePath, const std::vector<vType>& new_active_modes)
{
    for (int i = 0; i != m_Tensors.size(); ++i)
    {
        if (m_Tensors[i].filePath == tensorFilePath)
        {
            for (int j = 0; j != m_Tensors[i].active_modes.size(); ++j)
            {
                if (m_Tensors[i].active_modes[j][0] == new_active_modes[0] && m_Tensors[i].active_modes[j][1] == new_active_modes[1] && m_Tensors[i].active_modes[j][2] == new_active_modes[2])
                {
                    return;
                }
            }
            m_Tensors[i].active_modes.push_back(new_active_modes);
            return;
        }
    }

    m_Tensors.emplace_back(tensorFilePath, new_active_modes);
}

void SparseVizEngine::addMatrixOrdering(const std::string& orderingClassName, const std::string& orderingName, const std::string& orderingParameters)
{
    for (const auto& matrix: m_Matrices)
    {
#ifndef RABBIT_AVAILABLE
        std::vector<std::string> parameters = split(orderingParameters, '/');
        if (orderingClassName == "Patoh" && orderingParameters.size() > 5 && parameters[5] == "Rabbit")
        {
            std::cout << "SKIPPING ORDERING: " << orderingName << std::endl;
            std::cout << orderingClassName << " requires " << parameters[5] << " ordering, which is not available in your ordering set." << std::endl;
            continue;
        }
#endif
        MatrixOrdering* orderingClass = matrixOrderingFactory(*matrix, orderingClassName, orderingName, orderingParameters);
        if (orderingClass == nullptr)
        {
            std::cout << "SKIPPING ORDERING: " << orderingName << std::endl;
            std::cout << orderingClassName << " is not found in your available ordering set." << std::endl;
            continue;
        }
        m_MatrixOrderings.push_back(orderingClass);
    }
}

void SparseVizEngine::addTensorOrdering(const std::string& orderingClassName, const std::string& orderingName, const std::string& orderingParameters, const std::string& zooType)
{
    for (const auto& tensor: m_Tensors)
    {
        if(zooType == "TENS" || zooType == "TENSORD")
        {
            for(const auto& active_modes : tensor.active_modes)
            {
                TensorOrdering* orderingClass = tensorOrderingFactory(*tensor.tensor, active_modes, orderingClassName, orderingName, orderingParameters);
                if (orderingClass == nullptr)
                {
                    std::cout << "SKIPPING ORDERING: " << orderingName << std::endl;
                    std::cout << orderingClassName << " is not found in your available ordering set." << std::endl;
                    continue;
                }
                m_TensorOrderings.push_back(orderingClass);
            }
        }
        else if(zooType == "FULLTENSOR")
        {
            std::vector<vType> active_modes;
            for(int i = 0; i < ((tensor).tensor)->getOrder(); i++)
            {
                active_modes.push_back(i);
            }
            TensorOrdering* orderingClass = tensorOrderingFactory(*tensor.tensor, active_modes, orderingClassName, orderingName, orderingParameters);
             if (orderingClass == nullptr)
             {
                std::cout << "SKIPPING ORDERING: " << orderingName << std::endl;
                std::cout << orderingClassName << " is not found in your available ordering set." << std::endl;
                continue;
             }
             m_TensorOrderings.push_back(orderingClass);
        }
        else
        {
            throw std::runtime_error("Unknown zoo type: " + zooType);
        }
    }
}

void SparseVizEngine::addMatrixKernel(const std::string &kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
{
    MatrixKernelFunction* kernel = this->matrixKernelFactory(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);

    if (!kernel)
    {
        std::cout << "SKIPPING KERNEL: " << kernelName << std::endl;
        std::cout << "The definition of the kernel: " << kernelName << " could not be found" << std::endl;
        return;
    }

    m_MatrixKernels.emplace_back(kernel);
}

#ifdef CUDA_ENABLED
void SparseVizEngine::addGPUMatrixKernel(const std::string& kernelClassName, const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, const std::vector<int>& sharedMemorySizes, const std::string& kernelParameters, int nRun, int nIgnore)
{
    MatrixGPUKernel* kernel = this->matrixGPUKernelFactory(kernelClassName, kernelName, gridSizes, blockSizes, sharedMemorySizes, kernelParameters, nRun, nIgnore);

    if (!kernel)
    {
        std::cout << "SKIPPING MATRIX GPU KERNEL: " << kernelName << std::endl;
        std::cout << "The definition of the Matrix GPU Kernel: " << kernelName << " could not be found" << std::endl;
        return;
    }

    m_MatrixGPUKernels.push_back(kernel);
}
#endif

#ifdef CUDA_ENABLED
void SparseVizEngine::addGPUTensorKernel(const std::string& kernelClassName, const std::string &kernelName, const std::vector<int> &gridSizes, const std::vector<int> &blockSizes, const std::vector<int>& sharedMemorySizes, const std::string& kernelParameters, int nRun, int nIgnore)
{
    TensorGPUKernel* kernel = this->tensorGPUKernelFactory(kernelClassName, kernelName, gridSizes, blockSizes, sharedMemorySizes, kernelParameters, nRun, nIgnore);

    if (!kernel)
    {
        std::cout << "SKIPPING TENSOR GPU KERNEL: " << kernelName << std::endl;
        std::cout << "The definition of the Tensor GPU Kernel: " << kernelName << " could not be found" << std::endl;
        return;
    }

    m_TensorGPUKernels.push_back(kernel);
}
#endif

void SparseVizEngine::addTensorKernel(const std::string &kernelName, const std::vector<int>& threadCounts, const std::string &schedulingPolicy, int chunkSize, int nRun, int nIgnore, const std::string& kernelParameters)
{
    TensorKernelFunction* kernel = this->tensorKernelFactory(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore, kernelParameters);

    if (!kernel)
    {
        std::cout << "SKIPPING KERNEL: " << kernelName << std::endl;
        std::cout << "The definition of the kernel: " << kernelName << " could not be found" << std::endl;
        return;
    }

    m_TensorKernels.emplace_back(kernel);
}

void SparseVizEngine::runMatrixKernels()
{
    if (!m_MatrixKernels.empty())
    {
        for (auto& orderingPtr: m_MatrixOrderings)
        {
            NaturalOrdering* naturalOrderingPtr = dynamic_cast<NaturalOrdering*>(orderingPtr);
            if (naturalOrderingPtr == nullptr)
            {
                SparseMatrix* matrixConstructed = nullptr;
                if (USE_EXISTING_ORDERED_SPARSE_STRUCTURES)
                {
                    matrixConstructed = SparseVizIO::readOrderedMatrixFromBinaryFile(orderingPtr->getOrderingName(), orderingPtr->getMatrix().getName());
                }
                if (matrixConstructed == nullptr)
                {
                    matrixConstructed = this->constructOrderedMatrix(orderingPtr);

                    if (EXPORT_ORDERED_SPARSE_STRUCTURES)
                    {
                        SparseVizIO::writeOrderedMatrixToBinaryFile(matrixConstructed);
                    }
                }
                for (auto& kernel: m_MatrixKernels)
                {
                    if (KERNEL_PERFORMANCE_LOG)
                    {
                        std::vector<CPUBenchmarkSettings> settings = getSettings();
                        sparseVizPerformance->activatePerf(settings.data(), settings.size());
                    }
                    KernelResult result = (*kernel)(*matrixConstructed);
                    SparseVizPerformance::OperationResults operationResults;
                    if (KERNEL_PERFORMANCE_LOG)
                    {
                        operationResults = sparseVizPerformance->deactivatePerf();
                    }
                    if (result.durations.empty())
                    {
                        continue;
                    }
                    orderingPtr->pushKernelResult(result);
                    logger->logRunningMatrixKernel(result, orderingPtr, operationResults);
                }
                delete matrixConstructed;
            }
            else
            {
                const SparseMatrix& matrix = orderingPtr->getMatrix();
                for (auto& kernel: m_MatrixKernels)
                {
                    if (KERNEL_PERFORMANCE_LOG)
                    {
                        std::vector<CPUBenchmarkSettings> settings = getSettings();
                        sparseVizPerformance->activatePerf(settings.data(), settings.size());
                    }
                    KernelResult result = (*kernel)(matrix);
                    SparseVizPerformance::OperationResults operationResults;
                    if (KERNEL_PERFORMANCE_LOG)
                    {
                        operationResults = sparseVizPerformance->deactivatePerf();
                    }
                    if (result.durations.empty())
                    {
                        continue;
                    }
                    orderingPtr->pushKernelResult(result);
                    logger->logRunningMatrixKernel(result, orderingPtr, operationResults);
                }
            }
        }
    }
}

#ifdef CUDA_ENABLED
void SparseVizEngine::runGPUMatrixKernels()
{
    if (!m_MatrixGPUKernels.empty())
    {
        for (auto& orderingPtr: m_MatrixOrderings)
        {
            NaturalOrdering* naturalOrderingPtr = dynamic_cast<NaturalOrdering*>(orderingPtr);
            if (naturalOrderingPtr == nullptr)
            {
                SparseMatrix* matrixConstructed = nullptr;
                if (USE_EXISTING_ORDERED_SPARSE_STRUCTURES)
                {
                    matrixConstructed = SparseVizIO::readOrderedMatrixFromBinaryFile(orderingPtr->getOrderingName(), orderingPtr->getMatrix().getName());
                }
                if (matrixConstructed == nullptr)
                {
                    matrixConstructed = this->constructOrderedMatrix(orderingPtr);

                    if (EXPORT_ORDERED_SPARSE_STRUCTURES)
                    {
                        SparseVizIO::writeOrderedMatrixToBinaryFile(matrixConstructed);
                    }
                }
                for (auto& kernel: m_MatrixGPUKernels)
                {
                    GPUKernelResult result = (*kernel)(*matrixConstructed);
                    if (result.durations.empty())
                    {
                        continue;
                    }
                    orderingPtr->pushGPUKernelResult(result);
                    logger->logRunningGPUMatrixKernel(result, orderingPtr);
                }
                delete matrixConstructed;
            }
            else
            {
                const SparseMatrix& matrix = orderingPtr->getMatrix();
                for (auto& kernel: m_MatrixGPUKernels)
                {
                    GPUKernelResult result = (*kernel)(matrix);
                    if (result.durations.empty())
                    {
                        continue;
                    }
                    orderingPtr->pushGPUKernelResult(result);
                    logger->logRunningGPUMatrixKernel(result, orderingPtr);
                    delete kernel;
                }
            }
        }
    }
}
#endif

#ifdef CUDA_ENABLED
void SparseVizEngine::runGPUTensorKernels()
{
    if (!m_TensorGPUKernels.empty())
    {
        for (auto& orderingPtr: m_TensorOrderings)
        {
            TensorNaturalOrdering* naturalOrderingPtr = dynamic_cast<TensorNaturalOrdering*>(orderingPtr);
            if (naturalOrderingPtr == nullptr)
            {
                SparseTensor* tensorConstructed = nullptr;
                if (USE_EXISTING_ORDERED_SPARSE_STRUCTURES)
                {
                    tensorConstructed = SparseVizIO::readOrderedTensorFromBinaryFile(orderingPtr->getOrderingName(), orderingPtr->getTensor().getName());
                }
                if (tensorConstructed == nullptr)
                {
                    tensorConstructed = this->constructOrderedTensor(orderingPtr);

                    if (EXPORT_ORDERED_SPARSE_STRUCTURES)
                    {
                        SparseVizIO::writeOrderedTensorToBinaryFile(tensorConstructed);
                    }
                }
                for (auto& kernel: m_TensorGPUKernels)
                {
                    GPUKernelResult result = (*kernel)(*tensorConstructed);
                    if (result.durations.empty())
                    {
                        continue;
                    }
                    orderingPtr->pushGPUKernelResult(result);
                    logger->logRunningGPUTensorKernel(result, orderingPtr);
                }
                delete tensorConstructed;
            }
            else
            {
                const SparseTensor& tensor = orderingPtr->getTensor();
                for (auto& kernel: m_TensorGPUKernels)
                {
                    GPUKernelResult result = (*kernel)(tensor);
                    if (result.durations.empty())
                    {
                        continue;
                    }
                    orderingPtr->pushGPUKernelResult(result);
                    logger->logRunningGPUTensorKernel(result, orderingPtr);
                    delete kernel;
                }
            }
        }
    }
}
#endif

SparseMatrix* SparseVizEngine::constructOrderedMatrix(MatrixOrdering* ordering)
{
    double startTime = omp_get_wtime();
    SparseMatrix *orderedMatrix = new SparseMatrix(ordering->getMatrix());
    orderedMatrix->order(ordering->getRowIPermutation(), ordering->getColIPermutation(), ordering->getOrderingName());
    double endTime = omp_get_wtime();
    logger->logConstructingOrderedMatrix(ordering, endTime - startTime);
    return orderedMatrix;
}

void SparseVizEngine::runTensorKernels()
{
    if (!m_TensorKernels.empty())
    {
        for (auto& orderingPtr: m_TensorOrderings)
        {
            TensorNaturalOrdering* naturalOrderingPtr = dynamic_cast<TensorNaturalOrdering*>(orderingPtr);

            if (naturalOrderingPtr == nullptr)
            {           
                SparseTensor* tensorConstructed = nullptr;
                if (USE_EXISTING_ORDERED_SPARSE_STRUCTURES)
                {
                    tensorConstructed = SparseVizIO::readOrderedTensorFromBinaryFile(orderingPtr->getOrderingName(), orderingPtr->getTensor().getName());
                }
                if (tensorConstructed == nullptr)
                {
                    tensorConstructed = this->constructOrderedTensor(orderingPtr);

                    if (EXPORT_ORDERED_SPARSE_STRUCTURES)
                    {
                        SparseVizIO::writeOrderedTensorToBinaryFile(tensorConstructed);
                    }
                }
                for (auto& kernel: m_TensorKernels)
                {
                    if (KERNEL_PERFORMANCE_LOG)
                    {
                        std::vector<CPUBenchmarkSettings> settings = getSettings();
                        sparseVizPerformance->activatePerf(settings.data(), settings.size());
                    }
                    KernelResult result = (*kernel)(*tensorConstructed);
                    SparseVizPerformance::OperationResults operationResults;
                    if (KERNEL_PERFORMANCE_LOG)
                    {
                        operationResults = sparseVizPerformance->deactivatePerf();
                    }
                    if (result.durations.empty())
                    {
                        continue;
                    }
                    orderingPtr->pushKernelResult(result);
                    logger->logRunningTensorKernel(result, orderingPtr, operationResults);
                }
                delete tensorConstructed;
            }
            else
            {
                const SparseTensor& tensor = orderingPtr->getTensor();
                for (auto& kernel: m_TensorKernels)
                {
                    if (KERNEL_PERFORMANCE_LOG)
                    {
                        std::vector<CPUBenchmarkSettings> settings = getSettings();
                        sparseVizPerformance->activatePerf(settings.data(), settings.size());
                    }
                    KernelResult result = (*kernel)(tensor);
                    SparseVizPerformance::OperationResults operationResults;
                    if (KERNEL_PERFORMANCE_LOG)
                    {
                        operationResults = sparseVizPerformance->deactivatePerf();
                    }
                    if (result.durations.empty())
                    {
                        continue;
                    }
                    orderingPtr->pushKernelResult(result);
                    logger->logRunningTensorKernel(result, orderingPtr, operationResults);
                }
            }
        }
    }
}

SparseTensor *SparseVizEngine::constructOrderedTensor(TensorOrdering* ordering)
{
    double startTime = omp_get_wtime();
    SparseTensor* orderedTensor = ordering->getTensor().generateOrderedTensor(ordering->getOrderedDimensions(), ordering->getOrderingName(), ordering->getActiveModes());
    double endTime = omp_get_wtime();
    logger->logConstructingOrderedTensor(ordering, endTime - startTime);
    return orderedTensor;
}

MatrixKernelFunction *SparseVizEngine::matrixKernelFactory(const std::string &kernelName, const std::vector<int>& threadCounts, const std::string &schedulingPolicy, int chunkSize, int nRun, int nIgnore)
{
    if (kernelName == "SPMVRowBased")
    {
        return new SPMVRowBased(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);
    }
    else if (kernelName == "BFS")
    {
        return new BFS(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);
    }
    else if (kernelName == "SequentialBFS")
    {
        return new SequentialBFS(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);
    } 
    else if (kernelName == "SPMMRowBased")
    {
        return new SPMMRowBased(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);
    }
    else if (kernelName == "Dijkstra")
    {
        return new Dijkstra(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);
    }
    else if (kernelName == "Bellman_Ford")
    {
        return new Bellman_Ford(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);
    }
    else if (kernelName == "Floyd_Warshall")
    {
        return new Floyd_Warshall(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);
    }
    return nullptr;
}

#ifdef CUDA_ENABLED
MatrixGPUKernel *SparseVizEngine::matrixGPUKernelFactory(const std::string& kernelClassName, const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, const std::vector<int>& sharedMemorySizes, const std::string& kernelParameters, int nRun, int nIgnore)
{
    if (kernelClassName == "JackardWeights")
    {
        return new JackardWeights(kernelName, gridSizes, blockSizes, sharedMemorySizes, kernelParameters, nRun, nIgnore);
    }
    return nullptr;
}
#endif

TensorKernelFunction *SparseVizEngine::tensorKernelFactory(const std::string &kernelName, const std::vector<int>& threadCounts, const std::string &schedulingPolicy, int chunkSize, int nRun, int nIgnore, const std::string& kernelParameters)
{
    if (kernelName == "COOTensorKernel" && TENSOR_STORAGE_TYPE == COO)
    {
        return new COOTensorKernel(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);
    }
    return nullptr;
}

#ifdef CUDA_ENABLED
TensorGPUKernel *SparseVizEngine::tensorGPUKernelFactory(const std::string& kernelClassName, const std::string &kernelName, const std::vector<int> &gridSizes, const std::vector<int> &blockSizes, const std::vector<int>& sharedMemorySizes, const std::string& kernelParameters, int nRun, int nIgnore)
{
    if (kernelClassName == "COOTensorGPUKernelSoA" && TENSOR_STORAGE_TYPE == COO)
    {
        return new COOTensorGPUKernelAoS(kernelName, gridSizes, blockSizes, sharedMemorySizes, nRun, nIgnore);
    }
    else if (kernelClassName == "COOTensorGPUKernelAoS" && TENSOR_STORAGE_TYPE == COO)
    {
        return new COOTensorGPUKernelSoA(kernelName, gridSizes, blockSizes, sharedMemorySizes, nRun, nIgnore);
    }
    return nullptr;
}
#endif

MatrixOrdering *SparseVizEngine::matrixOrderingFactory(SparseMatrix& matrix, std::string orderingClassName, std::string orderingName, std::string orderingParameters)
{
    if (orderingClassName == "Trivial")
    {  
        return new TrivialOrdering(matrix, orderingName, orderingParameters);
    }
    else if (orderingClassName == "Natural")
    {
        return new NaturalOrdering(matrix, orderingName);
    }
    else if (orderingClassName == "RCM")
    {
        return new RCMOrdering(matrix, orderingName, orderingParameters);
    }
    else if (orderingClassName == "Gray")
    {
        return new GrayOrdering(matrix, orderingName, orderingParameters);
    }
    else if (orderingClassName == "BOBA")
    {
        return new BOBAOrdering(matrix, orderingName, orderingParameters);
    }
    else if (orderingClassName == "SlashBurn")
    {
        return new SlashBurnOrdering(matrix, orderingName, orderingParameters);
    }
    else if (orderingClassName == "Patoh")
    {
        return new PatohOrdering(matrix, orderingName, orderingParameters);
    }
    else if (orderingClassName == "DynaDeg")
    {
        return new DynaDegOrdering(matrix, orderingName, orderingParameters);
    }
    else if (orderingClassName == "MinDeg")
    {
        return new MinDegOrdering(matrix, orderingName);
    }
    else if (orderingClassName == "AMD")
    {
        return new AMDOrdering(matrix, orderingName);
    }
#ifdef RABBIT_AVAILABLE
    else if (orderingClassName == "Rabbit")
    {
        return new RabbitOrdering(matrix, orderingName);
    }
#endif
    return nullptr;
}

TensorOrdering *SparseVizEngine::tensorOrderingFactory(SparseTensor &tensor, const std::vector<vType>& active_modes, std::string orderingClassName, std::string orderingName, std::string orderingParameters)
{
    if (orderingClassName == "Natural")
    {
        return new TensorNaturalOrdering(tensor, active_modes, orderingName);
    }
    else if (orderingClassName == "KPartite")
    {
        return new COOKPartiteOrdering(tensor, active_modes, orderingName, orderingParameters);
    }
    return nullptr;
}

void SparseVizEngine::constructMatrices()
{
    if (m_IsMConstructed)
    {
        return;
    }

#pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < m_MatrixFilePaths.size(); ++i)
    {
        SparseMatrix* matrix = SparseVizIO::readMatrixFromMarketFile(m_MatrixFilePaths[i]);
        m_Matrices.push_back(matrix);
    }

    m_IsMConstructed = true;
}

void SparseVizEngine::constructTensors()
{
    if (m_IsTConstructed)
    {
        return;
    }

#pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i != m_Tensors.size(); ++i)
    {
        SparseTensor* tensor = SparseVizIO::readTensorFromMarketFile(m_Tensors[i].filePath);
        m_Tensors[i].tensor = tensor;
    }

    m_IsTConstructed = true;
}

void SparseVizEngine::generateMatrixOrderings()
{
    if (m_IsMGenerated)
    {
        return;
    }

    std::vector<MatrixOrdering*> orderingsOriginalOrder(m_MatrixOrderings);

    std::sort(m_MatrixOrderings.begin(), m_MatrixOrderings.end(), [](MatrixOrdering* a, MatrixOrdering* b)
    {
        return a->getOrderingPrice() > b->getOrderingPrice();
    });

#pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < m_MatrixOrderings.size(); ++i)
    {
        m_MatrixOrderings[i]->generateOrdering();
    }

    m_MatrixOrderings = orderingsOriginalOrder;

    m_IsMGenerated = true;
}

void SparseVizEngine::generateTensorOrderings()
{
    if (m_IsTGenerated)
    {
        return;
    }
    std::vector<TensorOrdering*> orderingsOriginalOrder(m_TensorOrderings);

    std::sort(m_TensorOrderings.begin(), m_TensorOrderings.end(), [](TensorOrdering* a, TensorOrdering* b)
    {
        return a->getOrderingPrice() > b->getOrderingPrice();
    });

#pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < m_TensorOrderings.size(); ++i)
    {
        m_TensorOrderings[i]->generateOrdering();
    }

    m_TensorOrderings = orderingsOriginalOrder;

    m_IsTGenerated = true;
}

std::unordered_map<std::string, std::vector<MatrixOrdering*>> SparseVizEngine::getMatrixOrderingPermutations()
{
    std::unordered_map<std::string, std::vector<MatrixOrdering*>> permutations;

    for (const auto& matrix: m_Matrices)
    {
        for (const auto& ordering: m_MatrixOrderings)
        {
            if (matrix->getName() == ordering->getMatrix().getName())
            {
                permutations[matrix->getName()].push_back(ordering);
            }
        }
    }

    return permutations;
}

std::unordered_map<std::string, std::vector<MatrixOrdering*>> SparseVizEngine::getMatrixPermutations()
{
    std::unordered_map<std::string, std::vector<MatrixOrdering*>> permutations;

    for (const auto& ordering: m_MatrixOrderings)
    {
        permutations[ordering->getOrderingName()].push_back(ordering);
    }

    return permutations;
}

std::unordered_map<std::string, std::vector<TensorOrdering *>> SparseVizEngine::getTensorOrderingPermutations()
{
    std::unordered_map<std::string, std::vector<TensorOrdering*>> permutations;

    for (const auto& tensor: m_Tensors)
    {
        for (const auto& ordering: m_TensorOrderings)
        {
            if (tensor.tensor->getName() == ordering->getTensor().getName())
            {
                std::string key = tensor.tensor->getName();
                for(const vType& m : ordering->getActiveModes()) {
                    key += "_" + std::to_string(m);
                }
                permutations[key].push_back(ordering); 
            }
        }
    }

    return permutations;
}

std::unordered_map<std::string, std::vector<TensorOrdering *>> SparseVizEngine::getTensorPermutations()
{
    std::unordered_map<std::string, std::vector<TensorOrdering*>> permutations;

    for (const auto& ordering: m_TensorOrderings)
    {                
        permutations[ordering->getOrderingName()].push_back(ordering);
    }

    return permutations;
}
