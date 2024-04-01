#ifndef SPARSEVIZ_SPARSEVIZENGINE_H
#define SPARSEVIZ_SPARSEVIZENGINE_H

// STD
#include <string>
#include <vector>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <exception>
// Sparse
#include "SparseMatrix.h"
#include "SparseTensor.h"
// Orderings
#include "MatrixOrdering.h"
#include "TensorOrdering.h"
#include "TrivialOrdering.h"
#include "NaturalOrdering.h"
#include "RCMOrdering.h"
#include "AMDOrdering.h"
#include "GrayOrdering.h"
#include "BOBAOrdering.h"
#include "SlashBurnOrdering.h"
#include "PatohOrdering.h"
#include "DynaDegOrdering.h"
#include "MinDegOrdering.h"
#include "TensorNaturalOrdering.h"
#include "COOKPartiteOrdering.h"
#ifdef RABBIT_AVAILABLE
#include "RabbitOrdering.h"
#endif
// Matrix Kernels
#include "MatrixKernelFunction.h"
#include "SpmvRowBased.h"
#include "Bfs.h"
#include "SequentialBfs.h"
#include "SpmmRowBased.h"
#include "Dijkstra.h"
#include "Bellman_Ford.h"
#include "Floyd_Warshall.h"
// Tensor Kernels
#include "TensorKernelFunction.h"
#include "COOTensorKernel.h"
#include "MTTKRP.h"
// Cuda Needed
#ifdef CUDA_ENABLED
#include "MatrixGPUKernel.h"
#include "TensorGPUKernel.h"
#include "GPUKernelResult.h"
#include "CudaHelloWorld.h"
#include "GPUSpmvRowBased.h"
#include "COOTensorGPUKernelAoS.h"
#include "COOTensorGPUKernelSoA.h"
#endif


/*!
 * @brief SparseVizEngine is the top module of the project, which is responsible for dictating other modules to conduct their responsibilities.
 */
class SparseVizEngine
{
public:
    /*!
     * @brief SparseVizEngine Constructor.
     */
    SparseVizEngine(): m_IsMConstructed(false), m_IsMGenerated(false), m_IsTConstructed(false), m_IsTGenerated(false) {}

    /*!
     * @brief Copy constructor is unavailable.
     */
    SparseVizEngine(const SparseVizEngine& other) = delete;

    /*!
     * @brief Move constructor is unavailable.
     */
    SparseVizEngine(const SparseVizEngine&& other) = delete;

    /*!
     * @brief Copy operator is unavailable.
     */
    SparseVizEngine& operator=(const SparseVizEngine& other) = delete;

    /*!
     * @brief Move operator is unavailable.
     */
    SparseVizEngine& operator=(const SparseVizEngine&& other) = delete;

    /*!
     * @brief Destructor to delete every object allocated during the lifetime of the program, including sparse data structures, orderings, CPU kernels, GPU kernels.
     */
    ~SparseVizEngine();

    /*!
     * @brief Runs the engine and start.
     */
    void runEngine();

    /*!
     * @brief Adds matrix to be processed, typically called by the ConfigFileReader.
     * @param matrixFilePath Matrix file path to be added.
     */
    void addMatrix(const std::string& matrixFilePath);

    /*!
     * @brief Constructs matrices that have been added by addMatrix method previously, gets called only one time throughout the program and constructs matrices added upto that point.
     * @see SparseVizEngine::addMatrix
     */
    void constructMatrices();

    /*!
     * @brief Adds matrix ordering that is desired to be implemented on matrices, typically called by the ConfigFileReader.
     * @param orderingClassName The name of the class deriving from MatrixOrdering.
     * @param orderingName The name of the ordering.
     * @param orderingParameters Ordering parameters given to the ordering class constructor, typically not splitted and expected to be done so in the ordering constructor.
     */
    void addMatrixOrdering(const std::string& orderingClassName, const std::string& orderingName, const std::string& orderingParameters);

    /*!
     * @brief Generates matrix orderings that have been added by addMatrixOrdering method previously, gets called only one time throughout the program and generates matrix orderings added upto that point.
     * @see SparseVizEngine::addMatrixOrderings
     */
    void generateMatrixOrderings();

    /*!
     * @brief Adds tensors to be processed, typically called by the ConfigFileReader.
     * @param tensorFilePath Tensor file path to be added.
     */
    void addTensor(const std::string& tensorFilePath, const std::vector<vType>& new_active_modes);

    /*!
     * @brief Constructs tensors that have been added by addTensor method previously, gets called only one time throughout the program and constructs tensors added upto that point.
     * @see SparseVizEngine::addTensor
     */
    void constructTensors();

    /*!
     * @brief Adds tensor ordering that is desired to be implemented on tensors, typically called by the ConfigFileReader.
     * @param orderingClassName The name of the class deriving from TensorOrdering.
     * @param orderingName The name of the ordering.
     * @param orderingParameters Ordering parameters given to the ordering class constructor, typically not splitted and expected to be done so in the ordering constructor.
     * @param zooType For zooType = FULLTENSOR active modes are ignored
     */
    void addTensorOrdering(const std::string& orderingClassName, const std::string& orderingName, const std::string& orderingParameters, const std::string& zooType);

    /*!
     * @brief Generates tensor orderings that have been added by addTensorOrdering method previously, gets called only one time throughout the program and generates tensor orderings added upto that point.
     * @see SparseVizEngine::addTensorOrdering
     */
    void generateTensorOrderings();

    /*!
     * @brief Adds matrix kernel that is desired to be executed on ordering matrices, typically called by the ConfigFileReader.
     * @param kernelName The name of the kernel class.
     * @param threadCounts Array of integers indicating the thread counts with which the kernel should be executed separately.
     * @param schedulingPolicy OMP scheduling policy -> policies available are: static, auto, dynamic, guided.
     * @param chunkSize OMP chunk size.
     * @param nRun Number of times kernel is desired to be executed.
     * @param nIgnore Number of times the initial executions are desired to be ignored.
     */
    void addMatrixKernel(const std::string &kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore);

    /*!
     * @brief Adds tensor kernel that is desired to be executed on ordering tensors, typically called by the ConfigFileReader.
     * @param kernelName The name of the kernel class.
     * @param threadCounts Array of integers indicating the thread counts with which the kernel should be executed separately.
     * @param schedulingPolicy OMP scheduling policy -> policies available are: static, auto, dynamic, guided.
     * @param chunkSize OMP chunk size.
     * @param nRun Number of times kernel is desired to be executed.
     * @param nIgnore Number of times the initial executions are desired to be ignored.
     * @param orderingParameters Ordering parameters given to the ordering class constructor, typically not splitted and expected to be done so in the ordering constructor.
     */
    void addTensorKernel(const std::string &kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore, const std::string& orderingParameters);

#ifdef CUDA_ENABLED
    /*!
     * @brief Adds matrix gpu kernel that is desired to be executed on ordering matrices, typically called by the ConfigFileReader.
     * @param kernelName The name of the kernel class.
     * @param gridSizes Array of integers indicating the grid sizes with which gpu kernels should be launched.
     * @param blockSizes Array of integers indicating the block sizes with which gpu kernels should be launched.
     * @param nRun Number of times kernel is desired to be executed.
     * @param nIgnore Number of times the initial executions are desired to be ignored.
     * @warning It is compiled only if CUDA and CUDA compatible device are present in the user machine.
     * @warning The length of the gridSizes should be equal to the length of the blockSizes as they are zipped while processing.
     */
    void addGPUMatrixKernel(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore);

    /*!
     * @brief Adds tensor gpu kernel that is desired to be executed on ordering tensors, typically called by the ConfigFileReader.
     * @param kernelName The name of the kernel class.
     * @param gridSizes Array of integers indicating the grid sizes with which gpu kernels should be launched.
     * @param blockSizes Array of integers indicating the block sizes with which gpu kernels should be launched.
     * @param nRun Number of times kernel is desired to be executed.
     * @param nIgnore Number of times the initial executions are desired to be ignored.
     * @warning It is compiled only if CUDA and CUDA compatible device are present in the user machine.
     * @warning The length of the gridSizes should be equal to the length of the blockSizes as they are zipped while processing.
     */
    void addGPUTensorKernel(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore);
#endif

    /*!
     * @brief Returns matrix x matrix ordering cartesian product.
     * @return Hash Table with keys being matrix names and values being array of matrix orderings belonging to each key.
     */
    std::unordered_map<std::string, std::vector<MatrixOrdering*>> getMatrixOrderingPermutations();

    /*!
     * @brief Returns matrix x matrix ordering cartesian product.
     * @return Hash Table with keys being ordering names and values being array of matrix orderings belonging to each key.
     */
    std::unordered_map<std::string, std::vector<MatrixOrdering*>> getMatrixPermutations();

    /*!
     * @brief Returns tensor x tensor ordering cartesian product.
     * @return Hash Table with keys being tensor names and values being array of tensor orderings belonging to each key.
     */
    std::unordered_map<std::string, std::vector<TensorOrdering*>> getTensorOrderingPermutations();

    /*!
     * @brief Returns tensor x tensor ordering cartesian product.
     * @return Hash Table with keys being ordering names and values being array of tensor orderings belonging to each key.
     */
    std::unordered_map<std::string, std::vector<TensorOrdering*>> getTensorPermutations();

private:
    /*!
     * @brief Constructs a new SparseMatrix object on which matrix ordering is implemented.
     * @param ordering Pointer to any custom ordering object deriving from MatrixOrdering that is desired to be implemented on a new SparseMatrix.
     * @return Pointer to a new SparseMatrix object on which matrix ordering has been implemented.
     */
    SparseMatrix* constructOrderedMatrix(MatrixOrdering* ordering);

    /*!
     * @brief Constructs a new SparseTensor object on which tensor ordering is implemented.
     * @param ordering Pointer to any custom ordering object deriving from TensorOrdering that is desired to be implemented on a new SparseTensor.
     * @return Pointer to a new SparseTensor object on which tensor ordering has been implemented.
     */
    SparseTensor* constructOrderedTensor(TensorOrdering* ordering);

    /*!
     * @brief Runs all matrix kernels.
     */
    void runMatrixKernels();

    /*!
     * @brief Runs all tensor kernels.
     */
    void runTensorKernels();

    /*!
     * @brief Matrix ordering factory in which custom orderings are produced.
     * @param matrix SparseMatrix for which ordering is generated.
     * @param orderingClassName The custom ordering class an ordering object is going to be instantiated from.
     * @param orderingName The name of the custom ordering object.
     * @param orderingParameters The ordering parameters to be given to the ordering constructor, typically not splitted and expected to be done so in the ordering class constructor.
     * @return MatrixOrdering abstract class from which every custom ordering class in the SparseViz ecosystem is derived.
     */
    MatrixOrdering* matrixOrderingFactory(SparseMatrix& matrix, std::string orderingClassName, std::string orderingName, std::string orderingParameters);

    /*!
     * @brief Tensor ordering factory in which custom orderings are produced.
     * @param tensor SparseTensor for which ordering is generated.
     * @param active_modes Active modes of the SparseTensor.
     * @param orderingClassName The custom ordering class an ordering object is going to be instantiated from.
     * @param orderingName The name of the custom ordering object.
     * @param orderingParameters The ordering parameters to be given to the ordering constructor, typically not splitted and expected to be done so in the ordering class constructor.
     * @return TensorOrdering abstract class from which every custom ordering class in the SparseViz ecosystem is derived.
     */
    TensorOrdering* tensorOrderingFactory(SparseTensor& tensor, const std::vector<vType>& active_modes, std::string orderingClassName, std::string orderingName, std::string orderingParameters);

    /*!
     * @brief Matrix kernel factory in which custom kernels are produced.
     * @param kernelName The custom kernel class a kernel object is going to be instantiated from.
     * @param threadCounts Array of integers indicating the thread counts with which the kernel should be executed separately.
     * @param schedulingPolicy OMP scheduling policy -> policies available are: static, auto, dynamic, guided.
     * @param chunkSize OMP chunk size.
     * @param nRun Number of times kernel is desired to be executed.
     * @param nIgnore Number of times the initial executions are desired to be ignored.
     * @return MatrixKernelFunction abstract class from which every custom matrix kernel class in the SparseViz ecosystem is derived.
     */
    MatrixKernelFunction* matrixKernelFactory(const std::string &kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore);

    /*!
     * @brief Tensor kernel factory in which custom kernels are produced.
     * @param kernelName The custom kernel class a kernel object is going to be instantiated from.
     * @param threadCounts Array of integers indicating the thread counts with which the kernel should be executed separately.
     * @param schedulingPolicy OMP scheduling policy -> policies available are: static, auto, dynamic, guided.
     * @param chunkSize OMP chunk size.
     * @param nRun Number of times kernel is desired to be executed.
     * @param nIgnore Number of times the initial executions are desired to be ignored.
     * @return TensorKernelFunction abstract class from which every custom tensor kernel class in the SparseViz ecosystem is derived.
     */
    TensorKernelFunction* tensorKernelFactory(const std::string &kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore, const std::string& orderingParameters);

#ifdef CUDA_ENABLED
    /*!
     * @brief Runs all GPU matrix kernels.
     * @warning It is compiled only if CUDA and CUDA compatible device are present in the user machine.
     */
    void runGPUMatrixKernels();

    /*!
     * @brief Matrix gpu kernel factory in which custom gpu kernels are produced.
     * @param kernelName The custom kernel class a kernel object is going to be instantiated from.
     * @param gridSizes Array of integers indicating the grid sizes with which gpu kernels should be launched.
     * @param blockSizes Array of integers indicating the block sizes with which gpu kernels should be launched.
     * @param nRun Number of times kernel is desired to be executed.
     * @param nIgnore Number of times the initial executions are desired to be ignored.
     * @return MatrixGPUKernel abstract class from which every custom matrix kernel class in the SparseViz ecosystem is derived.
     */
    MatrixGPUKernel* matrixGPUKernelFactory(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore);

    /*!
     * @brief Runs all GPU tensor kernels.
     * @warning It is compiled only if CUDA and CUDA compatible device are present in the user machine.
     */
    void runGPUTensorKernels();

    /*!
     * @brief Tensor gpu kernel factory in which custom gpu kernels are produced.
     * @param kernelName The custom kernel class a kernel object is going to be instantiated from.
     * @param gridSizes Array of integers indicating the grid sizes with which gpu kernels should be launched.
     * @param blockSizes Array of integers indicating the block sizes with which gpu kernels should be launched.
     * @param nRun Number of times kernel is desired to be executed.
     * @param nIgnore Number of times the initial executions are desired to be ignored.
     * @return TensorGPUKernel abstract class from which every custom matrix kernel class in the SparseViz ecosystem is derived.
     */
    TensorGPUKernel* tensorGPUKernelFactory(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore);
#endif

private:
    bool m_IsMConstructed;
    bool m_IsMGenerated;

    bool m_IsTConstructed;
    bool m_IsTGenerated;

    std::vector<std::string> m_MatrixFilePaths;
    std::vector<SparseMatrix*> m_Matrices;
    std::vector<MatrixOrdering*> m_MatrixOrderings;

    struct TensorBin
    {
        TensorBin(const std::string& path, const std::vector<vType>& dimension)
        :   filePath(path)
        {
            active_modes.push_back(dimension);
        }
        std::string filePath;
        SparseTensor* tensor;
        std::vector<std::vector<vType>> active_modes;
    };
    std::vector<TensorBin> m_Tensors;
    std::vector<TensorOrdering*> m_TensorOrderings;

    std::vector<MatrixKernelFunction*> m_MatrixKernels;
    std::vector<TensorKernelFunction*> m_TensorKernels;

#ifdef CUDA_ENABLED
    std::vector<MatrixGPUKernel*> m_MatrixGPUKernels;
    std::vector<TensorGPUKernel*> m_TensorGPUKernels;
#endif
};


#endif //SPARSEVIZ_SPARSEVIZENGINE_H
