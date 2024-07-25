#ifndef SPARSEVIZ_MATRIXORDERING_H
#define SPARSEVIZ_MATRIXORDERING_H

#include <stdexcept>
#include <string>
#include "config.h"
#include <iostream>
#include "SparseMatrix.h"
#include <stdexcept>
#include <exception>
#include "KernelResult.h"
#include "GPUKernelResult.h"


/*!
 * @brief MatrixOrdering is an abstract class from which every custom ordering in the SparseViz ecosystem is derived.
 * @brief It handles variety of operations under the hood to help users focus on their ordering algorithm itself and not the way in which it will be integrated onto SparseViz ecosystem.
 */
class MatrixOrdering
{
public:
    /*!
     * @brief Constructor of the abstract MatrixOrdering class.
     * @param orderingName The name of the ordering.
     * @param rectangularSupport Flag indicating whether or not the custom ordering has a support for rectangular matrices.
     * @param patternUnsymmetricSupport Flag indicating whether or not the custom ordering has a support for pattern unsymmetric matrices.
     * @param orderingFunctionPrice An integer from 0 to 10 representing the overhead of the ordering function that is dictated to be overridden by every custom ordering class. Defaults to 0, but specifying it is recommended to optimize program performance.
     */
    MatrixOrdering(SparseMatrix& matrix, std::string orderingName, bool rectangularSupport, bool patternUnsymmetricSupport, unsigned int orderingFunctionPrice = 0);

    /*!
     * @brief Copy Constructor is unavailable.
     */
    MatrixOrdering(const MatrixOrdering& other) = delete;

    /*!
     * @brief Copy Operator is unavailable.
     */
    MatrixOrdering(const MatrixOrdering&& other) = delete;

    /*!
     * @brief Move Constructor is unavailable.
     */
    MatrixOrdering& operator=(const MatrixOrdering& other) = delete;

    /*!
     * @brief Move Operator is unavailable.
     */
    MatrixOrdering& operator=(const MatrixOrdering&& other) = delete;

    /*!
     * @brief Destructor to clean up the allocated memory space.
     */
    virtual ~MatrixOrdering();

    /*!
     * @brief Public exposed method to run the orderingFunction, it acts as a wrapper and handles variety of operations behind the scenes beside running the virtual method this->orderingFunction.
     * @param writeOrdering Flag indicating whether or not ordering generated wants to be written into binary file.
     * @see Some operations implemented behind the scenes are as following:
     * @see MatrixOrdering::readOrdering
     * @see MatrixOrdering::checkOrderingSupportingStatus
     * @see MatrixOrdering::transformGeneratedPermutation
     * @see MatrixOrdering::checkPermutationCorrectness
     * @see MatrixOrdering::writeOrdering
     */
    void generateOrdering(bool writeOrdering = true);

    /*!
     * @brief Stores the result of a kernel that has been executed on the matrix ordered with the current (this) MatrixOrdering object.
     */
    void pushKernelResult(const KernelResult& kernelResult);

    /*!
     * @brief Stores the result of a gpu kernel that has been executed on the matrix ordered with the current (this) MatrixOrdering object.
     */
    void pushGPUKernelResult(const GPUKernelResult& gpuKernelResult);

    // GETTERS
    [[maybe_unused]] [[nodiscard]] std::string getOrderingName() const {return orderingName;}
    [[maybe_unused]] [[nodiscard]] double getOrderingPrice() const {return orderingPrice;}
    [[maybe_unused]] [[nodiscard]] bool hasRectangularSupport() const {return rectangularSupport;}
    [[maybe_unused]] [[nodiscard]] bool hasPatternUnsymmetricSupport() const {return patternUnsymmetricSupport;}
    [[maybe_unused]] [[nodiscard]] const vType* getRowIPermutation() const;
    [[maybe_unused]] [[nodiscard]] const vType* getColIPermutation() const;
    [[maybe_unused]] [[nodiscard]] const SparseMatrix& getMatrix();
    [[maybe_unused]] [[nodiscard]] const std::vector<KernelResult>& getKernelResults() const {return kernelResults;}
    [[maybe_unused]] [[nodiscard]] const std::vector<GPUKernelResult>& getGPUKernelResults() const {return gpuKernelResults;}

protected:
    /*!
     * @brief The virtual orderingFunction method is the only method desired to be overridden by the derived ordering classes for custom ordering algorithm implementations.
     */
    virtual void orderingFunction() = 0;

protected:
    vType* rowIPermutation;    //!< RowIPermutation is an array that every orderingFunction generates to indicate the new locations of the current row permutations. Initially, it is set to nullptr and should be properly initialized by the orderingFunction. Cleaning it up is not within the responsibilities of the custom ordering classes.
    vType* colIPermutation;    //!< ColIPermutation is an array that every orderingFunction generates to indicate the new locations of the current column permutations. Initially, it is set to nullptr and should be properly initialized by the orderingFunction. Cleaning it up is not within the responsibilities of the custom ordering classes.

private:
    /*!
     * @brief Checks whether or not the current sparse matrix is suitable for ordering, if not, the supported matrix shall be created.
     */
    void checkOrderingSupportingStatus();

    /*!
     * @brief If MatrixOrdering::checkOrderingSupportingStatus sees the need of creating an ordering supported matrix, then it is required to transform the generated permutation to be applicable for the current matrix, which is exactly what this method does.
     * @see MatrixOrdering::checkOrderingSupportingStatus
     */
    void transformGeneratedPermutation();

    /*!
     * @brief Calculates the ordering price based on certain parameters, such as the overhead that the orderingFunction has; the nnz count, the row count, the col count of the current matrix.
     */
    void calculateOrderingPrice(unsigned int orderingFunctionPrice);

    /*!
     * @brief Checks whether or not the permutation arrays generated by the orderingFunction has some kind of logical errors in it.
     * @throws std::logic_error, if it detected that there is a logical error made during the construction of the permutation arrays.
     */
    void checkPermutationCorrectness();

    /*!
     * @brief Reads ordering from the binary file if it is detected that the current matrix ordering has been called on the same matrix and has been written to a binary file previously.
     */
    bool readOrdering();

    /*!
     * @brief Writes matrix ordering to a binary file to prevent calling the orderingFunction in the future.
     */
    void writeOrdering();

private:
    SparseMatrix& matrix;
    bool isOrderingSupportedMatrixGenerated;
    std::vector<KernelResult> kernelResults;
    std::vector<GPUKernelResult> gpuKernelResults;
    std::string orderingName;
    double orderingPrice;
    bool rectangularSupport;
    bool patternUnsymmetricSupport;
};


#endif //SPARSEVIZ_MATRIXORDERING_H
