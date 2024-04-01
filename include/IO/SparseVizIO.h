#ifndef SPARSEVIZ_SPARSEVIZIO_H
#define SPARSEVIZ_SPARSEVIZIO_H

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "config.h"


class SparseMatrix;
class SparseTensor;


/*!
 * @brief SparseVizIO is a static class that is responsible for every input-output operation conducted in the project.
 */
class SparseVizIO
{
public:

    /*!
     * @brief Constructor is unavailable.
     */
    SparseVizIO() = delete;

    /*!
     * @brief Copy constructor is unavailable.
     */
    SparseVizIO(const SparseVizIO& other) = delete;

    /*!
     * @brief Move constructor is unavailable.
     */
    SparseVizIO(const SparseVizIO&& other) = delete;

    /*!
     * @brief Copy operator is unavailable.
     */
    SparseVizIO& operator=(const SparseVizIO& other) = delete;

    /*!
     * @brief Move operator is unavailable.
     */
    SparseVizIO& operator=(const SparseVizIO&& other) = delete;

    /*!
     * @brief Destructor is unavailable.
     */
    ~SparseVizIO() = delete;

    /*!
     * @brief Reads matrix from the market file (ending with .mtx).
     * @param marketFileName Filename from which matrix will be read.
     * @return Pointer to newly constructed SparseMatrix object read from the market file.
     */
    static SparseMatrix* readMatrixFromMarketFile(const std::string& marketFileName);

    /*!
     * @brief Reads matrix from the binary file (ending with .mtx.bin).
     * @brief It is significantly faster than reading matrix from market file.
     * @param binaryFileName Binary filename from which matrix will be read.
     * @param name The name of the matrix to be constructed.
     * @return Pointer to newly constructed SparseMatrix object read from the binary file.
     * @return Returns nullptr if there is no binary file named binaryFileName.
     * @warning In order for this method to return properly constructed SparseMatrix object (not nullptr), SparseVizIO::readMatrixFromMarketFile should have been called one time in the past.
     * @see SparseVizIO::readMatrixFromMarketFile
     */
    static SparseMatrix* readMatrixFromBinaryFile(const std::string& binaryFileName, const std::string& name);

    /*!
     * @brief Writes matrix to a binary file (ending with .mtx.bin) to increase the speed at which matrices will be read in the future.
     * @param binaryFileName Binary file name into which matrix will be written.
     * @param matrix Pointer to a SparseMatrix object requested to be written into binary file.
     */
    static void writeMatrixToBinaryFile(const std::string &binaryFileName, SparseMatrix* matrix);

    /*!
     * @brief Reads matrix ordering from the binary file (ending with .bin).
     * @param binaryFileName Binary file name from which matrix ordering will be read.
     * @param rowIPermutation Row permutation array showing how rows are permuted after the matrix ordering.
     * @param colIPermutation Column permutation array showing how columns are permuted after the matrix ordering.
     * @param nRow Number of rows.
     * @param nCol Number of columns.
     * @return Returns true if read operation has been completed successfully, false otherwise.
     */
    static bool readMatrixOrderingFromBinaryFile(const std::string& binaryFileName, vType** rowIPermutation, vType** colIPermutation, vType nRow, vType nCol);

    /*!
     * @brief Writes matrix ordering into a binary file (ending with .bin).
     * @param binaryFileName Binary file name into which matrix ordering will be written.
     * @param rowIPermutation Row permutation array showing how rows are permuted after the matrix ordering.
     * @param colIPermutation Column permutation array showing how columns are permuted after the matrix ordering.
     * @param nRow Number of rows.
     * @param nCol Number of columns.
     */
    static void writeMatrixOrderingToBinaryFile(const std::string& binaryFileName, vType* rowIPermutation, vType* colIPermutation, vType nRow, vType nCol);

    /*!
     * @brief Reads ordered matrix from the binary file (ending with .mtx.bin).
     * @param orderingName The name of the ordering class that the matrix has been ordered with.
     * @param matrixName The name of the matrix on which ordering has been implemented.
     * @return Pointer to SparseMatrix object with ordering being implemented.
     * @return Returns nullptr if there is no orderingName implemented on matrixName found.
     * @see It internally uses SparseVizIO::readMatrixFromBinaryFile
     */
    static SparseMatrix* readOrderedMatrixFromBinaryFile(const std::string &orderingName, const std::string &matrixName);

    /*!
     * @brief Writes ordered matrix into a binary file (ending with .mtx.bin).
     * @param matrix Pointer to a SparseMatrix object that has been ordered and that is requested to be written into a binary file.
     * @brief You may have a question as how the ordering name is inferred from only matrix object parameter, its answer can be found in the SparseMatrix::generateOrderedMatrix method. During this method the matrix name is prefixed with the name of the ordering with which matrix is getting ordered.
     * @see SparseMatrix::generateOrderedMatrix
     */
    static void writeOrderedMatrixToBinaryFile(SparseMatrix* matrix);

    /*!
     * @brief Reads tensor from the market file (ending with .tns).
     * @param marketFileName Filename from which tensor will be read.
     * @return Pointer to newly constructed SparseTensor object read from the market file.
     */
    static SparseTensor* readTensorFromMarketFile(const std::string& marketFileName);

    /*!
     * @brief Reads tensor from the binary file (ending with .tns.bin).
     * @brief It is significantly faster than reading tensor from market file.
     * @param binaryFileName Binary filename from which matrix will be read.
     * @param name The name of the tensor to be constructed.
     * @return Pointer to newly constructed SparseTensor object read from the binary file.
     * @return Returns nullptr if there is no binary file named binaryFileName.
     * @warning In order for this method to return properly constructed SparseTensor object (not nullptr), SparseVizIO::readTensorFromMarketFile should have been called one time in the past.
     * @see SparseVizIO::readTensorFromMarketFile
     */
    static SparseTensor* readTensorFromBinaryFile(const std::string& binaryFileName, const std::string& name);

    /*!
     * @brief Writes tensor to a binary file (ending with .tns.bin) to increase the speed at which tensors will be read in the future.
     * @param binaryFileName Binary file name into which tensor will be written.
     * @param tensor Pointer to a SparseTensor object requested to be written into binary file.
     */
    static void writeTensorToBinaryFile(const std::string& binaryFileName, SparseTensor* tensor);

    /*!
     * @brief Reads tensor ordering from the binary file (ending with .bin).
     * @param binaryFileName Binary file name from which tensor ordering will be read.
     * @param norder The number of modes tensor has.
     * @param dims Array indicating the dimension count of each mode that this tensor has.
     * @param orderedDimensions Dimension permutation array showing how dimensions are permuted after the matrix ordering.
     * @return Returns true if read operation has been completed successfully, false otherwise.
     */
    static bool readTensorOrderingFromBinaryFile(const std::string& binaryFileName, vType norder, vType* dims, vType**& orderedDimensions);

    /*!
     * @brief Writes tensor ordering into a binary file (ending with .bin).
     * @param binaryFileName Binary file name into which tensor ordering will be written.
     * @param norder The number of modes tensor has.
     * @param dims Array indicating the dimension count of each mode that this tensor has.
     * @param orderedDimensions Dimension permutation array showing how dimensions are permuted after the matrix ordering.
     */
    static void writeTensorOrderingToBinaryFile(const std::string& binaryFileName, vType norder, vType* dims, vType** orderedDimensions);

    /*!
     * @brief Reads ordered tensor from the binary file (ending with .tns.bin).
     * @param orderingName The name of the ordering class that the tensor has been ordered with.
     * @param tensorName The name of the tensor on which ordering has been implemented.
     * @return Pointer to SparseTensor object with ordering being implemented.
     * @return Returns nullptr if there is no orderingName implemented on tensorName found.
     * @see It internally uses SparseVizIO::readTensorFromBinaryFile
     */
    static SparseTensor* readOrderedTensorFromBinaryFile(const std::string &orderingName, const std::string &tensorName);

    /*!
     * @brief Writes ordered tensor into a binary file (ending with .tns.bin).
     * @param tensor Pointer to a SparseTensor object that has been ordered and that is requested to be written into a binary file.
     * @brief You may have a question as how the ordering name is inferred from only tensor object parameter, its answer can be found in the SparseTensor::generateOrderedTensor method. During this method the tensor name is prefixed with the name of the ordering with which tensor is getting ordered.
     * @see SparseTensor::generateOrderedTensor
     */
    static void writeOrderedTensorToBinaryFile(SparseTensor* tensor);
};


#endif //SPARSEVIZ_SPARSEVIZIO_H
