#ifndef SPARSEVIZ_SPARSEMATRIX_H
#define SPARSEVIZ_SPARSEMATRIX_H

#include <vector>
#include "config.h"
#include <iostream>
#include "helpers.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <string>


/*!
 * @brief SparseMatrix data structure that is capable of representing sparse matrix in a compressed form (it supports both compressed formats (CSR and CSC)).
 * @brief It can conduct various operations internally such as transformation between formats, symmetry checking, ordering implementation, making itself square or symmetric and more.
 */
class SparseMatrix
{
public:
    /*!
     * @brief Constructs SparseMatrix object from array of nonzeros.
     * @param name The name of the SparseMatrix.
     * @param r Row count.
     * @param c Col count.
     * @param nnz Number of nonzeros
     * @param storage Array of coordinates with a layout as: r1,c1,r2,c2.
     * @param values Array of values corresponding to each nonzero in the storage array.
     * @warning In order for the construction to be done properly, the nonzeros should be sorted in the order of increasing rows.
     */
    SparseMatrix(std::string name, vType r, vType c, eType nnz, vType* storage, valType* values);

    /*!
     * @brief Constructs SparseMatrix object from 3 arrays required to represent compressed matrix structure.
     * @param name The name of the SparseMatrix.
     * @param r Row count.
     * @param c Col count.
     * @param nnzCount Nonzero count.
     * @param ptrs Pointers pointing to the start of the each row (in CSR) or each col (in CSC).
     * @param ids Column indexes of each nonzero (in CSR) or row indexes of each nonzero (in CSC).
     * @param vals Values of each nonzero.
     */
    SparseMatrix(std::string name, vType r, vType c, vType nnzCount, vType* ptrs, vType* ids, valType* vals);

    /*!
     * @brief Constructs SparseMatrix object with every element in each of its arrays required to represent compressed matrix structure to be defaulted as 0.
     * @param name The name of the SparseMatrix.
     * @param r Row count.
     * @param c Col count.
     * @param nnzCount Nonzero count.
     * @param isSymmetric Flag indicating whether matrix has numerical symmetry | Default = false.
     * @param isPatternSymmetric Flag indicating whether matrix has pattern symmetry | Default = false.
     */
    SparseMatrix(std::string name, vType r, vType c, vType nnzCount, bool isSymmetric = false, bool isPatternSymmetric = false);

    /*!
     * @brief Constructs SparseMatrix object from 3 arrays required to represent compressed matrix structure.
     * @param name The name of the SparseMatrix.
     * @param r Row count.
     * @param c Col count.
     * @param nnzCount Nonzero count.
     * @param ptrs Pointers pointing to the start of the each row (in CSR) or each col (in CSC).
     * @param ids Column indexes of each nonzero (in CSR) or row indexes of each nonzero (in CSC).
     * @param vals Values of each nonzero.
     * @param isSymmetric Flag indicating whether matrix has numerical symmetry | Default = false.
     * @param isPatternSymmetric Flag indicating whether matrix has pattern symmetry | Default = false.

     */
    SparseMatrix(std::string name, vType r, vType c, vType nnzCount, vType* ptrs, vType* ids, valType* vals, bool isSymmetric, bool isPatternSymmetric);

    /*!
     * @brief Copy Constructor.
     */
    SparseMatrix(const SparseMatrix& other);

    /*!
     * @brief Copy Operator.
     */
    SparseMatrix& operator=(const SparseMatrix& other);

    /*!
     * @brief Move Constructor.
     */
    SparseMatrix(SparseMatrix&& other);

    /*!
     * @brief Move Operator.
     */
    SparseMatrix& operator=(SparseMatrix&& other);

    /*!
     * @brief Cleans up the memory space allocated by the 3 arrays.
     */
    ~SparseMatrix();

    /*!
     * @brief Prints the first count nonzeros sorted in the row major order.
     * @count Number of nonzeros to be printed, if bigger than row count then it prints the whole matrix in row major order.
     */
    void printNonzeros(vType count) const;

    /*!
     * @brief Transposes the matrix itself. CSR to CSC OR CSC to CSR.
     */
    void transpose();

    /*!
     * @brief Generates a new SparseMatrix object that is transposed form of the current matrix.
     * @return If this == CSR then return == CSC.
     * @return If this == CSC then return == CSR.
     */
    SparseMatrix generateTransposedMatrix() const;

    /*!
     * @brief Orders the current matrix itself.
     * @param orderedRow Row permutations to replace existing rows.
     * @param orderedCol Col permutations to replace existing cols.
     * @param orderingName The name of the ordering to be implemented.
     */
    void order(const vType* orderedRow, const vType* orderedCol, const std::string& orderingName);

    /*!
     * @brief Orders a copy of the current (this) matrix.
     * @param orderedRow Row permutations to replace existing rows.
     * @param orderedCol Col permutations to replace existing cols.
     * @param orderingName The name of the ordering to be implemented.
     * @return A copy of the current matrix on which the ordering is implemented.
     */
    SparseMatrix generateOrderedMatrix(const vType* orderedRow, const vType* orderedCol, const std::string& orderingName) const;

    /*!
     * @brief Sums 2 sparse matrices with same #rows and #cols.
     * @param Other sparse matrix that the sum operation is going to be conducted with.
     * @return *this + other.
     */
    SparseMatrix operator+(const SparseMatrix& other) const;

    /*!
     * @brief Symmetritizes the current sparse matrix itself.
     * @details Symmetritize operation is based on the following formula: A + A'.
     */
    void symmetritize();

    /*!
     * @brief Symmetritizes a copy of the current (this) matrix.
     * @return A copy of the current matrix that is symmetritized.
     * @details Symmetritize operation is based on the following formula: A + A'.
     */
    SparseMatrix generateSymmetritizedMatrix() const;

    /*!
     * @brief Converts current matrix to a one that is square shaped.
     */
    void makeSquareShapedMatrix();

    /*!
     * @brief Creates a copy of the current (this) matrix and converts it into a one that is square shaped.
     */
    SparseMatrix generateSquareShapedMatrix() const;

    /*!
     * @brief Generates a private member variable m_OrderingSupportedMatrix that is pattern symmetric and square shaped.
     */
    void generateOrderingSupportedMatrix();

    /*!
     * @brief Checks certain properties of the matrix such as whether or not it has numerical symmetry or pattern symmetry or is square shaped -> changes private member variables {m_IsSymmetric, m_IsPatternSymmetric, m_IsSquare}.
     */
    void checkProperties(bool adjSorted);

    /*!
     * @brief Checks whether or not the matrix has a numerical symmetry or pattern symmetry -> changes private member variables {m_IsSymmetric, m_IsPatternSymmetric}.
     */
    void checkSymmetry(bool adj_sorted);

    /*!
     * @brief Checks whether or not the matrix is square shaped -> changes private member variable m_IsSquare.
     */
    void checkShape();

    // Read-Only Getters
    [[maybe_unused]] [[nodiscard]] inline std::string getName() const {return m_Name;}
    [[maybe_unused]] [[nodiscard]] inline vType getRowCount() const {return m_Row;}
    [[maybe_unused]] [[nodiscard]] inline vType getColCount() const {return m_Col;}
    [[maybe_unused]] [[nodiscard]] inline vType getNNZCount() const {return m_NNZCount;}
    [[maybe_unused]] [[nodiscard]] inline bool isSymmetric() const {return m_IsSymmetric;}
    [[maybe_unused]] [[nodiscard]] inline bool isPatternSymmetric() const {return m_IsPatternSymmetric;}
    [[maybe_unused]] [[nodiscard]] inline bool isSquare() const {return m_IsSquare;}
    [[maybe_unused]] [[nodiscard]] inline vType* getPtr() const {return m_Ptr;}
    [[maybe_unused]] [[nodiscard]] inline vType* getInd() const {return m_Ind;}
    [[maybe_unused]] [[nodiscard]] inline valType* getValues() const {return m_Values;}

    // Read-Write Getters
    [[maybe_unused]] [[nodiscard]] inline std::string& getName() {return m_Name;}
    [[maybe_unused]] [[nodiscard]] inline vType& getRowCount() {return m_Row;}
    [[maybe_unused]] [[nodiscard]] inline vType& getColCount() {return m_Col;}
    [[maybe_unused]] [[nodiscard]] inline vType& getNNZCount() {return m_NNZCount;}
    [[maybe_unused]] [[nodiscard]] inline bool& isSymmetric() {return m_IsSymmetric;}
    [[maybe_unused]] [[nodiscard]] inline bool& isPatternSymmetric() {return m_IsPatternSymmetric;}
    [[maybe_unused]] [[nodiscard]] inline bool& isSquare() {return m_IsSquare;}
    [[maybe_unused]] [[nodiscard]] inline vType*& getPtr() {return m_Ptr;}
    [[maybe_unused]] [[nodiscard]] inline vType*& getInd() {return m_Ind;}
    [[maybe_unused]] [[nodiscard]] inline valType*& getValues() {return m_Values;}
    [[maybe_unused]] [[nodiscard]] inline SparseMatrix*& getOrderingSupportedMatrix() {return m_OrderingSupportedMatrix;}

private:
    /*!
     * @brief Conduct deep copy operation with the other matrix.
     * @param other sparse matrix that the deep copy operation will be conducted with.
     */
    void deepCopy(const SparseMatrix& other);

private:
    SparseMatrix* m_OrderingSupportedMatrix;

    std::string m_Name;
    vType m_Row;
    vType m_Col;
    vType m_NNZCount;
    bool m_IsSymmetric;
    bool m_IsPatternSymmetric;
    bool m_IsSquare;

    vType* m_Ptr;
    vType* m_Ind;
    valType* m_Values;
};


#endif //SPARSEVIZ_SPARSEMATRIX_H
