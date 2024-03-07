//
// Created on 1/14/24.
//

#ifndef SPARSEVIZ_SPARSETENSOR_H
#define SPARSEVIZ_SPARSETENSOR_H

#include "config.h"
#include <string>
#include <vector>
#include "pigo.hpp"
#include <iostream>
#include "type_traits"


/*!
 * @brief DimensionIterator is an iterator class written for SparseTensor class to facilitate the traversal of the modes of multi-dimensional SparseTensors.
 */
template <bool isConst>
class DimensionIterator
{
    friend class SparseTensor;
public:
    using TensorPointerType = std::conditional_t<isConst, const SparseTensor*, SparseTensor*>;
    using ReferenceType = std::conditional_t<isConst, const vType&, vType&>;

    /*!
     * @brief Iterator Constructor.
     * @param tensor Pointer to the tensor container on which iteration will be conducted.
     * @param dimension The dimension or the order desired to be iterated through.
     */
    DimensionIterator(TensorPointerType tensor, vType dimension);

    /*!
     * @brief Copy Constructor.
     */
    DimensionIterator(const DimensionIterator& other);

    /*!
     * @brief Copy Operator.
     */
    DimensionIterator& operator=(const DimensionIterator& other);

    /*!
     * @brief Move Constructor is unavailable.
     */
    DimensionIterator(const DimensionIterator&& other) = delete;

    /*!
     * @brief Move Operator is unavailable.
     */
    DimensionIterator& operator=(const DimensionIterator&& other) = delete;

    /*!
     * @brief Default compiler-generated destructor.
     */
    ~DimensionIterator() = default;

    /*!
     * @brief Dereference Operator.
     */
    ReferenceType operator*();

    /*!
     * @brief Subscript Operator.
     * @param count after the current element to be dereferenced.
     * @throws std::out_of_range whenever the address tried to be dereferenced is not within the bounds of the container's allocated memory space.
     */
    ReferenceType operator[](eType count);

    /*!
     * @brief Equality Operator.
     * @param other dimension operator with which the equality comparison will be made.
     * @throws std::out_of_range whenever the address tried to be dereferenced is not within the bounds of the container's allocated memory space.
     */
    bool operator==(const DimensionIterator& other);

    /*!
     * @brief Inequality Operator.
     * @param other dimension operator with which the inequality comparison will be made.
     */
    bool operator!=(const DimensionIterator& other);

    /*!
     * @brief Increment Operator.
     * @details Dimension iterator will point to the one past of the last element of the container if it is advanced past the end of the container.
     */
    DimensionIterator& operator++();

    /*!
     * @brief Decrement Operator.
     * @details Dimension iterator will point to the beginning of the container if it is moved before the beginning of the container.
     */
    DimensionIterator& operator--();

    /*!
     * @brief Addition Assignment Operator.
     * @param count The number of times the iterator will be advanced.
     * @return Reference to the iterator after being advanced 'count' times.
     * @details Dimension iterator will point to the one past of the last element of the container if it is advanced past the end of the container.
     */
    DimensionIterator& operator+=(eType count);

    /*!
     * @brief Subtraction Assignment Operator.
     * @param count The number of times the iterator will be moved backwards.
     * @return Reference to the iterator after being moved backwards 'count' times.
     * @details Dimension iterator will point to the beginning of the container if it is moved before the beginning of the container.
     */
    DimensionIterator& operator-=(eType count);

    /*!
     * @brief Addition Operator.
     * @param count The number of positions to advance the iterator.
     * @return A new iterator advanced 'count' positions from the current one.
     * @details Dimension iterator will point to the one past of the last element of the container if it is advanced past the end of the container.
     */
    DimensionIterator operator+(eType count);

    /*!
     * @brief Subtraction Operator.
     * @param count The number of positions to move the iterator backwards.
     * @return A new iterator moved 'count' positions backward from the current one.
     * @details Dimension iterator will point to the beginning of the container if it is moved before the beginning of the container.
     */
    DimensionIterator operator-(eType count);

private:
    /*!
     * @brief Checks if the iterator is at the beginning after moving backward 'count' times.
     * @param count The number of positions to move the iterator backward for the check.
     * @return True if moving the iterator backward 'count' times places it at the beginning of the container, false otherwise.
     */
    bool checkBegin(eType count);

    /*!
     * @brief Checks if the iterator is advanced past the end of the container after moving forward 'count' times.
     * @param count The number of positions to move the iterator forward for the check.
     * @return True if moving the iterator forward 'count' times places it out of the bounds of the container, false otherwise.
     */
    bool checkEnd(eType count);

private:
    TensorPointerType m_Tensor;
    vType m_Order;
    vType m_Dimension;
    eType m_CurrentNNZ;
    vType* m_Current;
    bool m_EndFlag;
};

/**
 * @brief SparseTensor class represents a sparse tensor data structure in SparseViz library.
 */
class SparseTensor
{
    friend class DimensionIterator<true>;
    friend class DimensionIterator<false>;
public:
    typedef DimensionIterator<false> NonConstDimensionIterator;
    typedef DimensionIterator<true> ConstDimensionIterator;
    typedef pigo::Tensor<vType, eType, vType *, valType, valType *, true> pigoTensorType;

    /**
     * @brief Constructs a SparseTensor with a given PIGO tensor.
     * @param name The name of the tensor.
     * @param pigoTensor Pointer to the PIGO tensor object.
     */
    SparseTensor(std::string name, pigoTensorType* pigoTensor);

    /**
     * @brief Constructs a SparseTensor with detailed specifications.
     * @param name The name of the tensor.
     * @param order The order of the tensor.
     * @param dims Array containing the dimensions of the tensor.
     * @param nnzCount The number of non-zero elements in the tensor.
     * @param storage Array containing the storage indices of non-zero elements.
     * @param weights Array containing the weights/values of non-zero elements.
     */
    SparseTensor(std::string name, vType order, vType* dims, eType nnzCount, vType* storage, valType* weights);

    /**
     * @brief Constructs a SparseTensor with the specified name, order, dimensions, and non-zero count.
     * @param name The name of the tensor.
     * @param order The order of the tensor.
     * @param dims Array containing the dimensions of the tensor.
     * @param nnzCount The number of non-zero elements in the tensor.
     */
    SparseTensor(std::string name, vType order, vType* dims, eType nnzCount);

    /**
     * @brief Copy constructor.
     * @param other The SparseTensor object to be copied.
     */
    SparseTensor(const SparseTensor& other);

    /**
     * @brief Copy assignment operator.
     * @param other The SparseTensor object to be assigned.
     * @return Reference to the assigned SparseTensor object.
     */
    SparseTensor& operator=(const SparseTensor& other);

    /**
     * @brief Move constructor.
     * @param other The SparseTensor object to be moved.
     */
    SparseTensor(SparseTensor&& other);

    /**
     * @brief Move assignment operator.
     * @param other The SparseTensor object to be moved and assigned.
     * @return Reference to the moved and assigned SparseTensor object.
     */
    SparseTensor& operator=(SparseTensor&& other);

    /**
     * @brief Destructor for SparseTensor.
     */
    ~SparseTensor();

    /**
     * @brief Returns a non-const iterator to the beginning of the specified dimension.
     * @param dimension The dimension to iterate over.
     * @return NonConstDimensionIterator to the start of the specified dimension.
     */
    NonConstDimensionIterator begin(vType dimension);

    /**
     * @brief Returns a non-const iterator to the end of the specified dimension.
     * @param dimension The dimension to iterate over.
     * @return NonConstDimensionIterator to the end of the specified dimension.
     */
    NonConstDimensionIterator end(vType dimension);

    /**
     * @brief Returns a const iterator to the beginning of the specified dimension.
     * @param dimension The dimension to iterate over.
     * @return ConstDimensionIterator to the start of the specified dimension.
     */
    ConstDimensionIterator begin(vType dimension) const;

    /**
     * @brief Returns a const iterator to the end of the specified dimension.
     * @param dimension The dimension to iterate over.
     * @return ConstDimensionIterator to the end of the specified dimension.
     */
    ConstDimensionIterator end(vType dimension) const;

    /**
     * @brief Saves the tensor to a binary file (ending with .tnx.bin).
     * @param filename The name of the binary file to save the tensor to.
     */
    void save(const std::string& filename);

    /**
     * @brief Frees the resources used by the tensor.
     */
    void free();

    /**
     * @brief Generates a new SparseTensor ordered according to the specified orders.
     * @param orders A pointer to an array of orders.
     * @param orderingName A string representing the name of the ordering.
     * @return A new SparseTensor object with elements ordered as specified.
     * @param active_modes The modes to be ordered
     */
    SparseTensor* generateOrderedTensor(vType** orders, const std::string& orderingName, const std::vector<vType>& active_modes) const;

    // Getters
    vType getOrder() const {return m_Order;}
    vType* getDims() const {return m_Dims;}
    std::string getName() const {return m_Name;}
    eType getNNZCount() const {return m_NNZCount;}
    vType* getStorage() const {return m_Storage;}
    valType* getWeights() const {return m_Weights;}

private:
    /*!
     * @brief Conduct deep copy operation with the other tensor.
     * @param other sparse tensor that the deep copy operation will be conducted with.
     */
    void deepCopy(const SparseTensor& other);

private:
    pigoTensorType* m_PigoTensor;
    std::string m_Name;
    eType m_NNZCount;
    vType m_Order;
    vType* m_Dims;
    vType* m_Storage;
    valType* m_Weights;
};

#endif //SPARSEVIZ_SPARSETENSOR_H
