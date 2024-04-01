//
// Created on 1/14/24.
//

#ifndef SPARSEVIZ_SPARSETENSOR_H
#define SPARSEVIZ_SPARSETENSOR_H

#include "config.h"
#include <string>
#include <vector>
#include <iostream>
#include "type_traits"


class SparseMatrix;

class SparseTensor
{
public:
    SparseTensor(TensorType tensorType, std::string name);
    SparseTensor(TensorType tensorType, std::string name, vType order, vType* dims, eType nnz);
    SparseTensor(const SparseTensor& other);
    SparseTensor& operator=(const SparseTensor& other);
    SparseTensor(SparseTensor&& other);
    SparseTensor& operator=(SparseTensor&& other);
    virtual ~SparseTensor();

    virtual void save(const std::string& filename) = 0;
    virtual void free() = 0;
    virtual SparseTensor* generateOrderedTensor(vType** orders, const std::string& orderingName, const std::vector<vType>& active_modes) const = 0;
    virtual SparseMatrix* matricizeTensor(vType mode) const = 0;

    // Getters
    [[maybe_unused]] [[nodiscard]] vType getOrder() const {return m_Order;}
    [[maybe_unused]] [[nodiscard]] vType* getDims() const {return m_Dims;}
    [[maybe_unused]] [[nodiscard]] std::string getName() const {return m_Name;}
    [[maybe_unused]] [[nodiscard]] TensorType getTensorType() const {return m_TensorType;};
    [[maybe_unused]] [[nodiscard]] eType getNNZ() const {return m_NNZ;}
    [[maybe_unused]] [[nodiscard]] valType* getValues() const {return m_Vals;}

protected:
    virtual void deepCopy(const SparseTensor* other) = 0;
    virtual void moveResources(SparseTensor* other) = 0;

private:
    void baseDeepCopy(const SparseTensor& other);
    void baseMoveResources(SparseTensor&& other);

protected:
    // Tensor Metadata
    TensorType m_TensorType;
    std::string m_Name;
    eType m_NNZ;
    vType m_Order;
    vType* m_Dims;

    // Value array
    valType* m_Vals;
};


#endif //SPARSEVIZ_SPARSETENSOR_H
