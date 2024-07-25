//
// Created by deniz on 3/26/24.
//

#ifndef SPARSEVIZ_SPARSETENSORCOO_H
#define SPARSEVIZ_SPARSETENSORCOO_H

#include "SparseTensor.h"


class SparseTensorCOO: public SparseTensor
{
public:
    SparseTensorCOO(std::string name);
    SparseTensorCOO(std::string name, vType order, vType* dims, eType nnz);
    SparseTensorCOO(std::string name, vType order, vType *dims, eType nnz, valType *values, vType* storage);
    SparseTensorCOO(const SparseTensorCOO& other);
    SparseTensorCOO(SparseTensorCOO&& other);
    virtual ~SparseTensorCOO() noexcept override;

    virtual SparseTensor* generateOrderedTensor(vType** orders, const std::string& orderingName, const std::vector<vType>& active_modes) const final;
    virtual SparseMatrix* matricizeTensor(vType mode) const final;

    // Getters
    [[maybe_unused]] [[nodiscard]] vType* getStorage() const {return m_Storage;}

private:
    virtual void free() noexcept final;
    virtual void deepCopy(const SparseTensor* other) final;
    virtual void moveResources(SparseTensor* other) noexcept final;

private:
    // Coordinates
    vType* m_Storage;
};


#endif //SPARSEVIZ_SPARSETENSORCOO_H
