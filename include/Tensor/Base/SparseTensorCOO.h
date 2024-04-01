//
// Created by deniz on 3/26/24.
//

#ifndef SPARSEVIZ_SPARSETENSORCOO_H
#define SPARSEVIZ_SPARSETENSORCOO_H

#include "pigo.hpp"
#include "SparseTensor.h"


class SparseTensorCOO: public SparseTensor
{
public:
    typedef pigo::Tensor<vType, eType, vType *, valType, valType *, true> pigoTensorType;

    SparseTensorCOO(std::string name, pigoTensorType* pigoTensor);
    SparseTensorCOO(std::string name, vType order, vType* dims, eType nnz);

    virtual void save(const std::string& filename) final;
    virtual void free() final;
    virtual SparseTensor* generateOrderedTensor(vType** orders, const std::string& orderingName, const std::vector<vType>& active_modes) const final;
    virtual SparseMatrix* matricizeTensor(vType mode) const final;

    // Getters
    [[maybe_unused]] [[nodiscard]] pigoTensorType* getPigo() const {return m_PigoTensor;}
    [[maybe_unused]] [[nodiscard]] vType* getStorage() const {return m_Storage;}

private:
    virtual void deepCopy(const SparseTensor* other) final;
    virtual void moveResources(SparseTensor* other) final;

private:
    // Pigo
    pigoTensorType* m_PigoTensor;

    // Coordinates
    vType* m_Storage;
};


#endif //SPARSEVIZ_SPARSETENSORCOO_H
