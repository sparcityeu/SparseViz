//
// Created by deniz on 3/22/24.
//

#ifndef SPARSEVIZ_SPARSETENSORCSF_H
#define SPARSEVIZ_SPARSETENSORCSF_H

#include "SparseTensor.h"


class SparseTensorCOO;

class SparseTensorCSF: public SparseTensor
{
public:
    static SparseTensorCSF* constructCSFFromCOO(std::string name, SparseTensorCOO* coo);
    SparseTensorCSF(const SparseTensorCSF &other);
    SparseTensorCSF(SparseTensorCSF &&other);
    virtual ~SparseTensorCSF() override;

    virtual void save(const std::string& filename) final;
    virtual void free() final;
    virtual SparseTensor* generateOrderedTensor(vType** orders, const std::string& orderingName, const std::vector<vType>& active_modes) const final;
    virtual SparseMatrix* matricizeTensor(vType mode) const final {}

    // Getters
    [[maybe_unused]] [[nodiscard]] vType* getNPtrs() const {return m_NPtrs;}
    [[maybe_unused]] [[nodiscard]] vType* getNInds() const {return m_NInds;}
    [[maybe_unused]] [[nodiscard]] vType** getFPtrs() const {return m_FPtrs;}
    [[maybe_unused]] [[nodiscard]] vType** getFInds() const {return m_FInds;}
    [[maybe_unused]] [[nodiscard]] SparseTensorCOO* getCOO() const {return m_COO;}

private:
    SparseTensorCSF(std::string name, SparseTensorCOO* coo);

    virtual void deepCopy(const SparseTensor* other) final;
    virtual void moveResources(SparseTensor* other) final;

private:
    // Metadata specific to CSF
    vType* m_NPtrs;
    vType* m_NInds;

    // Storage
    vType** m_FPtrs;
    vType** m_FInds;

    // COO
    SparseTensorCOO* m_COO;
};


#endif //SPARSEVIZ_SPARSETENSORCSF_H
