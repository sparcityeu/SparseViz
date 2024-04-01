//
// Created by deniz on 3/25/24.
//

#ifndef SPARSEVIZ_SPARSETENSORHICOO_H
#define SPARSEVIZ_SPARSETENSORHICOO_H

#include "SparseTensor.h"
#include <cmath>


class SparseTensorCOO;

class SparseTensorHICOO: public SparseTensor
{
public:
    static SparseTensorHICOO* constructHICOOFromCOO(std::string name, BlockType blockSize, SparseTensorCOO* coo, u_int8_t sbBits);
    SparseTensorHICOO(const SparseTensorHICOO &other);
    SparseTensorHICOO(SparseTensorHICOO &&other);
    virtual ~SparseTensorHICOO() override;

    virtual void save(const std::string& filename) final;
    virtual void free() final;
    virtual SparseTensor* generateOrderedTensor(vType** orders, const std::string& orderingName, const std::vector<vType>& active_modes) const final;
    virtual SparseMatrix* matricizeTensor(vType mode) const final {}

    // Getters
    [[maybe_unused]] [[nodiscard]] BlockType getBlockSize() const {return m_BlockSize;}
    [[maybe_unused]] [[nodiscard]] vType getBlockCount() const {return m_BlockCount;}
    [[maybe_unused]] [[nodiscard]] vType* getPtrs() const {return m_Ptrs;}
    [[maybe_unused]] [[nodiscard]] BlockType* getBInds() const {return m_BInds;}
    [[maybe_unused]] [[nodiscard]] OffsetType* getEInds() const {return m_EInds;}
    [[maybe_unused]] [[nodiscard]] SparseTensorCOO* getCOO() const {return m_COO;}

private:
    SparseTensorHICOO(std::string name, BlockType blockSize, SparseTensorCOO* coo, u_int8_t sbBits);

    virtual void deepCopy(const SparseTensor* other) final;
    virtual void moveResources(SparseTensor* other) final;

    // COO to HICOO Conversion Helpers
    inline bool isCorrectBlock(const BlockType* currentBlock, vType*& coordinate) const;
    BlockType* insertHeadNonzero(const eType& nonzeroNo, vType& blockIndex, vType*& coordinate, valType& value);
    void insertNonzero(const eType& nonzeroNo, vType& blockIndex, vType& offsetIndex, const BlockType* blockStart, vType*& coordinate, valType& value);

private:

    // Metadata specific to HiCOO
    BlockType m_BlockSize;
    vType m_BlockCount;
    u_int8_t m_SbBits;

    // Storage
    vType* m_Ptrs;
    BlockType* m_BInds;
    OffsetType* m_EInds;

    // COO
    SparseTensorCOO* m_COO;
};


#endif //SPARSEVIZ_SPARSETENSORHICOO_H
