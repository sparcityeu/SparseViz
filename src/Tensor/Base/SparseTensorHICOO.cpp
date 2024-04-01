//
// Created by deniz on 3/25/24.
//

#include "SparseTensorHICOO.h"
#include "omp.h"
#include "helpers.h"
#include "SparseTensorCOO.h"
#include "sort.h"


SparseTensorHICOO *
SparseTensorHICOO::constructHICOOFromCOO(std::string name, BlockType blockSize, SparseTensorCOO* coo, u_int8_t sbBits)
{
    vType order = coo->getOrder();
    if ((order != 3) && (order != 4))
    {
        throw std::runtime_error("HICOO representation is unavailable for tensors whose order is exceeding 4.");
    }

    SparseTensorHICOO* tensor = new SparseTensorHICOO(name, blockSize, coo, sbBits);
    eType nnz = tensor->m_NNZ;
    vType* storage = coo->getStorage();
    valType* vals = coo->getValues();

    Morton::sort(storage, vals, order, nnz, sbBits);

    vType blockIndex = 0;
    vType offsetIndex = 0;
    BlockType* currentBlockStart = tensor->insertHeadNonzero(0, blockIndex, storage, vals[0]);
    tensor->insertNonzero(0, blockIndex, offsetIndex, currentBlockStart, storage, vals[0]);

    for (eType i = 1; i < nnz; ++i)
    {
        vType* currentCoordinate = &storage[order * i];
        if (tensor->isCorrectBlock(currentBlockStart, currentCoordinate))
        {
            tensor->insertNonzero(i, blockIndex, offsetIndex, currentBlockStart, currentCoordinate, vals[i]);
        }
        else
        {
            delete[] currentBlockStart;
            currentBlockStart = tensor->insertHeadNonzero(i, blockIndex, currentCoordinate, vals[i]);
            tensor->insertNonzero(i, blockIndex, offsetIndex, currentBlockStart, currentCoordinate, vals[i]);
        }
    }
    delete[] currentBlockStart;

    return tensor;
}

SparseTensorHICOO::SparseTensorHICOO(std::string name, BlockType blockSize, SparseTensorCOO* coo, u_int8_t sbBits)
:   SparseTensor(HiCOO, name, coo->getOrder(), coo->getDims(), coo->getNNZ()),
    m_BlockSize(blockSize),
    m_COO(coo),
    m_SbBits(sbBits)
{
    vType blockCount = 0;
    for (vType i = 0; i < m_Order; ++i)
    {
        vType intermediate = std::ceil(m_Dims[i] / blockSize);
        if (intermediate > blockCount)
        {
            blockCount = intermediate;
        }
    }
    m_BlockCount = blockCount;

    m_Ptrs = new vType[m_BlockCount];
    memset(m_Ptrs, 0, sizeof(vType) * m_BlockCount);

    m_BInds = new BlockType[m_BlockCount * m_Order];
    memset(m_BInds, 0, sizeof(BlockType) * m_BlockCount * m_Order);

    m_EInds = new OffsetType[m_NNZ * m_Order];
    memset(m_EInds, 0, sizeof(OffsetType) * m_NNZ * m_Order);
}

SparseTensorHICOO::SparseTensorHICOO(const SparseTensorHICOO &other)
:   SparseTensor(other)
{
    this->deepCopy(&other);
}

SparseTensorHICOO::SparseTensorHICOO(SparseTensorHICOO &&other)
:   SparseTensor(other)
{
    this->moveResources(&other);
}

SparseTensorHICOO::~SparseTensorHICOO() noexcept
{
    this->free();
}

void SparseTensorHICOO::free()
{
    delete[] m_Ptrs;
    m_Ptrs = nullptr;

    delete[] m_BInds;
    m_BInds = nullptr;

    delete[] m_EInds;
    m_EInds = nullptr;

    delete m_COO;
    m_COO = nullptr;
}

void SparseTensorHICOO::save(const std::string &filename)
{
    std::cerr << "HiCOO tensors are not saveable currently." << std::endl;
    throw std::runtime_error("HiCOO tensors are not saveable currently.");
}

SparseTensor *SparseTensorHICOO::generateOrderedTensor(vType **orders, const std::string &orderingName, const std::vector<vType> &active_modes) const
{
    SparseTensorCOO* orderedCOO = getCOOFormat(m_COO->generateOrderedTensor(orders, orderingName, active_modes));
    return SparseTensorHICOO::constructHICOOFromCOO(orderingName + "_ordered_" + m_Name, m_BlockSize, orderedCOO, m_SbBits);
}

void SparseTensorHICOO::deepCopy(const SparseTensor* other)
{
    const SparseTensorHICOO* otherHICOO = dynamic_cast<const SparseTensorHICOO*>(other);

    m_BlockSize = otherHICOO->m_BlockSize;
    m_BlockCount = otherHICOO->m_BlockCount;

    m_Ptrs = new vType[m_BlockCount];
    memcpy(m_Ptrs, otherHICOO->m_Ptrs, sizeof(vType) * m_BlockCount);

    m_BInds = new BlockType[m_BlockCount * m_Order];
    memcpy(m_BInds, otherHICOO->m_BInds, sizeof(BlockType) * m_BlockCount * m_Order);

    m_EInds = new OffsetType[m_NNZ * m_Order];
    memcpy(m_EInds, otherHICOO->m_EInds, sizeof(OffsetType) * m_NNZ * m_Order);

    m_COO = new SparseTensorCOO(*otherHICOO->m_COO);
}

void SparseTensorHICOO::moveResources(SparseTensor* other)
{
    SparseTensorHICOO* otherHICOO = dynamic_cast<SparseTensorHICOO*>(other);

    m_BlockSize = otherHICOO->m_BlockSize;
    m_BlockCount = otherHICOO->m_BlockCount;

    m_Ptrs = otherHICOO->m_Ptrs;
    m_BInds = otherHICOO->m_BInds;
    m_EInds = otherHICOO->m_EInds;
    m_COO = otherHICOO->m_COO;

    otherHICOO->m_Ptrs = nullptr;
    otherHICOO->m_BInds = nullptr;
    otherHICOO->m_EInds = nullptr;
    otherHICOO->m_COO = nullptr;
}

bool SparseTensorHICOO::isCorrectBlock(const BlockType* currentBlock, vType*& coordinate) const
{
    for (vType i = 0; i < m_Order; ++i)
    {
        if ((coordinate[i] / m_BlockSize) != currentBlock[i])
        {
            return false;
        }
    }
    return true;
}

BlockType* SparseTensorHICOO::insertHeadNonzero(const eType& nonzeroNo, vType& blockIndex, vType*& coordinate, valType& value)
{
    BlockType* start = new BlockType[m_Order];

    for (vType i = 0; i < m_Order; ++i)
    {
        start[i] = (coordinate[i] / m_BlockSize);
        m_BInds[blockIndex + i] = start[i];
    }
    blockIndex += m_Order;

    m_Vals[nonzeroNo] = value;

    return start;
}

void SparseTensorHICOO::insertNonzero(const eType& nonzeroNo, vType& blockIndex, vType& offsetIndex, const BlockType* blockStart, vType*& coordinate, valType& value)
{
    for (vType i = 0; i < m_Order; ++i)
    {
        m_EInds[offsetIndex + i] = coordinate[i] - (m_BlockSize * blockStart[i]);
    }
    ++m_Ptrs[blockIndex / m_Order];
    offsetIndex += m_Order;

    m_Vals[nonzeroNo] = value;
}
