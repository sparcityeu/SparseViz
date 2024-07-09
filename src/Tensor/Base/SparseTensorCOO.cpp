//
// Created by deniz on 3/26/24.
//

#include "SparseTensorCOO.h"
#include "SparseMatrix.h"
#include <unordered_set>
#include "sort.h"


SparseTensorCOO::SparseTensorCOO(std::string name)
:   SparseTensor(COO, name)
{

}

SparseTensorCOO::SparseTensorCOO(std::string name, vType order, vType *dims, eType nnz)
:   SparseTensor(COO, name, order, dims, nnz)
{
    m_Storage = new vType[m_NNZ * m_Order];
}

SparseTensorCOO::SparseTensorCOO(std::string name, vType order, vType* dims, eType nnz, valType* values, vType* storage)
:   SparseTensor(COO, name, order, dims, nnz, values),
    m_Storage(storage)
{

}

SparseTensorCOO::SparseTensorCOO(const SparseTensorCOO &other)
:   SparseTensor(other)
{
    this->deepCopy(&other);
}

SparseTensorCOO::SparseTensorCOO(SparseTensorCOO &&other)
:   SparseTensor(other)
{
    this->moveResources(&other);
}

SparseTensorCOO::~SparseTensorCOO() noexcept
{
    this->free();
}

void SparseTensorCOO::free() noexcept
{
    delete[] m_Storage;
    m_Storage = nullptr;
}

void SparseTensorCOO::deepCopy(const SparseTensor *other)
{
    const SparseTensorCOO* otherCOO = dynamic_cast<const SparseTensorCOO*>(other);

    if (!otherCOO)
    {
        throw std::bad_cast();
    }

    m_Storage = new vType[m_NNZ * m_Order];
    memcpy(m_Storage, otherCOO->m_Storage, sizeof(vType) * m_NNZ * m_Order);
}

void SparseTensorCOO::moveResources(SparseTensor *other) noexcept
{
    SparseTensorCOO* otherCOO = dynamic_cast<SparseTensorCOO*>(other);

    if (!otherCOO)
    {
        throw std::bad_cast();
    }

    m_Storage = otherCOO->m_Storage;

    otherCOO->m_Storage = nullptr;
}

SparseTensor *SparseTensorCOO::generateOrderedTensor(vType **orders, const std::string &orderingName, const std::vector<vType> &active_modes) const
{
    SparseTensorCOO* orderedTensor = new SparseTensorCOO(orderingName + "_ordered_" + this->getName(), m_Order, m_Dims, m_NNZ);

    for(int m = 0; m < m_Order; m++)
    {
        if ( std::find(active_modes.begin(), active_modes.end(), m) == active_modes.end() )
        {
            for(vType v = 0; v < m_Dims[m]; v++)
            {
                orders[m][v] = v;
            }
        }
    }

    eType* source_permutation = new eType[m_NNZ];
    for(eType i = 0; i < m_NNZ; i++) source_permutation[i] = i;

    eType* target_permutation = new eType[m_NNZ];

    for(int m = m_Order - 1; m >= 0; --m)
    {
        vType* ordering = orders[m];
        vType mdim = m_Dims[m];

        vType* locs = new vType[mdim + 1];
        memset(locs, 0, sizeof(vType) * (mdim + 1));

        for(eType e = 0; e < m_NNZ; e++)
        {
            vType org_id = m_Storage[(e * m_Order) + m];
            vType ordered_id = ordering[org_id];
            locs[ordered_id + 1]++;
        }

        for(vType v = 2; v <= mdim; v++) locs[v] += locs[v-1];

        for(eType e = 0; e < m_NNZ; e++)
        {
            vType org_id = m_Storage[(source_permutation[e] * m_Order) + m];
            vType ordered_id = ordering[org_id];
            target_permutation[locs[ordered_id]++] = source_permutation[e];
        }
        delete [] locs;

        eType* temp = source_permutation;
        source_permutation = target_permutation;
        target_permutation = temp;
    }

    vType* target_ptr = orderedTensor->m_Storage;
    for(eType e = 0; e < m_NNZ; e++)
    {
        vType* source_ptr = m_Storage + (source_permutation[e] * m_Order);
        for(int m = 0; m < m_Order; m++)
        {
            *target_ptr++ = orders[m][source_ptr[m]];
        }
        orderedTensor->m_Vals[e] = m_Vals[source_permutation[e]];
    }

    delete [] source_permutation;
    delete [] target_permutation;

    return orderedTensor;
}

SparseMatrix *SparseTensorCOO::matricizeTensor(vType mode) const
{
    // test needed

    vType nRow = m_Dims[mode];
    vType nCol = 1;
    for(vType m = 1; m < m_Order; ++m)
    {
        nCol *= m_Dims[(mode + m) % m_Order];
    }

    vType* storage = new vType[m_NNZ * m_Order];
    memcpy(storage, m_Storage, sizeof(vType) * m_NNZ * m_Order);
    valType* values = new valType[m_NNZ];
    memcpy(values, m_Vals, sizeof(valType) * m_NNZ);
    sortNonzeros(m_Order, m_Dims, m_NNZ, storage, values);

    SparseMatrix* matrix = new SparseMatrix("matricized_" + m_Name, nRow, nCol, m_NNZ);
    vType*& rowPtrs = matrix->getPtr();
    vType*& colInds = matrix->getInd();
    valType*& vals = matrix->getValues();

    vType row = 0;
    for (eType i = 0; i < m_NNZ; ++i)
    {
        eType nnzStart = i * m_Order;
        vType currentRow = storage[nnzStart + mode];
        while (row <= currentRow)
        {
            rowPtrs[row] = i;
            ++row;
        }

        vals[i] = values[i];

        vType col = 0;
        vType mult = 1;
        for (vType m = 0; m < m_Order; ++m)
        {
            vType reverseM = m_Order - 1 - m;
            if (reverseM == mode)
            {
                continue;
            }
            col += storage[nnzStart + reverseM] * mult;
            mult *= m_Dims[reverseM];
        }
        colInds[i] = col;
    }

    for (vType i = row; i <= nRow; ++i)
    {
        rowPtrs[i] = m_NNZ;
    }

    delete[] storage;
    delete[] values;

    return matrix;
}
