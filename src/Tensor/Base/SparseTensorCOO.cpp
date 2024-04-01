//
// Created by deniz on 3/26/24.
//

#include "SparseTensorCOO.h"
#include "SparseMatrix.h"
#include <unordered_set>
#include "sort.h"


SparseTensorCOO::SparseTensorCOO(std::string name, SparseTensorCOO::pigoTensorType *pigoTensor)
:   SparseTensor(COO, name),
    m_PigoTensor(pigoTensor)
{
    m_Order = m_PigoTensor->order();
    m_NNZ = m_PigoTensor->m();
    m_Storage = m_PigoTensor->c();
    m_Vals = m_PigoTensor->w();
    m_Dims = m_PigoTensor->getDims();
}

SparseTensorCOO::SparseTensorCOO(std::string name, vType order, vType* dims, eType nnz)
:   SparseTensor(COO, name, order, dims, nnz)
{
    m_Storage = new vType[m_NNZ * m_Order];
    m_PigoTensor = new pigoTensorType();

    m_PigoTensor->m() = m_NNZ;
    m_PigoTensor->order() = m_Order;
    m_PigoTensor->getDims() = m_Dims;
    m_PigoTensor->c() = m_Storage;
    m_PigoTensor->w() = m_Vals;
}

void SparseTensorCOO::save(const std::string &filename)
{
    m_PigoTensor->save(filename);
}

void SparseTensorCOO::free()
{
    delete[] m_Storage;
    delete[] m_Dims;
    delete[] m_Vals;

    m_Storage = nullptr;
    m_Dims = nullptr;
    m_Vals = nullptr;

    m_PigoTensor->c() = nullptr;
    m_PigoTensor->w() = nullptr;
    m_PigoTensor->getDims() = nullptr;
    m_PigoTensor->free();

    delete m_PigoTensor;
    m_PigoTensor = nullptr;
}

void SparseTensorCOO::deepCopy(const SparseTensor *other)
{
    const SparseTensorCOO* otherCOO = dynamic_cast<const SparseTensorCOO*>(other);

    m_PigoTensor = new pigoTensorType(*otherCOO->m_PigoTensor);
    m_Storage = m_PigoTensor->c();
}

void SparseTensorCOO::moveResources(SparseTensor *other)
{
    SparseTensorCOO* otherCOO = dynamic_cast<SparseTensorCOO*>(other);

    m_PigoTensor = otherCOO->m_PigoTensor;
    m_Storage = otherCOO->m_Storage;

    otherCOO->m_PigoTensor = nullptr;
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

    for(vType m = m_Order - 1; m >= 0; m--)
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
        eType nnzStart = i * 2;
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

    return matrix;
}
