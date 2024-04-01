//
// Created by deniz on 3/22/24.
//

#include "SparseTensorCSF.h"
#include "SparseTensorCOO.h"
#include <iostream>
#include <vector>
#include "helpers.h"


SparseTensorCSF* SparseTensorCSF::constructCSFFromCOO(std::string name, SparseTensorCOO* coo)
{
    SparseTensorCSF* tensor = new SparseTensorCSF(name, coo);
    vType order = tensor->m_Order;
    eType nnz = tensor->m_NNZ;
    vType* storage = coo->getStorage();
    valType* vals = coo->getValues();

    std::vector<std::vector<vType>> fPtrs(order - 1);
    std::vector<std::vector<vType>> fInds(order - 1);

    tensor->m_FInds[order - 1] = new vType[nnz];
    for (eType i = 0; i < nnz; ++i) // for the last index and val array
    {
        tensor->m_NInds[order - 1] = nnz;
        tensor->m_FInds[order - 1][i] = storage[order * i + (order - 1)];
        tensor->m_Vals[i] = vals[i];
    }

    for (vType dim = 0; dim < order - 2; ++dim) // for fptrs and finds
    {
        vType prevUniqueValue = -1;
        vType uniqueCount = 0;
        vType currIndex = -1;
        fPtrs[dim].reserve(nnz);
        fInds[dim].reserve(nnz);
        fPtrs[dim].emplace_back(0);
        for (eType i = 0; i < nnz; ++i)
        {
            vType index = order * i + dim;
            if (storage[index] != currIndex)
            {
                fPtrs[dim].emplace_back(fPtrs[dim].back() + uniqueCount); // implicitly doing a prefix sum
                fInds[dim].emplace_back(storage[index]);
                currIndex = storage[index];
            }
            if (storage[index + 1] != prevUniqueValue)
            {
                ++uniqueCount;
                prevUniqueValue = storage[index + 1];
            }
        }
    }

    for (vType i = 0; i < order - 1; ++i) // copying
    {
        tensor->m_NInds[i] = fInds[i].size();
        tensor->m_FInds[i] = new vType[fInds[i].size()];
        memcpy(tensor->m_FInds[i], fInds[i].data(), sizeof(vType) * fInds[i].size());

        tensor->m_NPtrs[i] = fPtrs[i].size();
        tensor->m_FPtrs[i] = new vType[fPtrs[i].size()];
        memcpy(tensor->m_FPtrs[i], fPtrs[i].data(), sizeof(vType) * fPtrs[i].size());
    }

    return tensor;
}

SparseTensorCSF::SparseTensorCSF(std::string name, SparseTensorCOO* coo)
:   SparseTensor(CSF, name, coo->getOrder(), coo->getDims(), coo->getNNZ()),
    m_COO(coo)
{
    m_NPtrs = new vType[m_Order - 1];
    m_NInds = new vType[m_Order];

    m_FPtrs = new vType*[m_Order - 1];
    m_FInds = new vType*[m_Order];
}

SparseTensorCSF::SparseTensorCSF(const SparseTensorCSF &other)
:   SparseTensor(other)
{
    this->deepCopy(&other);
}

SparseTensorCSF::SparseTensorCSF(SparseTensorCSF &&other)
:   SparseTensor(other)
{
    this->moveResources(&other);
}

SparseTensorCSF::~SparseTensorCSF() noexcept
{
    this->free();
}

void SparseTensorCSF::free()
{
    delete[] m_NPtrs;
    m_NPtrs = nullptr;
    delete[] m_NInds;
    m_NInds = nullptr;

    for (vType i = 0; i < m_Order - 1; ++i)
    {
        delete[] m_FPtrs[i];
        m_FPtrs[i] = nullptr;
    }
    delete[] m_FPtrs;
    m_FPtrs = nullptr;

    for (vType i = 0; i < m_Order; ++i)
    {
        delete[] m_FInds[i];
        m_FInds[i] = nullptr;
    }
    delete[] m_FInds;
    m_FInds = nullptr;

    delete m_COO;
    m_COO = nullptr;
}

void SparseTensorCSF::deepCopy(const SparseTensor* other)
{
    const SparseTensorCSF* otherCSF = dynamic_cast<const SparseTensorCSF*>(other);

    m_NPtrs = new vType[m_Order - 1];
    memcpy(m_NPtrs, otherCSF->m_NPtrs, sizeof(vType) * (m_Order - 1));

    m_NInds = new vType[m_Order];
    memcpy(m_NInds, otherCSF->m_NInds, sizeof(vType) * m_Order);

    m_FPtrs = new vType*[m_Order - 1];
    for (vType i = 0; i < m_Order - 1; ++i)
    {
        m_FPtrs[i] = new vType[m_NPtrs[i]];
        memcpy(m_FPtrs[i], otherCSF->m_FPtrs[i], sizeof(vType) * m_NPtrs[i]);
    }

    m_FInds = new vType*[m_Order];
    for (vType i = 0; i < m_Order; ++i)
    {
        m_FInds[i] = new vType[m_NInds[i]];
        memcpy(m_FInds[i], otherCSF->m_FInds[i], sizeof(vType) * m_NInds[i]);
    }

    m_COO = new SparseTensorCOO(*otherCSF->m_COO);
}

void SparseTensorCSF::moveResources(SparseTensor* other)
{
    SparseTensorCSF* otherCSF = dynamic_cast<SparseTensorCSF*>(other);

    m_NPtrs = otherCSF->m_NPtrs;
    m_NInds = otherCSF->m_NInds;
    m_FPtrs = otherCSF->m_FPtrs;
    m_FInds = otherCSF->m_FInds;
    m_COO = otherCSF->m_COO;

    otherCSF->m_NPtrs = nullptr;
    otherCSF->m_NInds = nullptr;
    otherCSF->m_FPtrs = nullptr;
    otherCSF->m_FInds = nullptr;
    otherCSF->m_COO = nullptr;
}

void SparseTensorCSF::save(const std::string &filename)
{
    std::cerr << "CSF tensors are not saveable currently." << std::endl;
    throw std::runtime_error("CSF tensors are not saveable currently.");
}

SparseTensor *SparseTensorCSF::generateOrderedTensor(vType **orders, const std::string &orderingName, const std::vector<vType> &active_modes) const
{
    SparseTensorCOO* orderedCOO = getCOOFormat(m_COO->generateOrderedTensor(orders, orderingName, active_modes));
    return SparseTensorCSF::constructCSFFromCOO(orderingName + "_ordered_" + m_Name, orderedCOO);
}
