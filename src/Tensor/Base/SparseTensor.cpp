//
// Created on 1/14/24.
//

#include "SparseTensor.h"


SparseTensor::SparseTensor(TensorType tensorType, std::string name)
:   m_TensorType(tensorType),
    m_Name(name)
{

}

SparseTensor::SparseTensor(TensorType tensorType, std::string name, vType order, vType* dims, eType nnz)
:   m_TensorType(tensorType),
    m_Name(name),
    m_NNZ(nnz),
    m_Order(order)
{
    m_Vals = new valType[m_NNZ];
    m_Dims = new vType[m_Order];

    for (int i = 0; i != m_Order; ++i)
    {
        m_Dims[i] = dims[i];
    }
}

SparseTensor::~SparseTensor()
{
    delete[] m_Vals;
    delete[] m_Dims;

    m_Vals = nullptr;
    m_Dims = nullptr;
}

SparseTensor::SparseTensor(const SparseTensor &other)
{
    this->baseDeepCopy(other);
}

SparseTensor &SparseTensor::operator=(const SparseTensor &other)
{
    if (this == &other)
    {
        return *this;
    }

    this->free();
    this->baseDeepCopy(other);
    this->deepCopy(&other);

    return *this;
}

SparseTensor::SparseTensor(SparseTensor &&other)
{
    this->baseMoveResources(std::move(other));
}

SparseTensor &SparseTensor::operator=(SparseTensor &&other)
{
    if (this == &other)
    {
        return *this;
    }

    this->free();
    this->baseMoveResources(std::move(other));
    this->moveResources(&other);

    return *this;
}

void SparseTensor::baseDeepCopy(const SparseTensor &other)
{
    m_TensorType = other.m_TensorType;
    m_Name = other.m_Name;
    m_NNZ = other.m_NNZ;
    m_Order = other.m_Order;

    m_Dims = new vType[m_Order];
    memcpy(m_Dims, other.m_Dims, sizeof(vType) * m_Order);

    m_Vals = new valType[m_NNZ];
    memcpy(m_Vals, other.m_Vals, sizeof(valType) * m_NNZ);
}

void SparseTensor::baseMoveResources(SparseTensor &&other)
{
    m_TensorType = other.m_TensorType;
    m_Name = other.m_Name;
    m_NNZ = other.m_NNZ;
    m_Order = other.m_Order;

    m_Dims = other.m_Dims;
    other.m_Dims = nullptr;

    m_Vals = other.m_Vals;
    other.m_Vals = nullptr;
}
