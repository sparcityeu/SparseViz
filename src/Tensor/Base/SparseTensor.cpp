//
// Created on 1/14/24.
//

#include "SparseTensor.h"
#include "SparseMatrix.h"
#include <unordered_set>


template <bool isConst>
DimensionIterator<isConst>::DimensionIterator(TensorPointerType tensor, vType dimension)
:   m_Tensor(tensor),
    m_Order(tensor->m_Order),
    m_Dimension(dimension),
    m_CurrentNNZ(1),
    m_Current(&tensor->m_Storage[dimension]),
    m_EndFlag(false)
{}

template <bool isConst>
DimensionIterator<isConst>::DimensionIterator(const DimensionIterator &other)
:   m_Tensor(other.m_Tensor),
    m_Order(other.m_Order),
    m_Dimension(other.m_Dimension),
    m_CurrentNNZ(other.m_CurrentNNZ),
    m_Current(other.m_Current),
    m_EndFlag(other.m_EndFlag)
{}

template <bool isConst>
DimensionIterator<isConst>& DimensionIterator<isConst>::operator=(const DimensionIterator &other)
{
    if (this == &other)
    {
        return *this;
    }

    m_Tensor = other.m_Tensor;
    m_Order = other.m_Order;
    m_Dimension = other.m_Dimension;
    m_Current = other.m_Current;
    m_CurrentNNZ = other.m_CurrentNNZ;
    m_EndFlag = other.m_EndFlag;

    return *this;
}

template <bool isConst>
bool DimensionIterator<isConst>::operator==(const DimensionIterator &other)
{
    if (m_EndFlag && other.m_EndFlag)
    {
        return m_Tensor == other.m_Tensor && m_Dimension == other.m_Dimension;
    }
    return m_Tensor == other.m_Tensor && m_Dimension == other.m_Dimension && m_EndFlag == other.m_EndFlag && m_Current == other.m_Current;
}

template <bool isConst>
bool DimensionIterator<isConst>::operator!=(const DimensionIterator &other)
{
    return !(*this == other);
}

template <bool isConst>
DimensionIterator<isConst>::ReferenceType DimensionIterator<isConst>::operator*()
{
    if (!m_EndFlag)
    {
        return *m_Current;
    }
    throw std::out_of_range("Try of dereferencing an element that is not within the bounds of the array");
}

template <bool isConst>
DimensionIterator<isConst>::ReferenceType DimensionIterator<isConst>::operator[](eType count)
{
    if (!checkEnd(count))
    {
        return *(m_Current + m_Order * count);
    }
    throw std::out_of_range("Try of accessing an element that is not within the bounds of the array");
}

template <bool isConst>
DimensionIterator<isConst>& DimensionIterator<isConst>::operator++()
{
    if (!checkEnd(1))
    {
        m_Current += m_Order;
        ++m_CurrentNNZ;
    }
    return *this;
}

template <bool isConst>
DimensionIterator<isConst>& DimensionIterator<isConst>::operator--()
{
    if (!checkBegin(1))
    {
        m_Current -= m_Order;
        --m_CurrentNNZ;
    }
    return *this;
}

template <bool isConst>
DimensionIterator<isConst>& DimensionIterator<isConst>::operator+=(eType count)
{
    if (!checkEnd(count))
    {
        m_Current += m_Order * count;
        m_CurrentNNZ += count;
    }
    return *this;
}

template <bool isConst>
DimensionIterator<isConst>& DimensionIterator<isConst>::operator-=(eType count)
{
    if (!checkBegin(count))
    {
        m_Current -= m_Order * count;
        m_CurrentNNZ -= count;
    }
    return *this;
}

template <bool isConst>
DimensionIterator<isConst> DimensionIterator<isConst>::operator+(eType count)
{
    DimensionIterator ret(*this);
    ret += m_Order * count;
    return ret;
}

template <bool isConst>
DimensionIterator<isConst> DimensionIterator<isConst>::operator-(eType count)
{
    DimensionIterator ret(*this);
    ret -= m_Order * count;
    return ret;
}

template <bool isConst>
bool DimensionIterator<isConst>::checkBegin(eType count)
{
    // to disqualify the iterators that past the end of the container
    if (m_EndFlag)
    {
        return true;
    }

    if (m_CurrentNNZ - count > 0)
    {
        return false;
    }
    m_CurrentNNZ = 1;
    m_Current = &m_Tensor->m_Storage[m_Dimension];
    return true;
}

template <bool isConst>
bool DimensionIterator<isConst>::checkEnd(eType count)
{
   if (m_CurrentNNZ + count <= m_Tensor->m_NNZCount)
   {
       return false;
   }
   m_EndFlag = true;
   return true;
}


SparseTensor::SparseTensor(std::string name, vType order, vType* dims, eType nnzCount)
:   m_Name(name),
    m_NNZCount(nnzCount),
    m_Order(order)
{
    m_Storage = new vType[nnzCount * m_Order];
    m_Weights = new valType[nnzCount];
    m_Dims = new vType[m_Order];
    for (int i = 0; i != m_Order; ++i)
    {
        m_Dims[i] = dims[i];
    }

    m_PigoTensor = new pigoTensorType();
    m_PigoTensor->m() = m_NNZCount;
    m_PigoTensor->order() = m_Order;
    m_PigoTensor->getDims() = m_Dims;
    m_PigoTensor->c() = m_Storage;
    m_PigoTensor->w() = m_Weights;
}

SparseTensor::SparseTensor(std::string name, vType order, vType* dims, eType nnzCount, vType* storage, valType* weights)
:   m_Name(name),
    m_NNZCount(nnzCount),
    m_Order(order),
    m_Storage(storage),
    m_Weights(weights)
{
    m_Dims = new vType[m_Order];
    for (int i = 0; i != m_Order; ++i)
    {
        m_Dims[i] = dims[i];
    }

    m_PigoTensor = new pigoTensorType();
    m_PigoTensor->m() = m_NNZCount;
    m_PigoTensor->order() = m_Order;
    m_PigoTensor->getDims() = m_Dims;
    m_PigoTensor->c() = m_Storage;
    m_PigoTensor->w() = m_Weights;
}

SparseTensor::SparseTensor(std::string name, pigoTensorType* pigoTensor)
:   m_Name(name),
    m_PigoTensor(pigoTensor)
{
    m_Order = m_PigoTensor->order();
    m_NNZCount = m_PigoTensor->m();
    m_Storage = m_PigoTensor->c();
    m_Weights = m_PigoTensor->w();
    m_Dims = pigoTensor->getDims();
}

void SparseTensor::free()
{
    delete[] m_Storage;
    delete[] m_Weights;
    delete[] m_Dims;

    m_Storage = nullptr;
    m_Weights = nullptr;
    m_Dims = nullptr;

    m_PigoTensor->c() = nullptr;
    m_PigoTensor->w() = nullptr;
    m_PigoTensor->getDims() = nullptr;
    m_PigoTensor->free();

    delete m_PigoTensor;
    m_PigoTensor = nullptr;
}

SparseTensor::~SparseTensor()
{
    this->free();
}

void SparseTensor::deepCopy(const SparseTensor &other)
{
    m_Name = other.m_Name;
    m_Order = other.m_Order;
    m_NNZCount = other.m_NNZCount;

    m_PigoTensor = new pigoTensorType(*other.m_PigoTensor);

    m_Dims = m_PigoTensor->getDims();
    m_Storage = m_PigoTensor->c();
    m_Weights = m_PigoTensor->w();
}

SparseTensor::SparseTensor(const SparseTensor &other) 
{
    this->deepCopy(other);
}

SparseTensor &SparseTensor::operator=(const SparseTensor &other)
{
    if (this == &other)
    {
        return *this;
    }

    this->free();
    this->deepCopy(other);

    return *this;
}

SparseTensor::SparseTensor(SparseTensor &&other)
:   m_Name(other.m_Name),
    m_NNZCount(other.m_NNZCount),
    m_Order(other.m_Order),
    m_Dims(other.m_Dims),
    m_Storage(other.m_Storage),
    m_Weights(other.m_Weights),
    m_PigoTensor(other.m_PigoTensor)
{
    other.m_Storage = nullptr;
    other.m_Dims = nullptr;
    other.m_Weights = nullptr;
    other.m_PigoTensor = nullptr;
}

SparseTensor &SparseTensor::operator=(SparseTensor &&other)
{
    if (this == &other)
    {
        return *this;
    }

    this->free();

    m_Name = other.m_Name;
    m_Order = other.m_Order;
    m_Dims = other.m_Dims;
    m_NNZCount = other.m_NNZCount;
    m_Storage = other.m_Storage;
    m_Weights = other.m_Weights;
    m_PigoTensor = other.m_PigoTensor;

    other.m_Storage = nullptr;
    other.m_Dims = nullptr;
    other.m_Weights = nullptr;
    other.m_PigoTensor = nullptr;

    return *this;
}

SparseTensor::NonConstDimensionIterator SparseTensor::begin(vType dimension)
{
    return {this, dimension};
}

SparseTensor::NonConstDimensionIterator SparseTensor::end(vType dimension)
{
    NonConstDimensionIterator end(this, dimension);
    end.m_EndFlag = true;
    return end;
}

SparseTensor::ConstDimensionIterator SparseTensor::begin(vType dimension) const
{
    return {this, dimension};
}

SparseTensor::ConstDimensionIterator SparseTensor::end(vType dimension) const
{
    ConstDimensionIterator end(this, dimension);
    end.m_EndFlag = true;
    return end;
}

void SparseTensor::save(const std::string &filename)
{
    m_PigoTensor->save(filename);
}

SparseTensor* SparseTensor::generateOrderedTensor(vType** orders, const std::string& orderingName, const std::vector<vType>& active_modes) const
{
    SparseTensor* orderedTensor = new SparseTensor(orderingName + "_ordered_" + this->getName(), m_Order, m_Dims, m_NNZCount);

    for(int m = 0; m < m_Order; m++) {
       if ( std::find(active_modes.begin(), active_modes.end(), m) == active_modes.end() ){
            for(vType v = 0; v < m_Dims[m]; v++) {
                orders[m][v] = v;
            }
       }
    }

    eType* source_permutation = new eType[m_NNZCount];
    for(eType i = 0; i < m_NNZCount; i++) source_permutation[i] = i;
    
    /*for(eType e = 0; e < 50; e++) {
        std::cout <<  e << ": " << source_permutation[e] << " - ";
        for(int m = 0; m < m_Order; m++) {
            std::cout << m_Storage[source_permutation[e] * m_Order + m] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------" << std::endl;*/

    eType* target_permutation = new eType[m_NNZCount];

    for(int m = m_Order - 1; m >= 0; m--) {
        vType* ordering = orders[m];
        vType mdim = m_Dims[m];

        vType* locs = new vType[mdim + 1];
        memset(locs, 0, sizeof(vType) * (mdim + 1));
        
        for(eType e = 0; e < m_NNZCount; e++) {
            vType org_id = m_Storage[(e * m_Order) + m];
            vType ordered_id = ordering[org_id];
            locs[ordered_id + 1]++;
        }
        for(vType v = 2; v <= mdim; v++) locs[v] += locs[v-1];
        
        for(eType e = 0; e < m_NNZCount; e++) {
            vType org_id = m_Storage[(source_permutation[e] * m_Order) + m];
            vType ordered_id = ordering[org_id];
            target_permutation[locs[ordered_id]++] = source_permutation[e];
        }
        delete [] locs;

        eType* temp = source_permutation; 
        source_permutation = target_permutation; 
        target_permutation = temp;

        /*for(eType e = 0; e < 100; e++) {
            std::cout <<  e << ": " << source_permutation[e] << " - ";
            for(int m = 0; m < m_Order; m++) {
                std::cout << orders[m][m_Storage[source_permutation[e] * m_Order + m]] << " ";
            }
            std::cout << std::endl;
        }*/
    }

    vType* target_ptr =  orderedTensor->m_Storage;
    for(eType e = 0; e < m_NNZCount; e++) {
        vType* source_ptr = m_Storage + (source_permutation[e] * m_Order);
        for(int m = 0; m < m_Order; m++) {
            *target_ptr++ = orders[m][source_ptr[m]];
        }
        orderedTensor->m_Weights[e] = m_Weights[source_permutation[e]];
    }

    /*for(eType e = 0; e < 500; e++) {
        std::cout <<  e << ": ";
        for(int m = 0; m < m_Order; m++) {
            std::cout << orderedTensor->m_Storage[e * m_Order + m] << " ";
        }
        std::cout << std::endl;
    }*/

    delete [] source_permutation;
    delete [] target_permutation;
    return orderedTensor;
}
