//
// Created on 7 August. 2023.
//

#include "TrivialOrdering.h"
#include <vector>
#include "Parameters.h"
#include "helpers.h"
#include <cstdlib>
#include "sort.h"


TrivialOrdering::TrivialOrdering(SparseMatrix &matrix, std::string orderingName, std::string orderingParameters)
:   MatrixOrdering(matrix, orderingName, true, true, 2)
{
    std::vector<std::string> orderingParametersSplitted = split(orderingParameters, '/');
    std::vector<std::string> parameters;

    for (const auto& orderingParameter: orderingParametersSplitted)
    {
        if (lowerString(orderingParameter) == "inc")
        {
            m_SortingDirection = Parameters::INCREASING;
        }
        else if (lowerString(orderingParameter) == "dec")
        {
            m_SortingDirection = Parameters::DECREASING;
        }
        else if (lowerString(orderingParameter) == "r")
        {
            m_OrderingOrientation = Parameters::ROW;
        }
        else if (lowerString(orderingParameter) == "c")
        {
            m_OrderingOrientation = Parameters::COLUMN;
        }
        else if (lowerString(orderingParameter) == "rc")
        {
            m_OrderingOrientation = Parameters::ROWANDCOLUMN;
        }
        else if (lowerString(orderingParameter) == "abs")
        {
            m_OrderingCriteria = Parameters::ABSSUM;
        }
        else if (lowerString(orderingParameter) == "nnz")
        {
            m_OrderingCriteria = Parameters::NNZCOUNT;
        }
    }
}

void TrivialOrdering::orderingFunction()
{
    unsigned n = this->getMatrix().getRowCount();
    unsigned m = this->getMatrix().getColCount();

    unsigned* ptrs = this->getMatrix().getPtr();
    unsigned* ids = this->getMatrix().getInd();
    double* vals = this->getMatrix().getValues();

    if (rowIPermutation || colIPermutation) return;

    rowIPermutation = new vType[n];
    colIPermutation = new vType[m];

    for (unsigned i = 0; i != n; ++i) rowIPermutation[i] = i;
    for (unsigned i = 0; i != m; ++i) colIPermutation[i] = i;
    
    if (m_OrderingCriteria == Parameters::ABSSUM)
    {
        if (m_OrderingOrientation == Parameters::ROW || m_OrderingOrientation == Parameters::ROWANDCOLUMN)
        {
            std::vector<std::pair<vType, valType>> rowAbsSums(n, std::pair<vType, valType>(0, 0));
            for (unsigned i = 0; i != n; ++i)
            {
                for (unsigned ptr = ptrs[i]; ptr != ptrs[i+1]; ++ptr)
                {
                    rowAbsSums[i].first = i;
                    rowAbsSums[i].second += fabs(vals[ptr]);
                }
            }

            if (m_SortingDirection == Parameters::INCREASING)
            {
                std::qsort(rowAbsSums.data(), rowAbsSums.size(), sizeof(std::pair<vType, valType>), ascendingComparator);
            }
            else
            {
                std::qsort(rowAbsSums.data(), rowAbsSums.size(), sizeof(std::pair<vType, valType>), descendingComparator);
            }

            for (int i = 0; i != this->getMatrix().getRowCount(); ++i)
            {
                rowIPermutation[rowAbsSums[i].first] = i;
            }
        }

        if (m_OrderingOrientation == Parameters::COLUMN || m_OrderingOrientation == Parameters::ROWANDCOLUMN)
        {
            std::vector<std::pair<vType, valType>> colAbsSums(m, std::pair<vType, valType>(0, 0));
            for (unsigned i = 0; i != n; ++i)
            {
                for (unsigned ptr = ptrs[i]; ptr != ptrs[i+1]; ++ptr)
                {
                    colAbsSums[ids[ptr]].first = ids[ptr];
                    colAbsSums[ids[ptr]].second += fabs(vals[ptr]);
                }
            }

            if (m_SortingDirection == Parameters::INCREASING)
            {
                std::qsort(colAbsSums.data(), colAbsSums.size(), sizeof(std::pair<vType, valType>), ascendingComparator);
            }
            else
            {
                std::qsort(colAbsSums.data(), colAbsSums.size(), sizeof(std::pair<vType, valType>), descendingComparator);
            }

            for (unsigned i = 0; i != m; ++i)
            {
                colIPermutation[colAbsSums[i].first] = i;
            }
        }
    }
    else if (m_OrderingCriteria == Parameters::NNZCOUNT)
    {
        if (m_OrderingOrientation == Parameters::ROW || m_OrderingOrientation == Parameters::ROWANDCOLUMN)
        {
            std::vector<std::pair<vType, vType>> rowNNZCounts(n, std::pair<vType, vType>(0, 0));
            for (unsigned i = 0; i != n; ++i)
            {   
                rowNNZCounts[i].second = ptrs[i+1] - ptrs[i];

                for (int ptr = ptrs[i]; ptr != ptrs[i+1]; ++ptr)
                {
                    rowNNZCounts[i].first = i;
                }
            }

            countSort(rowNNZCounts, rowIPermutation);
            if (m_SortingDirection == Parameters::DECREASING)
            {
               for(unsigned i = 0; i < n; i++) {
                    rowIPermutation[i] = n - rowIPermutation[i] - 1;
               }
            }
        }
        if (m_OrderingOrientation == Parameters::COLUMN || m_OrderingOrientation == Parameters::ROWANDCOLUMN)
        {
            std::vector<std::pair<vType, vType>> colNNZCounts(m, std::pair<vType, vType>(0, 0));
            for (unsigned i = 0; i != n; ++i)
            {   
                for (int ptr = ptrs[i]; ptr != ptrs[i+1]; ++ptr)
                {
                    colNNZCounts[ids[ptr]].first = ids[ptr];
                    ++colNNZCounts[ids[ptr]].second;
                }
            }

            countSort(colNNZCounts, colIPermutation);
            if (m_SortingDirection == Parameters::DECREASING)
            {
               for(unsigned i = 0; i < m; i++) {
                    colIPermutation[i] = m - colIPermutation[i] - 1;
               }
            }
        }
    }
}

int TrivialOrdering::ascendingComparator(const void *a, const void *b)
{
    auto x = static_cast<const std::pair<vType, valType>*>(a);
    auto y = static_cast<const std::pair<vType, valType>*>(b);

    if (x->second > y->second)
    {
        return 1;
    }
    else if (x->second < y->second)
    {
        return -1;
    }
    return 0;
}

int TrivialOrdering::descendingComparator(const void *a, const void *b)
{
    auto x = static_cast<const std::pair<vType, valType>*>(a);
    auto y = static_cast<const std::pair<vType, valType>*>(b);

    if (x->second < y->second)
    {
        return 1;
    }
    else if (x->second > y->second)
    {
        return -1;
    }
    return 0;
}
