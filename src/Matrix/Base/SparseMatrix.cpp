#include "SparseMatrix.h"
#include <omp.h>


SparseMatrix::SparseMatrix(std::string name, vType r, vType c, eType nnz, vType* storage, valType* values)
:   m_Name(name),
    m_Row(r),
    m_Col(c),
    m_NNZCount(nnz),
    m_Values(values)
{
    m_OrderingSupportedMatrix = nullptr;

    m_Ptr = new vType[m_Row + 1];
    memset(m_Ptr, 0, sizeof(vType) * (m_Row + 1));

    m_Ind = new vType[m_NNZCount];

    for (vType i = 0; i < nnz; ++i)
    {
        eType nnzStart = i * 2;
        ++m_Ptr[storage[nnzStart] + 1];
        m_Ind[i] = storage[nnzStart + 1];
    }

    for (vType i = 0; i < m_Row; ++i)
    {
        m_Ptr[i + 1] += m_Ptr[i];
    }

    this->checkProperties(false);
}

SparseMatrix::SparseMatrix(std::string name, vType r, vType c, vType nnzCount, vType *ptrs, vType *ids, valType *vals)
:   m_Name(name),
    m_Row(r),
    m_Col(c),
    m_NNZCount(nnzCount),
    m_Ptr(ptrs),
    m_Ind(ids),
    m_Values(vals)
{
    m_OrderingSupportedMatrix = nullptr;

    this->checkProperties(false);
}

SparseMatrix::SparseMatrix(std::string name, vType r, vType c, vType nnzCount, vType *ptrs, vType *ids, valType *vals, bool isSymmetric, bool isPatternSymmetric)
:   m_Name(name),
    m_Row(r),
    m_Col(c),
    m_NNZCount(nnzCount),
    m_Ptr(ptrs),
    m_Ind(ids),
    m_Values(vals),
    m_IsSymmetric(isSymmetric),
    m_IsPatternSymmetric(isPatternSymmetric)
{
    m_OrderingSupportedMatrix = nullptr;

    this->checkShape();
}

SparseMatrix::SparseMatrix(std::string name, vType r, vType c, vType nnzCount, bool isSymmetric, bool isPatternSymmetric)
:   m_Name(name),
    m_Row(r),
    m_Col(c),
    m_NNZCount(nnzCount),
    m_IsSymmetric(isSymmetric),
    m_IsPatternSymmetric(isPatternSymmetric)
{
    m_OrderingSupportedMatrix = nullptr;

    m_Ptr = new vType[m_Row + 1];
    memset(m_Ptr, 0, sizeof(vType) * (m_Row + 1));

    m_Ind = new vType[m_NNZCount];
    memset(m_Ind, 0, sizeof(vType) * m_NNZCount);

    m_Values = new valType[m_NNZCount];
    memset(m_Values, 0, sizeof(valType) * m_NNZCount);

    this->checkShape();
}

SparseMatrix::~SparseMatrix()
{
    delete[] m_Ptr;
    delete[] m_Ind;
    delete[] m_Values;
}

void SparseMatrix::checkProperties(bool adjSorted)
{
    this->checkShape();
    this->checkSymmetry(adjSorted);
}

void SparseMatrix::deepCopy(const SparseMatrix &other)
{
    m_Name = other.m_Name;
    m_Row = other.m_Row;
    m_Col = other.m_Col;
    m_NNZCount = other.m_NNZCount;
    m_IsSymmetric = other.m_IsSymmetric;
    m_IsPatternSymmetric = other.m_IsPatternSymmetric;
    m_IsSquare = other.m_IsSquare;
    m_OrderingSupportedMatrix = other.m_OrderingSupportedMatrix;

    m_Ptr = new vType[m_Row + 1];
    memcpy(m_Ptr, other.m_Ptr, sizeof(vType) * (m_Row + 1));

    m_Ind = new vType[m_NNZCount];
    memcpy(m_Ind, other.m_Ind, sizeof(vType) * m_NNZCount);

    m_Values = new valType[m_NNZCount];
    memcpy(m_Values, other.m_Values, sizeof(vType) * m_NNZCount);
}

SparseMatrix::SparseMatrix(const SparseMatrix &other)
{
    this->deepCopy(other);
}

SparseMatrix &SparseMatrix::operator=(const SparseMatrix &other)
{
    if (this == &other)
    {
        return *this;
    }

    delete[] m_Ptr;
    delete[] m_Ind;
    delete[] m_Values;

    this->deepCopy(other);

    return *this;
}

SparseMatrix::SparseMatrix(SparseMatrix&& other)
:   m_Name(other.m_Name),
    m_Row(other.m_Row),
    m_Col(other.m_Col),
    m_NNZCount(other.m_NNZCount),
    m_IsSymmetric(other.m_IsSymmetric),
    m_IsPatternSymmetric(other.m_IsPatternSymmetric),
    m_IsSquare(other.m_IsSquare),
    m_Ptr(other.m_Ptr),
    m_Ind(other.m_Ind),
    m_Values(other.m_Values)
{
    m_OrderingSupportedMatrix = other.m_OrderingSupportedMatrix;

    other.m_Ptr = nullptr;
    other.m_Ind = nullptr;
    other.m_Values = nullptr;
    other.m_OrderingSupportedMatrix = nullptr;
}

SparseMatrix& SparseMatrix::operator=(SparseMatrix&& other)
{
    if (this == &other)
    {
        return *this;
    }

    delete[] m_Ptr;
    delete[] m_Ind;
    delete[] m_Values;

    m_Name = other.m_Name;
    m_Row = other.m_Row;
    m_Col = other.m_Col;
    m_NNZCount = other.m_NNZCount;
    m_IsSymmetric = other.m_IsSymmetric;
    m_IsPatternSymmetric = other.m_IsPatternSymmetric;
    m_IsSquare = other.m_IsSquare;

    m_Ptr = other.m_Ptr;
    m_Ind = other.m_Ind;
    m_Values = other.m_Values;
    m_OrderingSupportedMatrix = other.m_OrderingSupportedMatrix;

    other.m_Ptr = nullptr;
    other.m_Ind = nullptr;
    other.m_Values = nullptr;
    other.m_OrderingSupportedMatrix = nullptr;

    return *this;
}

void SparseMatrix::checkSymmetry(bool adj_sorted)
{
    vType m = this->m_Row;
    vType n = this->m_Col;
    vType* ptrs = this->m_Ptr;
    vType nnz = ptrs[m];
    vType* ids = this->m_Ind;
    valType* vals = this->m_Values;

    // Check if the matrix is square
    if (m != n) {
        m_IsSymmetric = false;
        m_IsPatternSymmetric = false;
        return;
    }
    
    //start with the symettric assumption
    m_IsSymmetric = true;
    m_IsPatternSymmetric = true;

    if(adj_sorted) {
        vType* tptrs = new vType[n + 1]; 
        memcpy(tptrs, ptrs, sizeof(vType) * (n + 1));
        for(vType v = 0; v < m; v++) {
            for(vType ptr = ptrs[v]; ptr < ptrs[v+1]; ptr++) {
                vType nbr = ids[ptr];
                vType inbr = ids[tptrs[nbr]];
                if((tptrs[nbr] == ptrs[nbr + 1]) || (inbr != v)) {
                    m_IsSymmetric = false;
                    m_IsPatternSymmetric = false;
                    return;
                }
                if(vals[ptr] != vals[tptrs[nbr]]) {
                   m_IsPatternSymmetric = false;
                }
                tptrs[nbr]++;
            }
        }
    } else {   
        //lists are not sorted so we either need to look for the vertex in the adjacency list of the neighbor (easy to parallelize but expensive) 
        //or do the things below which first takes the transpose of the matrix 

        vType* tptrs = new vType[n + 1]; 
        memset(tptrs, 0, sizeof(vType) * (n + 1));
        vType* tids = new vType[nnz];
        valType* tvals = new valType[nnz];

        for(vType i = 0; i < nnz; i++) tptrs[ids[i] + 1]++;
        for(vType i = 1; i <= n; i++) {
            tptrs[i] += tptrs[i-1];
            if(tptrs[i] != ptrs[i]) {
                m_IsSymmetric = false;
                m_IsPatternSymmetric = false;
                return;
            }
        }

        for(vType i = 0; i < m; i++) {
            for(vType ptr = ptrs[i]; ptr < ptrs[i+1]; ptr++) {
                vType nbr = ids[ptr];
                tids[tptrs[nbr]] = nbr;
                tvals[tptrs[nbr]++] = vals[ptr];
            }
        }
        for(vType i = n-1; i > 0; i--) tptrs[i] = tptrs[i-1]; //transpose ended here
        tptrs[0] = 0;
       
        //apply the previous trick over tptrs - lists are sorted so doable
        vType* t2ptrs = new vType[n + 1]; 
        memcpy(t2ptrs, tptrs, sizeof(vType) * (n + 1));
        for(vType v = 0; v < n; v++) {
            for(vType ptr = tptrs[v]; ptr < tptrs[v+1]; ptr++) {
                vType nbr = tids[ptr];
                vType inbr = tids[t2ptrs[nbr]];

                if((t2ptrs[nbr] == tptrs[nbr + 1]) || (inbr != v)) {
                    m_IsSymmetric = false;
                    m_IsPatternSymmetric = false;
                    return;
                }
                if(tvals[ptr] != tvals[t2ptrs[nbr]]) {
                    m_IsPatternSymmetric = false;
                }
                t2ptrs[nbr]++;
            }
        }
    }   
}

void SparseMatrix::printNonzeros(vType count) const
{
    if (count > m_Row)
    {
        count = m_Row;
    }

    std::cout << std::endl;
    std::cout << "Printing the first " << count << " nonzeros of " << this->getName() << ':' << std::endl;
    for (vType r = 1; r <= count; ++r)
    {
        for (vType nnz = m_Ptr[r-1]; nnz != m_Ptr[r]; ++nnz)
        {
            std::cout << r - 1 << ' ' << m_Ind[nnz] << ' ' << (double) m_Values[nnz] << std::endl;
        }
    }
    std::cout << std::endl;
}

SparseMatrix SparseMatrix::generateTransposedMatrix() const
{
    // Create a transposed matrix with swapped dimensions
    SparseMatrix matrixTransposed("Transposed " + m_Name, m_Col, m_Row, m_NNZCount, m_IsSymmetric, m_IsPatternSymmetric);

    // Count the number of non-zeros in each column of the transposed matrix
    for (vType i = 0; i < m_NNZCount; ++i)
    {
        matrixTransposed.m_Ptr[m_Ind[i] + 1]++;
    }

    // Cumulative sum
    for (vType i = 1; i <= m_Col; ++i)
    {
        matrixTransposed.m_Ptr[i] += matrixTransposed.m_Ptr[i - 1];
    }

    vType* workingPtr = new vType[m_Col + 1];
    memcpy(workingPtr, matrixTransposed.m_Ptr, sizeof(vType) * (m_Col + 1));

    // Fill m_Ind and m_Values for the transposed matrix
    for (vType i = 0; i < m_Row; ++i)
    {
        for (vType j = m_Ptr[i]; j < m_Ptr[i + 1]; ++j)
        {
            vType index = workingPtr[m_Ind[j]]++;
            matrixTransposed.m_Ind[index] = i;
            matrixTransposed.m_Values[index] = m_Values[j];
        }
    }

    delete[] workingPtr;

    return matrixTransposed;
}

void SparseMatrix::transpose()
{
    *this = this->generateTransposedMatrix();
}

SparseMatrix SparseMatrix::operator+(const SparseMatrix &other) const
{
    if (m_Row != other.m_Row || m_Col != other.m_Col)
    {
        throw std::runtime_error("The operands do not have same row and column numbers");
    }

    SparseMatrix newMatrix(m_Name + " + " + other.m_Name, m_Row, m_Col, 0);

    delete[] newMatrix.m_Values;
    delete[] newMatrix.m_Ind;

    std::vector<vType> tempInd(m_NNZCount + other.m_NNZCount);
    std::vector<valType> tempValues(m_NNZCount + other.m_NNZCount);
    tempInd.reserve(m_NNZCount + other.m_NNZCount);
    tempValues.reserve(m_NNZCount + other.m_NNZCount);

    vType row1 = 0;
    vType row2 = 0;
    vType nnz1 = 0;
    vType nnz2 = 0;
    while (nnz1 < m_NNZCount && nnz2 < other.m_NNZCount)
    {
        if (m_Ptr[row1 + 1] > nnz1)
        {
            ++row1;
        }
        if (other.m_Ptr[row2 + 1] > nnz2)
        {
            ++row2;
        }

        if (row1 == row2 && m_Ind[nnz1] == other.m_Ind[nnz2])
        {
            ++newMatrix.m_Ptr[row1 + 1];
            tempInd.push_back(m_Ind[nnz1]);
            tempValues.push_back(m_Values[nnz1] + other.m_Values[nnz2]);
            ++nnz1;
            ++nnz2;
        }
        else
        {
            if ((row2 < row1) || (other.m_Ind[nnz2] > m_Ind[nnz1]))
            {
                ++newMatrix.m_Ptr[row2 + 1];
                tempInd.push_back(other.m_Ind[nnz2]);
                tempValues.push_back(other.m_Values[nnz2]);
                ++nnz2;
            }
            else
            {
                ++newMatrix.m_Ptr[row1 + 1];
                tempInd.push_back(m_Ind[nnz1]);
                tempValues.push_back(m_Values[nnz1]);
                ++nnz1;
            }
        }
    }

    while (nnz1 < m_NNZCount)
    {
        if (m_Ptr[row1 + 1] > nnz1)
        {
            ++row1;
        }

        ++newMatrix.m_Ptr[row1 + 1];
        tempInd.push_back(m_Ind[nnz1]);
        tempValues.push_back(m_Values[nnz1]);
        ++nnz1;
    }

    while (nnz2 < other.m_NNZCount)
    {
        if (other.m_Ptr[row2 + 1] > nnz2)
        {
            ++row2;
        }

        ++newMatrix.m_Ptr[row2 + 1];
        tempInd.push_back(other.m_Ind[nnz2]);
        tempValues.push_back(other.m_Values[nnz2]);
        ++nnz2;
    }

    tempInd.shrink_to_fit();
    tempValues.shrink_to_fit();

    newMatrix.m_NNZCount = tempInd.size();
    newMatrix.m_Ind = new vType[newMatrix.m_NNZCount];
    newMatrix.m_Values = new valType[newMatrix.m_NNZCount];
    std::copy(tempInd.begin(), tempInd.end(), newMatrix.m_Ind);
    std::copy(tempValues.begin(), tempValues.end(), newMatrix.m_Values);

    newMatrix.checkSymmetry(false);

    return newMatrix;
}

void SparseMatrix::symmetritize()
{
    *this = this->generateSymmetritizedMatrix();
}

SparseMatrix SparseMatrix::generateSymmetritizedMatrix() const
{
    return *this + this->generateTransposedMatrix();
}

void SparseMatrix::checkShape()
{
    if (m_Row == m_Col)
    {
        m_IsSquare = true;
    }
    else
    {
        m_IsSquare = false;
    }
}

void SparseMatrix::makeSquareShapedMatrix()
{
    *this = this->generateSquareShapedMatrix();
}

SparseMatrix SparseMatrix::generateSquareShapedMatrix() const
{
    double start_time = omp_get_wtime();

    vType dimensionCount = std::max(m_Row, m_Col);
    SparseMatrix squareShapedMatrix("Square Shaped " + m_Name, dimensionCount, dimensionCount, 0);
    delete[] squareShapedMatrix.m_Ind;
    delete[] squareShapedMatrix.m_Values;

    SparseMatrix transposedMatrix(this->generateTransposedMatrix());

    std::vector<vType> tempPtr(dimensionCount + 1, 0);
    std::vector<vType> tempInd;
    std::vector<valType> tempValues;
    tempInd.reserve(m_NNZCount + transposedMatrix.m_NNZCount);
    tempValues.reserve(m_NNZCount + transposedMatrix.m_NNZCount);

    vType row1 = 0;
    vType row2 = 0;
    vType nnz1 = 0;
    vType nnz2 = 0;
    while (nnz1 < m_NNZCount && nnz2 < transposedMatrix.m_NNZCount)
    {
        if (m_Ptr[row1 + 1] > nnz1)
        {
            ++row1;
        }
        if (transposedMatrix.m_Ptr[row2 + 1] > nnz2)
        {
            ++row2;
        }

        if (row1 == row2 && m_Ind[nnz1] == transposedMatrix.m_Ind[nnz2])
        {
            ++tempPtr[row1 + 1];
            tempInd.push_back(m_Ind[nnz1]);
            tempValues.push_back(m_Values[nnz1] + transposedMatrix.m_Values[nnz2]);
            ++nnz1;
            ++nnz2;
        }
        else
        {
            if ((row2 < row1) || (transposedMatrix.m_Ind[nnz2] > m_Ind[nnz1]))
            {
                ++tempPtr[row2 + 1];
                tempInd.push_back(transposedMatrix.m_Ind[nnz2]);
                tempValues.push_back(transposedMatrix.m_Values[nnz2]);
                ++nnz2;
            }
            else
            {
                ++tempPtr[row1 + 1];
                tempInd.push_back(m_Ind[nnz1]);
                tempValues.push_back(m_Values[nnz1]);
                ++nnz1;
            }
        }
    }

    while (nnz1 < m_NNZCount)
    {
        if (m_Ptr[row1 + 1] > nnz1)
        {
            ++row1;
        }

        ++tempPtr[row1 + 1];
        tempInd.push_back(m_Ind[nnz1]);
        tempValues.push_back(m_Values[nnz1]);
        ++nnz1;
    }

    while (nnz2 < transposedMatrix.m_NNZCount)
    {
        if (transposedMatrix.m_Ptr[row2 + 1] > nnz2)
        {
            ++row2;
        }

        ++tempPtr[row2 + 1];
        tempInd.push_back(transposedMatrix.m_Ind[nnz2]);
        tempValues.push_back(transposedMatrix.m_Values[nnz2]);
        ++nnz2;
    }

    tempInd.shrink_to_fit();
    tempValues.shrink_to_fit();

    std::copy(tempPtr.begin(), tempPtr.end(), squareShapedMatrix.m_Ptr);
    squareShapedMatrix.m_NNZCount = tempInd.size();
    squareShapedMatrix.m_Ind = new vType[squareShapedMatrix.m_NNZCount];
    squareShapedMatrix.m_Values = new valType[squareShapedMatrix.m_NNZCount];
    std::copy(tempInd.begin(), tempInd.end(), squareShapedMatrix.m_Ind);
    std::copy(tempValues.begin(), tempValues.end(), squareShapedMatrix.m_Values);

    squareShapedMatrix.checkSymmetry(false);

    double end_time = omp_get_wtime();
    logger->makeSilentLog("Generating square of a matrix " + m_Name, end_time - start_time);

    return squareShapedMatrix;
}

void SparseMatrix::order(const vType *orderedRow, const vType *orderedCol, const std::string& orderingName)
{
    *this = this->generateOrderedMatrix(orderedRow, orderedCol, orderingName);
}

//the adjacency lists need to be sorted
SparseMatrix SparseMatrix::generateOrderedMatrix(const vType *orderedRow, const vType *orderedCol, const std::string& orderingName) const
{
    //generate the memory for the ordered matrix
    vType* new_Ptr = new vType[m_Row + 1];
    vType* new_Ind = new vType[m_NNZCount]; 
    valType* new_Values = new valType[m_NNZCount];

    vType* new_Col_Ptr = new vType[m_Col + 1];
    vType* new_Col_Ind = new vType[m_NNZCount];
    valType* new_Col_Values = new valType[m_NNZCount];

    memset(new_Col_Ptr, 0, sizeof(vType) * (m_Col + 1));
    for(vType i = 0; i < m_NNZCount; i++) new_Col_Ptr[m_Ind[i] + 1]++;
    for(vType i = 2; i <= m_Col; i++) new_Col_Ptr[i] += new_Col_Ptr[i-1];

    for (vType i = 0; i < m_Row; ++i) {
        vType newRow = orderedRow[i];
        for (vType j = m_Ptr[i]; j < m_Ptr[i + 1]; ++j) {
            vType oldcolId = m_Ind[j];
            new_Col_Ind[new_Col_Ptr[oldcolId]] = newRow;
            new_Col_Values[new_Col_Ptr[oldcolId]++] = m_Values[j];
        }
    }
    for(vType i = m_Col; i >= 1; i--) new_Col_Ptr[i] = new_Col_Ptr[i-1];
    new_Col_Ptr[0] = 0;

    new_Ptr[0] = 0;
    for (vType i = 0; i < m_Row; ++i) new_Ptr[orderedRow[i] + 1] = m_Ptr[i + 1] - m_Ptr[i];
    for (vType i = 1; i <= m_Row; ++i) new_Ptr[i] += new_Ptr[i - 1];

    vType* icolperm = new vType[m_Col];
    for(vType i = 0; i < m_Col; i++) icolperm[orderedCol[i]] = i; 
    for (vType i = 0; i < m_Col; ++i) {
        vType newCol = icolperm[i];
        for (vType j = new_Col_Ptr[i]; j < new_Col_Ptr[i + 1]; ++j) {
            vType newrowId = new_Col_Ind[j];
            new_Ind[new_Ptr[newrowId]] = newCol;
            new_Values[new_Ptr[newrowId]++] = new_Col_Values[j];
        }
    }

    SparseMatrix orderedMatrix(orderingName + "_ordered_" + this->getName() , m_Row, m_Col, m_NNZCount, new_Ptr, new_Ind, new_Values);
    //std::cout << this->m_Row << " " << this->m_Col << " " << this->m_NNZCount << " " << this->m_Ptr << " " << this->m_Ind << " " << this->m_Values << std::endl;

    if(m_Row != m_Col) { 
        orderedMatrix.isPatternSymmetric() = false;
        orderedMatrix.isSymmetric() = false;
    } else  {
        bool check = true;
        for (vType i = 0; i < m_Row; ++i) {
            if(orderedRow[i] != orderedCol[i]) {
                check = false;
                break;
            }
        }

        if(check) {
            orderedMatrix.isPatternSymmetric() = isPatternSymmetric();
            orderedMatrix.isSymmetric() = isSymmetric();
        } else {
            orderedMatrix.checkSymmetry(false);
        }
    }

    delete [] new_Col_Ptr;
    delete [] new_Col_Ind;
    delete [] new_Col_Values;
    delete [] icolperm;

    return orderedMatrix;
}

void SparseMatrix::generateOrderingSupportedMatrix()
{
    if (m_OrderingSupportedMatrix)
    {
        return;
    }

    std::vector<entry> entries;
    const SparseMatrix& org = *this;
    entries.reserve(org.getNNZCount() * 2);

    for(int row = 0; row < org.getRowCount(); row++) {
        for(int ptr = org.getPtr()[row]; ptr < org.getPtr()[row+1]; ptr++) {
            int nbr = org.getInd()[ptr];
            if(row != nbr) { //just to be sure - zero diagonal
                entry e(row, nbr);
                entries.push_back(e);

                entry e2(nbr, row);
                entries.push_back(e2);
            }
        }
    }

    entry* sorted_entries = (entry*)malloc(sizeof(entry) * entries.size());
    vType maxmn = std::max((int)org.getRowCount(), (int)org.getColCount());

    vType *os_ptrs = new vType[maxmn + 1];
    vType *os_ids = new vType[entries.size()];

    m_OrderingSupportedMatrix = std::make_shared<SparseMatrix>(SparseMatrix("Ordering Supported " + m_Name, maxmn, maxmn, 0, os_ptrs, os_ids, nullptr));
    m_OrderingSupportedMatrix->isSymmetric() = true;
    m_OrderingSupportedMatrix->isPatternSymmetric() = true;
    m_OrderingSupportedMatrix->isSquare() = true;

    vType *last_ids = new vType[maxmn];
    for(vType j = 0; j < maxmn; j++) last_ids[j] = -1;

    //sort entries wrt. columns
    memset(os_ptrs, 0, sizeof(vType) * (maxmn + 1));
    for(auto e : entries) os_ptrs[e.col + 1]++;
    for(vType j = 1; j < maxmn; j++) os_ptrs[j] += os_ptrs[j - 1];
    for(auto e : entries) sorted_entries[os_ptrs[e.col]++] = e;

    //sort entries with respect to rows - phase 1: counting / mark if duplicate
    memset(os_ptrs, 0, sizeof(int) * (maxmn + 1));
    for(vType j = 0; j < entries.size(); j++) {
        auto &e = sorted_entries[j];
        if(last_ids[e.row] != e.col) {
            os_ptrs[e.row + 1]++;
            last_ids[e.row] = e.col;
        } else {
            e.row = -1;
        }
    }
    for(vType j = 1; j <= maxmn; j++) os_ptrs[j] += os_ptrs[j - 1];

    //phase 2: distribute the entries and sort
    for(vType j = 0; j < entries.size(); j++) {
        auto e = sorted_entries[j];
        if(e.row != -1) {
            os_ids[os_ptrs[e.row]++] = e.col;
        }
    }
    for(vType j = maxmn; j > 0; j--) os_ptrs[j] = os_ptrs[j-1];
    os_ptrs[0] = 0;
    m_OrderingSupportedMatrix->getNNZCount() = os_ptrs[maxmn];

    delete [] last_ids;
}
