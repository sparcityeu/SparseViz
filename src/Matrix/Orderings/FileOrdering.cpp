#include "FileOrdering.h"
#include <fstream>

FileOrdering::FileOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters)
: MatrixOrdering(matrix, orderingName, true, true)
{
    m_FileName = MATRIX_ORDERING_FILES_DIR + this->getMatrix().getName() + '_' + orderingParameters + ".ord";
}

void FileOrdering::orderingFunction()
{
    rowIPermutation = new vType[this->getMatrix().getRowCount()];
    colIPermutation = new vType[this->getMatrix().getColCount()];

    std::ifstream file(m_FileName);
    for (unsigned i = 0; i < this->getMatrix().getRowCount(); ++i)
    {
        file >> rowIPermutation[i];
    }
    for (unsigned i = 0; i < this->getMatrix().getColCount(); ++i)
    {
        file >> colIPermutation[i];
    }
    file.close();
}
