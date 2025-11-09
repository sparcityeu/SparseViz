#pragma once

#include "MatrixOrdering.h"

class FileOrdering: public MatrixOrdering
{
public:
    FileOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters);

private:
    virtual void orderingFunction() override;

private:
    std::string m_FileName;
};
