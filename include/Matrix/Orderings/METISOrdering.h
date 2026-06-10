#pragma once
#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include <string>
#include <vector>
#include <metis.h>

class METISOrdering : public MatrixOrdering {
public:
    METISOrdering(SparseMatrix& m, std::string name, std::string params)
        : MatrixOrdering(m, std::move(name),
                         /*rectangularSupport=*/false,
                         /*patternUnsymmetricSupport=*/false,
                         /*price=*/5) 
    {
        
    }

private:
    void orderingFunction() override;
};