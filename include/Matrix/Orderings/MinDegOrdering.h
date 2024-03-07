//
// Created by Kamer Kaya on 25.12.2023.
//

#ifndef SPARSEVIZ_MINDEGORDERING_H
#define SPARSEVIZ_MINDEGORDERING_H

#include "MatrixOrdering.h"
#include <string>

class MinDegOrdering : public MatrixOrdering
{
public:
    MinDegOrdering(SparseMatrix& matrix, std::string orderingName): MatrixOrdering(matrix, orderingName, false, false) {}

private:
    virtual void orderingFunction() override;
    struct Node {
        int cdeg; 
        int cloc;
        std::vector<int> cliqueIDs;
    };
};


#endif //SPARSEVIZ_MINDEGORDERING_H
