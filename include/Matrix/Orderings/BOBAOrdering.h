//
// Created by Kamer Kaya on 22.12.2023.
//

#ifndef SPARSEVIZ_BOBAORDERING_H
#define SPARSEVIZ_BOBAORDERING_H

#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include <vector>
#include <algorithm>
#include "config.h"
#include "helpers.h"

class BOBAOrdering: public MatrixOrdering
{
public:
    BOBAOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters) :
    MatrixOrdering(matrix, orderingName, false, false), m_Execution("SEQ") {
    if(orderingParameters != "") {
        std::vector<std::string> parameters = split(orderingParameters, '/');
        if(parameters.size() > 0) {
            m_Execution = parameters[0];
        }
    }
}

private:
    std::string m_Execution; 

    virtual void orderingFunction() override;
    void applyOrdering(unsigned* order);
};

#endif //SPARSEVIZ_BOBAORDERING_H
