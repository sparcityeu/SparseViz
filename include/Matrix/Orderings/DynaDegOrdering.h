//
// Created by Kamer Kaya on 31.12.2023.
//

#ifndef SPARSEVIZ_MINDORDERING_H
#define SPARSEVIZ_MINDORDERING_H

#include "MatrixOrdering.h"
#include <string>

class DynaDegOrdering : public MatrixOrdering
{
public:
    DynaDegOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters): MatrixOrdering(matrix, orderingName, true, true), m_Direction("MAX") {
        if(orderingParameters != "") {
            std::vector<std::string> parameters = split(orderingParameters, '/');

            if (parameters.size() > 0) {
                m_Direction = parameters[0];
            }
        }
    }

private:
    virtual void orderingFunction() override;
    struct Node {
        int cdeg; 
        int cloc;
    };
    std::string m_Direction;
};

#endif //SPARSEVIZ_MINDORDERING_H
