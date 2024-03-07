//
// Created by Kamer Kaya on 22.12.2023.
//

#ifndef SPARSEVIZ_RCMORDERING_H
#define SPARSEVIZ_RCMORDERING_H

#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include <vector>
#include <algorithm>
#include "config.h"
#include "helpers.h"

class RCMOrdering: public MatrixOrdering
{
public:
    RCMOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters) :
    MatrixOrdering(matrix, orderingName, false, false), m_SortingCriteria("DEG"), m_RootSelection(true) {

        if(orderingParameters != "") {
            std::vector<std::string> parameters = split(orderingParameters, '/');

            if (parameters.size() > 0) {
                m_SortingCriteria = parameters[0];
            } 
            if (parameters.size() > 1) {
                if(parameters[1] == "FALSE") {
                    m_RootSelection = false;
                }
            }
        }
    }

private:
    std::string m_SortingCriteria; 
    bool m_RootSelection; 

    virtual void orderingFunction() override;
    bool findRoot (    unsigned& root, unsigned* queue,
                       int* level, int& max_level, unsigned& ccsize);  
    void rcm (  unsigned& root, unsigned* queue,    
                double* sort_values,
                bool* is_permuted);
};

#endif //SPARSEVIZ_RCMORDERING_H
