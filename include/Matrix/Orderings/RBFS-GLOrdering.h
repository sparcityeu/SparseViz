//
// Created by orderinggroup on 11/24/25.
//

#ifndef SPARSEVIZ_RBFSGLORDERING_H
#define SPARSEVIZ_RBFSGLORDERING_H

#pragma once

#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include "config.h"
#include "helpers.h"

#include <vector>
#include <string>

class RBFSGLOrdering : public MatrixOrdering {
public:
    RBFSGLOrdering(SparseMatrix& m,
                   std::string orderingName,
                   std::string /*unused*/)
        : MatrixOrdering(m,
                         std::move(orderingName),
                         /*rectangularSupport=*/false,
                         /*patternUnsymmetricSupport=*/false,
                         /*price=*/3.0) {}

private:
    void orderingFunction() override;

    // BFS with Gibbs–Lewis priority rule (level, degree)
    void bfsGL(unsigned root,
               const std::vector<unsigned>& degree,
               std::vector<char>& assigned,
               std::vector<unsigned>& visitOrder);
};


#endif //SPARSEVIZ_RBFSGLORDERING_H