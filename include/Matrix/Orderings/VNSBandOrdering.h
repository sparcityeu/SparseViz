//
// Created by grkmshn on 11/20/25.
//

#ifndef SPARSEVIZ_VNSBANDORDERING_H
#define SPARSEVIZ_VNSBANDORDERING_H

#pragma once
#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include <string>


class VNSBandOrdering : public MatrixOrdering {
public:
    VNSBandOrdering(SparseMatrix& m,
                    std::string name = "VNSBand",
                    std::string /*unused*/ = "")
    : MatrixOrdering(m, name,
                     /*rectangularSupport=*/false,
                     /*patternUnsymmetricSupport=*/false, //I am not exactly sure of this, I should ask hoca at some point
                     /*price=*/9.0) {}

private:
    void orderingFunction() override;
};



#endif //SPARSEVIZ_VNSBANDORDERING_H