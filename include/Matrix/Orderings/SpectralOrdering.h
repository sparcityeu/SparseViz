#pragma once
#include "MatrixOrdering.h"
#include "SparseMatrix.h"
// No Eigen/Spectra includes needed here for the manual version!

class SpectralOrdering : public MatrixOrdering {
public:
    SpectralOrdering(SparseMatrix& m, std::string name, std::string params);
private:
    void orderingFunction() override;
};