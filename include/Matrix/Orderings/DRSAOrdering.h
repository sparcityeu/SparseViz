#pragma once

#include "MatrixOrdering.h"
#include <vector>
#include <random>

class DRSAOrdering : public MatrixOrdering
{
public:
    DRSAOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters);
    ~DRSAOrdering() override = default;

protected:
    void orderingFunction() override;

private:
    // Simulated Annealing Parameters tuned via the paper
    double m_T0;
    double m_Alpha;
    double m_Tf;
    double m_L0;
    double m_Lf;
    
    std::mt19937 m_Rng;
};
