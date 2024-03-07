//
// Created by Kamer Kaya on 22.12.2023.
//

#ifndef SPARSEVIZ_PATOHORDERING_H
#define SPARSEVIZ_PATOHORDERING_H

#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include <vector>
#include <algorithm>
#include "config.h"
#include "helpers.h"
#include <string>

class PatohOrdering: public MatrixOrdering
{
public:
    PatohOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters) :
    MatrixOrdering(matrix, orderingName, true, true), m_NoParts(8), m_Model("CN"), m_CellWght("UNIT"), 
                                                      m_Metric("CUT"), m_NetPlacement("END"), m_Inner("NONE") {
       if(orderingParameters != "") {
            std::vector<std::string> parameters = split(orderingParameters, '/');

            if (parameters.size() > 0) {
                m_NoParts = stoi(parameters[0]);
            } 
            
            if (parameters.size() > 1) {
                m_Model = parameters[1];    
            }

            if (parameters.size() > 2) {
                m_CellWght = parameters[2];    
            }

            if (parameters.size() > 3) {
                m_Metric = parameters[3];
            }

            if (parameters.size() > 4) {
                m_NetPlacement = parameters[4];
            }

            if (parameters.size() > 5) {
                m_Inner = parameters[5];
            }
        }
    }
    static omp_lock_t patoh_lock;

private:
    int m_NoParts;
    std::string m_Model; 
    std::string m_CellWght;
    std::string m_Metric;
    std::string m_NetPlacement;
    std::string m_Inner; 

    virtual void orderingFunction() override;
    void wrapPaToH(	int _c, int _n, int *xpins, int *pins, int *cwghts, 
				    int *nwghts, int *partvec, int no_cons, int patoh_no_parts);
    void colNetOrder(int* ptrs, int* js, int m, int n, int* partv );
    void rowNetOrder(int* ptrs, int* js, int m, int n, int* partv);
    void twoDimOrder(int* ptrs, int* I, int* J, int m, int n, int nz, int* partv);
    void innerOrder(int* ptrs, int* js, int m, int n, int* rpartv, int* cpartv);
};
#endif //SPARSEVIZ_PATOHORDERING_H
