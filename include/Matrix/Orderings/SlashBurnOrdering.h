


#ifndef SPARSEVIZ_SBURNORDERING_H
#define SPARSEVIZ_SBURNORDERING_H

#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include <vector>
#include <algorithm>
#include "config.h"
#include "helpers.h"
#include <string>

class SlashBurnOrdering: public MatrixOrdering
{
public:
    SlashBurnOrdering(SparseMatrix& matrix, std::string orderingName,  std::string orderingParameters) :
    MatrixOrdering(matrix, orderingName, false, false), KSIZE(0.04 * matrix.getRowCount()), GREEDY(false), HUB_ORDER(true) {
        if(orderingParameters != "") {
            std::vector<std::string> parameters = split(orderingParameters, '/');
            if(parameters.size() > 0) {
                KSIZE = stoi(parameters[0]);
            }
            if(parameters.size() > 1) {
                if (parameters[1] == "TRUE") {
                    GREEDY = true;
                }
            } 
            if(parameters.size() > 2) {
                if (parameters[2] == "TRUE") {
                    HUB_ORDER = true;
                }
            } 
        }
    }

private:
    int KSIZE;
    bool GREEDY;
    bool HUB_ORDER; 
    virtual void orderingFunction() override;
    unsigned* computeDegree(unsigned* rptr, unsigned* col, unsigned n, unsigned *v_flag, unsigned level);
    unsigned* removeKHubsetGreedy(  unsigned *rptr, unsigned *col, unsigned n, unsigned k, 
                                    unsigned *v_flag, unsigned *order, unsigned *degree, unsigned level, unsigned min_id);
    unsigned* removeKHubset(        unsigned *rptr, unsigned *col, unsigned n, unsigned k, 
                                    unsigned *v_flag, unsigned *order, unsigned level, unsigned min_id);
    unsigned findCC(                unsigned* rptr,  unsigned* col, 
                                    unsigned *v_flag, unsigned level, unsigned root);
    unsigned orderCC(               unsigned* rptr,  unsigned* col, 
                                    unsigned *v_flag, unsigned *order, unsigned level, unsigned root, unsigned max_id);
    void slashloop(                 unsigned *rptr, unsigned *col, unsigned n, unsigned k, 
                                    unsigned *v_flag, unsigned *order, unsigned level, unsigned max_id);
};

#endif //SPARSEVIZ_SBURNORDERING_H
