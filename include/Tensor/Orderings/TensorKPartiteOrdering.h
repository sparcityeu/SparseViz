//
// Created by kamer on 1/15/24.
//

#ifndef SPARSEVIZ_TENSORKPARTITEORDERING_H
#define SPARSEVIZ_TENSORKPARTITEORDERING_H

#include "TensorOrdering.h"
#include <string>
//#include <unordered_map>

class TensorKPartiteOrdering: public TensorOrdering
{
public:
    TensorKPartiteOrdering(SparseTensor& tensor, const std::vector<vType>& active_modes, std::string orderingName, std::string orderingType) 
        : TensorOrdering(tensor, active_modes, orderingName), m_OrderingType(orderingType) {}

    static std::unordered_map<std::string, SparseMatrix*> kpartite_matrices;
    static std::unordered_map<std::string, omp_lock_t*> kpartite_locks;
    static omp_lock_t kpartite_lock;

private:
    virtual void orderingFunction() override;
    std::string m_OrderingType;
    SparseMatrix* getKPartite();
    SparseMatrix* generateKPartiteMatrix(std::string);
};
#endif //SPARSEVIZ_TENSORKPARTITEORDERING_H
