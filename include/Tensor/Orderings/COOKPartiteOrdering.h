//
// Created by kamer on 1/15/24.
//

#ifndef SPARSEVIZ_COOKPARTITEORDERING_H
#define SPARSEVIZ_COOKPARTITEORDERING_H

#include "TensorOrdering.h"
#include <string>


class COOKPartiteOrdering: public TensorOrdering
{
public:
    COOKPartiteOrdering(SparseTensor& tensor, const std::vector<vType>& active_modes, std::string orderingName, std::string orderingType)
        : TensorOrdering(tensor, active_modes, orderingName), m_OrderingType(orderingType) {}

public:
    static std::unordered_map<std::string, SparseMatrix*> kpartite_matrices;
    static std::unordered_map<std::string, omp_lock_t*> kpartite_locks;
    static omp_lock_t kpartite_lock;

private:
    virtual void orderingFunction() override;
    SparseMatrix* getKPartite();
    SparseMatrix* generateKPartiteMatrix(std::string);

private:
    std::string m_OrderingType;
};


#endif //SPARSEVIZ_COOKPARTITEORDERING_H
