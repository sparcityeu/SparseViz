//                                                                                                                                                                                                          
// Created on 22 October 2023                                                                                                                                                                
//                                                                                                                                                                                                          

#include "RabbitOrdering.h"
#include "Parameters.h"
#include <vector>
#include <algorithm>
#include <rabbit_order.hpp>

typedef std::vector<std::vector<std::pair<unsigned int, float> > > adjacency_list;

void RabbitOrdering::orderingFunction() {
    int n = this->getMatrix().getRowCount();
    const int* ptrs = (const int*) this->getMatrix().getPtr();
    const int* ids = (const int*) this->getMatrix().getInd();

    rowIPermutation = new unsigned[n];
    colIPermutation = new unsigned[n];
    for(int i = 0; i < n; i++) {
        rowIPermutation[i] = colIPermutation[i] = i;
    }

    adjacency_list graph(n);
    for (int i = 0; i < n; i++) {
        for (int j = ptrs[i]; j < ptrs[i + 1]; j++) {
            int source = i;
            int target = ids[j];
            graph[source].push_back(std::make_pair((unsigned)target, 1.0f));
        }
    }

    const auto g = rabbit_order::aggregate(graph);
    const auto p = rabbit_order::compute_perm(std::move(g));

    for(int i = 0; i < n; i++) {
        rowIPermutation[i] = colIPermutation[i] = p.get()[i];
    }
    
}