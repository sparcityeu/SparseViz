#include "METISOrdering.h"
#include <vector>
#include <metis.h>
#include <iostream>
#include <algorithm>
#include <map>


bool shouldBeSeparator(idx_t node, const std::vector<idx_t>& xadj, 
                       const std::vector<idx_t>& adjncy, 
                       const std::vector<idx_t>& part, idx_t myPartID) {
    for (idx_t j = xadj[node]; j < xadj[node+1]; ++j) {
        idx_t nbr = adjncy[j];
        if (part[nbr] > myPartID) return true; 
    }
    return false;
}

void runNestedDissection(idx_t n, const std::vector<idx_t>& xadj, 
                         const std::vector<idx_t>& adjncy, 
                         const std::vector<idx_t>& original_ids,
                         std::vector<idx_t>& final_perm, 
                         idx_t& current_rank,
                         idx_t branching_factor) { 
        
    if (n < 20) {
        for (idx_t i = 0; i < n; ++i) {
            final_perm[original_ids[i]] = current_rank++;
        }
        return;
    }

    
    idx_t nparts = branching_factor;
    idx_t ncon = 1;
    std::vector<idx_t> part(n);
    idx_t objval;
    
    std::vector<idx_t> local_xadj = xadj;
    std::vector<idx_t> local_adjncy = adjncy;

    
    int status;
    if (nparts > 2) {
         status = METIS_PartGraphKway(&n, &ncon, local_xadj.data(), local_adjncy.data(), 
                                      nullptr, nullptr, nullptr, &nparts, 
                                      nullptr, nullptr, nullptr, &objval, part.data());
    } else {
         status = METIS_PartGraphRecursive(&n, &ncon, local_xadj.data(), local_adjncy.data(), 
                                           nullptr, nullptr, nullptr, &nparts, 
                                           nullptr, nullptr, nullptr, &objval, part.data());
    }

    
    if (status != METIS_OK) {
        
        for (idx_t i = 0; i < n; ++i) final_perm[original_ids[i]] = current_rank++;
        return;
    }

    
    std::vector<std::vector<idx_t>> domains(nparts);
    std::vector<idx_t> separator_nodes;

    for (idx_t i = 0; i < n; ++i) {
        idx_t p = part[i];
        
        
        if (shouldBeSeparator(i, xadj, adjncy, part, p)) {
            separator_nodes.push_back(i);
        } else {
            if (p >= 0 && p < nparts) {
                domains[p].push_back(i);
            } else {                
                separator_nodes.push_back(i);
            }
        }
    }

    for (idx_t k = 0; k < nparts; ++k) {
        const std::vector<idx_t>& nodes = domains[k];
        if (nodes.empty()) continue;

        idx_t m = nodes.size();
        std::map<idx_t, idx_t> local_map;
        for(idx_t i=0; i<m; ++i) local_map[nodes[i]] = i;

        std::vector<idx_t> sub_xadj(m + 1);
        std::vector<idx_t> sub_adjncy;
        std::vector<idx_t> sub_orig_ids(m);
        idx_t sub_idx = 0;

        for(idx_t i=0; i<m; ++i) {
            idx_t u = nodes[i];
            sub_orig_ids[i] = original_ids[u]; 
            sub_xadj[i] = sub_idx;
            
            for(idx_t j = xadj[u]; j < xadj[u+1]; ++j) {
                idx_t v = adjncy[j];
                
                if (local_map.count(v)) {
                    sub_adjncy.push_back(local_map[v]);
                    sub_idx++;
                }
            }
        }
        sub_xadj[m] = sub_idx;

        
        runNestedDissection(m, sub_xadj, sub_adjncy, sub_orig_ids, final_perm, current_rank, branching_factor);
    }

    
    for (idx_t node_idx : separator_nodes) {
        final_perm[original_ids[node_idx]] = current_rank++;
    }
}

void METISOrdering::orderingFunction()
{
    SparseMatrix* mat = const_cast<SparseMatrix*>(&getMatrix()); 
    idx_t n = static_cast<idx_t>(mat->getRowCount());

    std::vector<idx_t> xadj(n + 1);
    std::vector<idx_t> adjncy;
    idx_t nnzCount = 0;
    
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = mat->getPtr()[i]; j < mat->getPtr()[i + 1]; ++j) {
            if (mat->getInd()[j] != i) nnzCount++;
        }
    }
    adjncy.resize(nnzCount);
    
    idx_t currentIdx = 0;
    for (idx_t i = 0; i < n; ++i) {
        xadj[i] = currentIdx;
        for (idx_t j = mat->getPtr()[i]; j < mat->getPtr()[i + 1]; ++j) {
            idx_t nbr = static_cast<idx_t>(mat->getInd()[j]);
            if (nbr != i) adjncy[currentIdx++] = nbr;
        }
    }
    xadj[n] = currentIdx;

    std::vector<idx_t> original_ids(n);
    for(idx_t i=0; i<n; ++i) original_ids[i] = i;

    std::vector<idx_t> final_perm(n);
    idx_t current_rank = 0;

    idx_t branching_factor = 2; 

    runNestedDissection(n, xadj, adjncy, original_ids, final_perm, current_rank, branching_factor);

    this->rowIPermutation = new vType[n];
    this->colIPermutation = new vType[n];

    for (idx_t i = 0; i < n; ++i)
    {
        this->rowIPermutation[i] = static_cast<vType>(final_perm[i]);
        this->colIPermutation[i] = static_cast<vType>(final_perm[i]);
    }
}