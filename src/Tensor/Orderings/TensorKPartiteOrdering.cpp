//
// Created on 9/15/23.
//

#include "TensorKPartiteOrdering.h"
#include "MatrixOrdering.h"
#include "RCMOrdering.h"
#include "BOBAOrdering.h"
#include "DynaDegOrdering.h"
#include "RabbitOrdering.h"
#include "SlashBurnOrdering.h"
#include "PatohOrdering.h"
#include "GrayOrdering.h"
#include "AMDOrdering.h"
#include <unordered_set>

#include <cstring>
#include <iostream>
#include <fstream>

omp_lock_t TensorKPartiteOrdering::kpartite_lock;
std::unordered_map<std::string, SparseMatrix*> TensorKPartiteOrdering::kpartite_matrices;
std::unordered_map<std::string, omp_lock_t*> TensorKPartiteOrdering::kpartite_locks;

void TensorKPartiteOrdering::orderingFunction()
{
    if (orderedDimensions) {
        return;
    }
     
    int tensor_order = tensor.getOrder();
    const vType* dims = tensor.getDims();

    orderedDimensions = new vType*[tensor_order];
    for (int i = 0; i != tensor_order; ++i) {
        orderedDimensions[i] = new vType[dims[i]];
    } 
    
    SparseMatrix* kpartite = getKPartite();
    std::string fname = tensor.getName();
    for(const vType& m : active_modes) {
        fname += "_" + std::to_string(m);
    }

    /*std::ofstream myfile;
    myfile.open (fname + "_KPartite.mtx");
    myfile << "%%MatrixMarket matrix coordinate real general\n";
    myfile << kpartite->getRowCount() << " " << kpartite->getColCount() << " " << kpartite->getNNZCount() << "\n";
    for(vType v = 0; v < kpartite->getRowCount(); v++) {
        for(vType ptr = kpartite->getPtr()[v]; ptr < kpartite->getPtr()[v+1]; ptr++) {
             myfile << v + 1 << " " << kpartite->getInd()[ptr] + 1 << "\n";
        }
    }
    myfile.close();*/
    MatrixOrdering* matrix_ordering;
    if(m_OrderingType == "RCM") {
        matrix_ordering = new RCMOrdering(*kpartite, "KPartite_RCM", "");
    } else if(m_OrderingType == "BOBA") {
        matrix_ordering = new BOBAOrdering(*kpartite, "KPartite_BOBA", "");
    } else if(m_OrderingType == "AMD") {
        matrix_ordering = new AMDOrdering(*kpartite, "KPartite_AMD");
    } else if(m_OrderingType == "DYNADEG") {
        matrix_ordering = new DynaDegOrdering(*kpartite, "KPartite_DYNADEG", "");
    } else if(m_OrderingType == "RABBIT") {
        matrix_ordering = new RabbitOrdering(*kpartite, "KPartite_RABBIT");
    } else if(m_OrderingType == "SBURN") {
        matrix_ordering = new SlashBurnOrdering(*kpartite, "KPartite_SBURN", "");
    } else if(m_OrderingType == "GRAY") {
        matrix_ordering = new GrayOrdering(*kpartite, "KPartite_GRAY", "");
    } else if(m_OrderingType == "PATOH") {
        matrix_ordering = new PatohOrdering(*kpartite, "KPartite_PATOH", "8/CN/UNIT/CUT/END/RCM");
    } else {
       throw std::runtime_error("Unknown ordering type " + m_OrderingType + " for KPartite!");
    }

    matrix_ordering->generateOrdering();

    vType* orderedVs = new vType[tensor_order];
    memset(orderedVs, 0, sizeof(vType) * tensor_order);

    const vType* rowIPermutation = matrix_ordering->getRowIPermutation();

    vType* rowPermutation = new vType[kpartite->getRowCount()];
    for(vType i = 0; i < kpartite->getRowCount(); i++) {
        rowPermutation[rowIPermutation[i]] = i;
    }

    vType* offsets = new vType[active_modes.size() + 1];
    offsets[0] = 0;
    for(int i = 0; i < active_modes.size(); i++) offsets[i + 1] = dims[active_modes[i]];
    for(int i = 2; i <= active_modes.size(); i++) offsets[i] += offsets[i - 1];

    vType* v_counts = new vType[active_modes.size()];
    memset(v_counts, 0, sizeof(vType) * active_modes.size());
    for(vType i = 0; i < kpartite->getRowCount(); i++) {
        vType v = rowPermutation[i];
        for(int j = 0; j < active_modes.size(); j++) {
            if(v >= offsets[j] && v < offsets[j+1]) {
                orderedDimensions[active_modes[j]][v - offsets[j]] = v_counts[j]++;
            }
        }
    }
    delete [] offsets;
    delete [] v_counts;
    delete matrix_ordering;
    delete [] rowPermutation;
}

SparseMatrix* TensorKPartiteOrdering::getKPartite()
{
    std::string key = tensor.getName();
    for(const vType& m : active_modes) {
        key += "_" + std::to_string(m);
    }

    omp_set_lock(&kpartite_lock);
    std::unordered_map<std::string, omp_lock_t*>::const_iterator got_lock = kpartite_locks.find (key);
    if (got_lock == kpartite_locks.end()) {
        kpartite_locks[key] = new omp_lock_t;
        omp_init_lock(kpartite_locks[key]);
    }
    omp_unset_lock(&kpartite_lock);

    omp_set_lock(kpartite_locks[key]);
    std::unordered_map<std::string, SparseMatrix*>::const_iterator got_matrix = kpartite_matrices.find (key);
    if(got_matrix == kpartite_matrices.end()) {
        kpartite_matrices[key] = generateKPartiteMatrix(key);
    }
    omp_unset_lock(kpartite_locks[key]);

    return kpartite_matrices[key];
}

SparseMatrix* TensorKPartiteOrdering::generateKPartiteMatrix(std::string name) {
    SparseMatrix* kpartite_matrix = nullptr;
    
    std::unordered_set<vpair, pair_hash> nnz_set;
    vType matrix_dim = 0;
    vType* offsets = new vType[active_modes.size() + 1];
   
    const int tensor_order = tensor.getOrder();
    const vType* tensor_dims = tensor.getDims();
    const vType nnzcount = tensor.getNNZCount();
    const vType* storage = tensor.getStorage();

    offsets[0] = 0;
    for(int i = 0; i < active_modes.size(); i++) {
        matrix_dim += tensor_dims[active_modes[i]];
        offsets[i + 1] = tensor_dims[active_modes[i]];
    }

    for(int i = 2; i <= active_modes.size(); i++) {
        offsets[i] += offsets[i - 1];
    }   
    
    double start_time = omp_get_wtime();
    int count = 0;
    for(vType nnz = 0; nnz < nnzcount; nnz++) {
        const vType* nnz_ids = storage + (nnz * tensor_order);
        for(int i = 0; i < active_modes.size(); i++) {
            if((offsets[i + 1] - offsets[i]) >= (matrix_dim / 100)) {
                vType id_i = nnz_ids[active_modes[i]] + offsets[i];
                for(int j = i + 1; j < active_modes.size(); j++) {
                    if((offsets[j + 1] - offsets[j]) >= (matrix_dim / 100)) {
                        vType id_j = nnz_ids[active_modes[j]] + offsets[j];
                        nnz_set.insert(vpair(id_i, id_j)); 
                        count += 1; 
                    }
                }    
            }
        }
    }

    double end_time = omp_get_wtime();
    #pragma omp critical 
    {
         std::cout << orderingName << "-" << this->tensor.getName() << ": time for k-partite " << nnz_set.size() << " nonzeros out of " << count << " " << end_time - start_time << std::endl;
    }
    
    std::vector<vpair> nonzeros;
    nonzeros.assign(nnz_set.begin(), nnz_set.end());

    eType old_nnz = nonzeros.size();
    for(eType e = 0; e < old_nnz; e++) nonzeros.push_back(vpair(nonzeros[e].second, nonzeros[e].first));
    for(vType v = 0; v < matrix_dim; v++) nonzeros.push_back(vpair(v, v));   

    eType matrix_nnz = nonzeros.size();
    eType* ptrs = new eType[matrix_dim + 1];
    memset(ptrs, 0, sizeof(eType) * (matrix_dim + 1));
    for (eType i = 0; i < matrix_nnz; i++) ptrs[nonzeros[i].second + 1]++;
    for (vType i = 2; i <= matrix_dim; i++) ptrs[i] += ptrs[i - 1];

    vType* temp_ids = new vType[matrix_nnz];
    for (eType i = 0; i < matrix_nnz; i++) temp_ids[ptrs[nonzeros[i].second]++] = nonzeros[i].first;
    for (vType i = matrix_dim; i > 0; i--) ptrs[i] = ptrs[i-1]; ptrs[0] = 0;
    
    eType* temp_ptrs = new eType[matrix_dim + 1];
    memcpy(temp_ptrs, ptrs, sizeof(eType) * (matrix_dim + 1));
    vType* ids = new vType[matrix_nnz];
    for(vType v = 0; v < matrix_dim; v++) {
        for (eType e = ptrs[v]; e < ptrs[v+1]; e++) {
            ids[temp_ptrs[temp_ids[e]]++] = v;
        }
    }
 
    valType* vals = new valType[matrix_nnz];
    for (eType i = 0; i < matrix_nnz; i++) vals[i] = 1;

    kpartite_matrix = new SparseMatrix(name + "_kpartite", matrix_dim, matrix_dim, matrix_nnz, ptrs, ids, vals, true, true);    
    
    double end_time2 = omp_get_wtime();
    #pragma omp critical 
    {
        std::cout << orderingName << "-" << this->tensor.getName() << ": time for k-partite sorting " << nonzeros.size() << " nonzeros out of " << count << " " << end_time2 - end_time << std::endl;
    }
    
    delete [] offsets;
    delete [] temp_ids;
    delete [] temp_ptrs;
    return kpartite_matrix;
}
