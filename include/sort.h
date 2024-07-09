//
// Created by deniz on 3/30/24.
//

#ifndef SPARSEVIZ_SORT_H
#define SPARSEVIZ_SORT_H

#include "config.h"


void countSort(std::vector< std::pair<vType, vType> >& inputArray, vType* permutation);


/*!
 * @brief Sorts the nonzeros in the order in which they appear in the storage.
 * @param order Order of the structure (this should be equal to the size of the dims array).
 * @param dims The dimensions that the nonzeros span across.
 * @param nonzeros The nonzero storage to be sorted, with a layout as: x1,y1,z1,x2,y2,z2 (example is in 3D space).
 * @details Works in linear time relative to the number of nonzeros.
 */
void sortNonzeros(vType order, vType* dims, eType nnz, vType* storage, valType* values);


#endif //SPARSEVIZ_SORT_H
