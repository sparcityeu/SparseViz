//
// Created on 1/15/24.
//

#include "TensorNaturalOrdering.h"
#include <cstring>

void TensorNaturalOrdering::orderingFunction()
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
    
    for(const vType& m : active_modes) { 
        for (vType j = 0; j != dims[m]; ++j) {
            orderedDimensions[m][j] = j;
        }
    }
}
