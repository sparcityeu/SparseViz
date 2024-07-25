//
// Created on 1/15/24.
//

#include "TensorNaturalOrdering.h"


void TensorNaturalOrdering::orderingFunction()
{
    if (orderedDimensions)
    {
        return;
    }
    
    vType order = tensor.getOrder();
    const vType* dims = tensor.getDims();

    orderedDimensions = new vType*[order];

    for (vType i = 0; i != order; ++i)
    {
        orderedDimensions[i] = new vType[dims[i]];
    } 
    
    for(const vType& m : active_modes)
    {
        for (vType j = 0; j != dims[m]; ++j)
        {
            orderedDimensions[m][j] = j;
        }
    }
}
