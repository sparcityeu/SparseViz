//
// Created on 1/15/24.
//

#ifndef SPARSEVIZ_TENSORNATURALORDERING_H
#define SPARSEVIZ_TENSORNATURALORDERING_H

#include "TensorOrdering.h"
#include <string>

class TensorNaturalOrdering: public TensorOrdering
{
public:
    TensorNaturalOrdering(SparseTensor& tensor, const std::vector<vType>& active_modes, std::string orderingName)
        : TensorOrdering(tensor, active_modes, orderingName) {

            
        }

private:
    virtual void orderingFunction() override;
};


#endif //SPARSEVIZ_TENSORNATURALORDERING_H
