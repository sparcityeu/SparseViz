#include "AMDOrdering.h"
#include "Parameters.h"
#include <vector>
#include <algorithm>
#include <amd.h>


void AMDOrdering::orderingFunction()
{
    int n = this->getMatrix().getRowCount();
    const int* ptrs = (const int*) this->getMatrix().getPtr();
    const int* ids = (const int*) this->getMatrix().getInd();

    rowIPermutation = new vType[n];
    colIPermutation = new vType[n];
    for(int i = 0; i < n; i++)
    {
        rowIPermutation[i] = colIPermutation[i] = i;
    }

    int* i_order = new int[n];
    int status = amd_order(n, ptrs, ids, i_order,  (double *) NULL, (double *) NULL);
    if (status == AMD_INVALID)
    {
        logger.makeSilentLog("AMD is performing preprocessing on " + this->getMatrix().getName());
        int* r_ptrs = new int[n + 1];
        int* r_ids = new int[ptrs[n]];
        if (amd_preprocess (n, ptrs, ids, r_ptrs, r_ids) == AMD_OK)
        {
            status = amd_order(n, r_ptrs, r_ids, i_order,  (double *) NULL, (double *) NULL);
        }
    }
    if(status == AMD_OK)
    {
        for (unsigned i = 0; i < n; i++)
        {
            rowIPermutation[i_order[i]] = colIPermutation[i_order[i]] = i;
        }
    }
    else
    {
        logger.makeSilentLog("AMD is failed on " + this->getMatrix().getName());
        throw std::logic_error("AMD is failed on " + this->getMatrix().getName());
    }
}
