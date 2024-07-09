//
// Created by delbek on 7/7/24.
//

#ifndef SPARSEVIZ_GPUHELPERS_H
#define SPARSEVIZ_GPUHELPERS_H

#include "cuda_runtime.h"
#include "config.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


#endif //SPARSEVIZ_GPUHELPERS_H
