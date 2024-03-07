#include "COOTensorGPUKernelSoA.h"
#include "GPUKernels.cuh"
#include "cuda_runtime.h"

bool COOTensorGPUKernelSoA::init(const SparseTensor &A)
{        
    h_output = 0.0;
    h_nonzeros = A.getStorage();
    dims = A.getDims();
    nnzcount = A.getNNZCount();
    order = A.getOrder();

    vType* dim_offsets = new vType[order + 1];
    dim_offsets[0] = 0;
    for(int i = 0; i < order; i++) dim_offsets[i+1] = dims[i];

    h_arrays = new float[dim_offsets[order]];
    for(vType i = 0; i < dim_offsets[order]; i++) {
        h_arrays[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    cudaMalloc(&d_arrays, sizeof(float) * dim_offsets[order]);
    cudaMemcpy(d_arrays, h_arrays,  sizeof(float) * dim_offsets[order], cudaMemcpyHostToDevice);

    vType* nonzeros_soa = new vType[nnzcount * order];
    for(eType i = 0; i < nnzcount; i++) {
        for(int j = 0; j < order; j++) {
            nonzeros_soa[(j * nnzcount) + i] = h_nonzeros[(i * order) + j] + dim_offsets[j];    
        }
    }

    gpuErrchk( cudaMalloc(&d_nonzeros, nnzcount * order * sizeof(vType)));
    gpuErrchk( cudaMemcpy(d_nonzeros, nonzeros_soa, nnzcount * order * sizeof(vType), cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMalloc(&d_output, sizeof(float)));
    gpuErrchk( cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice));
    
    delete [] nonzeros_soa;
    delete [] dim_offsets;
    return true;
}

COOTensorGPUKernelSoA::~COOTensorGPUKernelSoA()
{
    gpuErrchk( cudaFree(d_arrays));
    delete [] h_arrays;
    
    gpuErrchk( cudaFree(d_nonzeros));
    gpuErrchk( cudaFree(d_output));

}

void COOTensorGPUKernelSoA::hostFunction(const SparseTensor &A, int iterNumber, int gridSize, int blockSize)
{
    for(int i = 0; i < 10; i++) {
        cooTensorAoS<<<gridSize, blockSize>>>(d_nonzeros, nnzcount, order, d_arrays, d_output);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
