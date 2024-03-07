#include "GPUSpmvRowBased.h"
#include "GPUKernels.cuh"
#include "cuda_runtime.h"


bool GPUSpmvRowBased::init(const SparseMatrix &A)
{
    inputVector = new valType[A.getColCount()];
    outputVector = new valType[A.getRowCount()];

    vType inputM = A.getRowCount();

    gpuErrchk( cudaMalloc(&deviceM, sizeof(vType)));
    gpuErrchk( cudaMemcpy(deviceM, &inputM, sizeof(vType), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc(&ptrs, (A.getRowCount() + 1) * sizeof(vType)));
    gpuErrchk( cudaMemcpy(ptrs, A.getPtr(), (A.getRowCount() + 1) * sizeof(vType), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc(&inds, A.getNNZCount() * sizeof(vType)));
    gpuErrchk( cudaMemcpy(inds, A.getInd(), A.getNNZCount() * sizeof(vType), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc(&vals, A.getNNZCount() * sizeof(valType)));
    gpuErrchk( cudaMemcpy(vals, A.getValues(), A.getNNZCount() * sizeof(valType), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc(&deviceInput, A.getColCount() * sizeof(valType)));
    gpuErrchk( cudaMemcpy(deviceInput, inputVector, A.getColCount() * sizeof(valType), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMalloc(&deviceOutput, A.getRowCount() * sizeof(valType)));

    return true;
}

GPUSpmvRowBased::~GPUSpmvRowBased()
{
    gpuErrchk( cudaFree(deviceM));
    gpuErrchk( cudaFree(ptrs));
    gpuErrchk( cudaFree(inds));
    gpuErrchk( cudaFree(vals));
    gpuErrchk( cudaFree(deviceInput));
    gpuErrchk( cudaFree(deviceOutput));
    delete[] inputVector;
    delete[] outputVector;
}

void GPUSpmvRowBased::hostFunction(const SparseMatrix &A, int iterNumber, int gridSize, int blockSize)
{
    for(int i = 0; i < 10; i++) {
        spmvRowBased<<<gridSize, blockSize>>>(ptrs, inds, vals, deviceM, deviceInput, deviceOutput);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
} 

