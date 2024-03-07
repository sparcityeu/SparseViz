#include "CudaHelloWorld.h"
#include "GPUKernels.cuh"
#include "cuda_runtime.h"


bool CudaHelloWorld::init(const SparseMatrix &A)
{
    hostOutput = new char[N];
    return true;
}

CudaHelloWorld::~CudaHelloWorld()
{
    delete[] hostOutput;
}

void CudaHelloWorld::hostFunction(const SparseMatrix &A, int iterNumber, int gridSize, int blockSize)
{
    char* deviceOutput;

    cudaMalloc(&deviceOutput, N * sizeof(char));

    cudaHelloWorldKernel<<<gridSize, blockSize>>>(deviceOutput);

    cudaMemcpy(hostOutput, deviceOutput, N * sizeof(char), cudaMemcpyDeviceToHost);

    std::cout << hostOutput << std::endl;

    cudaFree(deviceOutput);
}
