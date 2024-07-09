//
// Created by deniz on 5/8/24.
//

#include "JackardWeights.h"
#include "GPUHelpers.h"


__device__ int bSearch(const vType& vertexSearched, const vType* adj, const vType& adjSize)
{
    vType left = 0;
    vType right = adjSize - 1;

    while (left <= right)
    {
        vType mid = left + (right - left) / 2;
        vType currentVertex = adj[mid];
        if (vertexSearched == currentVertex)
        {
            return 1;
        }
        else if (vertexSearched < currentVertex)
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return -1;
}

__device__ unsigned d(const vType* ptrs, const vType& vertex)
{
    return ptrs[vertex + 1] - ptrs[vertex];
}

__global__ void jackardGlobal(const vType* ptrs, const vType* inds, vType M, double* results, unsigned assemblySize, unsigned searchGroupSize)
{
    __shared__ vType inter;
    if (threadIdx.x == 0)
    {
        inter = 0;
    }
    __syncthreads();

    for (vType assembly = blockIdx.x; assembly < M; assembly += assemblySize)
    {
        vType u = assembly;
        vType uDegree = d(ptrs, u);
        for (vType vidx = ptrs[u]; vidx < ptrs[u + 1]; ++vidx)
        {
            vType v = inds[vidx];
            vType vDegree = d(ptrs, v);
            if (uDegree > vDegree || (uDegree == vDegree && v > u))
            {
                if (bSearch(u, &inds[ptrs[v]], vDegree) != -1)
                {
                    continue;
                }
            }

            if (threadIdx.x == 0)
            {
                inter = 0;
            }
            __syncthreads();

            for (vType searchGroup = ptrs[u] + threadIdx.x; searchGroup < uDegree; searchGroup += searchGroupSize)
            {
                vType local = 0;
                for (vType xidx = ptrs[searchGroup]; xidx < ptrs[searchGroup + 1]; ++xidx)
                {
                    vType x = inds[xidx];
                    if (bSearch(x, &inds[ptrs[v]], vDegree) != -1)
                    {
                        ++local;
                    }
                }
                atomicAdd(&inter, local);
            }
            __syncthreads();

            if (threadIdx.x == 0)
            {
                double jackard = double(inter) / double(uDegree + vDegree - inter);
                results[u * M + v] = jackard;
                results[v * M + u] = jackard;
            }
            __syncthreads();
        }
    }
}


JackardWeights::JackardWeights(const std::string &kernelName, const std::vector<int> &gridSizes,
                               const std::vector<int> &blockSizes, const std::vector<int> &sharedMemorySizes,
                               std::string kernelParameters, int nRun, int nIgnore)
:   MatrixGPUKernel(kernelName, gridSizes, blockSizes, sharedMemorySizes, nRun, nIgnore)
{
    std::vector<std::string> parameters = split(kernelParameters, '/');
    assemblySize = std::stoi(parameters[0]);
    searchGroupSize = std::stoi(parameters[1]);
}

bool JackardWeights::init(const SparseMatrix &A)
{
    gpuErrchk( cudaMalloc(&ptrs, sizeof(vType) * (A.getRowCount() + 1)) );
    gpuErrchk( cudaMemcpy(ptrs, A.getPtr(), sizeof(vType) * (A.getRowCount() + 1), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc(&inds, sizeof(vType) * (A.getNNZCount())) );
    gpuErrchk( cudaMemcpy(inds, A.getInd(), sizeof(vType) * (A.getNNZCount()), cudaMemcpyHostToDevice) );

    return true;
}

JackardWeights::~JackardWeights()
{
    gpuErrchk( cudaFree(ptrs) );
    gpuErrchk( cudaFree(inds) );
}

void JackardWeights::hostFunction(const SparseMatrix &A, int iterNumber, int gridSize, int blockSize, int sharedMemorySize)
{
    double* hostResult = new double[A.getRowCount() * A.getRowCount()];

    double* deviceResult;
    gpuErrchk( cudaMalloc(&deviceResult, sizeof(double) * A.getRowCount() * A.getRowCount()) );

    jackardGlobal<<<assemblySize, searchGroupSize>>>(ptrs, inds, A.getRowCount(), deviceResult, assemblySize, searchGroupSize);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk( cudaMemcpy(hostResult, deviceResult, sizeof(double) * A.getRowCount() * A.getRowCount(), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(deviceResult) );
    delete[] hostResult;
}
