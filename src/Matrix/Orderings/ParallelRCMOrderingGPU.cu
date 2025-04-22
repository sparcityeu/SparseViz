//
// Created by delbek on 10/24/24.
//
#include "ParallelRCMOrderingGPU.h"
#include "cuda_runtime.h"
#include "GPUHelpers.h"


__global__ void pushBasedBFS(eType* ptrs, vType* inds, vType* levels, vType* sources, vType* frontiers, vType* frontierNo, vType currentLevel)
{
    vType threadID = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadID < *frontierNo)
    {
        vType vertex = frontiers[threadID];
        for (eType ptr = ptrs[vertex]; ptr < ptrs[vertex + 1]; ++ptr)
        {
            vType nbr = inds[ptr];
            vType level = atomicMin(&levels[nbr], currentLevel);
            if (level >= currentLevel)
            {
                atomicMin(&sources[nbr], vertex);
            }
        }
    }
}

ParallelRCMOrderingGPU::ParallelRCMOrderingGPU(SparseMatrix &matrix, std::string orderingName,
                                               std::string orderingParameters)
:   MatrixOrdering(matrix, orderingName, false, false)
{

}

void ParallelRCMOrderingGPU::orderingFunction()
{
    eType* h_ptrs = this->getMatrix().getPtr();
    vType* h_inds = this->getMatrix().getInd();
    vType n = this->getMatrix().getRowCount();
    eType nnz = this->getMatrix().getNNZCount();

    vType* h_levels = new vType[n];
    vType* h_sources = new vType[n];
    vType* h_frontiers = new vType[n];
    vType h_frontierNo;

    memset(h_sources, UINT32_MAX, sizeof(vType) * n);
    memset(h_levels, UINT32_MAX, sizeof(vType) * n);

    eType* d_ptrs;
    eType* d_inds;
    vType* d_levels;
    vType* d_frontiers;
    vType* d_frontierNo;
    vType* d_sources;

    gpuErrchk(cudaMalloc(&d_ptrs, sizeof(eType) * (n + 1)))
    gpuErrchk(cudaMalloc(&d_inds, sizeof(vType) * nnz))
    gpuErrchk(cudaMalloc(&d_levels, sizeof(vType) * n))
    gpuErrchk(cudaMalloc(&d_frontiers, sizeof(vType) * n))
    gpuErrchk(cudaMalloc(&d_frontierNo, sizeof(vType)))
    gpuErrchk(cudaMalloc(&d_sources, sizeof(vType) * n))

    gpuErrchk(cudaMemcpy(d_ptrs, h_ptrs, sizeof(eType) * (n + 1), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_inds, h_inds, sizeof(vType) * nnz, cudaMemcpyHostToDevice))

    vType root = 0;
    h_frontiers[0] = root;
    h_frontierNo = 1;
    vType currentLevel = 0;
    h_levels[root] = currentLevel;
    h_sources[root] = 0;

    gpuErrchk(cudaMemcpy(d_levels, h_levels, sizeof(vType) * n, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_sources, h_sources, sizeof(vType) * n, cudaMemcpyHostToDevice))

    while (h_frontierNo != 0)
    {
        ++currentLevel;

        gpuErrchk(cudaMemcpy(d_frontiers, h_frontiers, sizeof(vType) * h_frontierNo, cudaMemcpyHostToDevice))
        gpuErrchk(cudaMemcpy(d_frontierNo, &h_frontierNo, sizeof(vType), cudaMemcpyHostToDevice))

        unsigned gridDimension = std::min((int)h_frontierNo, 80);
        unsigned blockDimension = std::ceil(h_frontierNo / double(80));
        pushBasedBFS<<<gridDimension, blockDimension>>>(d_ptrs, d_inds, d_levels, d_sources, d_frontiers, d_frontierNo, currentLevel);
        gpuErrchk(cudaDeviceSynchronize())

        gpuErrchk(cudaMemcpy(h_levels, d_levels, sizeof(vType) * n, cudaMemcpyDeviceToHost))

        h_frontierNo = 0;
        for (vType i = 0; i < n; ++i)
        {
            if (h_levels[i] == currentLevel)
            {
                h_frontiers[h_frontierNo++] = i;
            }
        }
    }

    gpuErrchk(cudaMemcpy(h_sources, d_sources, sizeof(vType) * n, cudaMemcpyDeviceToHost))

    std::vector<std::vector<vType>> buckets(n);
    for (vType i = 0; i < n; ++i)
    {
        buckets[h_sources[i]].emplace_back(i);
    }

    vType* permutation = new vType[n];
    vType index = 0;
    for (auto& bucket: buckets)
    {
        std::sort(bucket.begin(), bucket.end(), [&h_ptrs](int u, int v)
        {
            return (h_ptrs[u+1] - h_ptrs[u]) < (h_ptrs[v+1] - h_ptrs[v]);
        });
        for (vType i = 0; i < bucket.size(); ++i)
        {
            permutation[index++] = bucket[i];
        }
    }

    vType* rowIPermutation = new vType[n];
    vType* colIPermutation = new vType[n];

    for (vType i = 0; i < n; ++i)
    {
        rowIPermutation[permutation[i]] = i;
        colIPermutation[permutation[i]] = i;
    }

    gpuErrchk(cudaFree(d_ptrs))
    gpuErrchk(cudaFree(d_inds))
    gpuErrchk(cudaFree(d_levels))
    gpuErrchk(cudaFree(d_sources))
    gpuErrchk(cudaFree(d_frontiers))
    gpuErrchk(cudaFree(d_frontierNo))

    delete[] h_levels;
    delete[] h_sources;
    delete[] h_frontiers;
    delete[] permutation;
}
