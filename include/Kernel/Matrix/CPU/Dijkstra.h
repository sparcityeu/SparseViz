//
// Created on 2/11/24.
//

#ifndef SPARSEVIZ_DIJKSTRA_H
#define SPARSEVIZ_DIJKSTRA_H

#include "MatrixKernelFunction.h"
#include <queue>


class Dijkstra: public MatrixKernelFunction
{
    typedef std::pair<vType, unsigned> pair;
public:
    Dijkstra(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
    :   MatrixKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore) {}
    virtual ~Dijkstra() override;
    virtual bool init(const SparseMatrix& A) override;
    virtual void preprocess(const SparseMatrix& A) override;
    virtual void functionBody(const SparseMatrix& A, int iterNumber) override;

private:
    struct MinHeapComparator
    {
        bool operator()(const pair& pair1, const pair& pair2);
    };

public:
    vType source;
    valType* dists;

private:
    const vType* ptrs;
    const vType* inds;
    const valType* vals;

    std::priority_queue<pair, std::vector<pair>, MinHeapComparator> minHeap;
};


#endif //SPARSEVIZ_DIJKSTRA_H
