//
// Created on 2/11/24.
//

#include "Dijkstra.h"


bool Dijkstra::init(const SparseMatrix &A)
{
    if (!A.isSquare())
    {
        return false;
    }

    ptrs = A.getPtr();
    inds = A.getInd();
    vals = A.getValues();

    dists = nullptr;

    return true;
}

Dijkstra::~Dijkstra()
{
    delete[] dists;
}

void Dijkstra::preprocess(const SparseMatrix &A)
{
    delete[] dists;

    vType vertexCount = A.getRowCount();
    dists = new valType[vertexCount];

    source = 0;
    dists[source] = 0;

    for (vType v = 0; v != vertexCount; ++v)
    {
        if (v != source)
        {
            minHeap.emplace(v, UINT32_MAX);
            dists[v] = UINT32_MAX;
        }
    }
}

void Dijkstra::functionBody(const SparseMatrix &A, int iterNumber)
{
    while (!minHeap.empty())
    {
        pair u = minHeap.top();
        minHeap.pop();
        vType vertex = u.first;
        valType weight = u.second;

        if (weight != dists[vertex]) continue;

        for (vType i = ptrs[vertex]; i != ptrs[vertex + 1]; ++i)
        {
            vType adjVertex = inds[i];
            valType edgeWeight = vals[i];

            if (dists[adjVertex] > weight + edgeWeight)
            {
                dists[adjVertex] = weight + edgeWeight;
                minHeap.emplace(adjVertex, dists[adjVertex]);
            }
        }
    }
}

bool Dijkstra::MinHeapComparator::operator()(const Dijkstra::pair &pair1, const Dijkstra::pair &pair2)
{
    return pair1.second < pair2.second;
}
