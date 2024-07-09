//
// Created on 2/11/24.
//

#include "Bellman_Ford.h"


bool Bellman_Ford::init(const SparseMatrix &A)
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

Bellman_Ford::~Bellman_Ford()
{
    delete[] dists;
}

void Bellman_Ford::preprocess(const SparseMatrix &A)
{
    delete[] dists;

    vType vertexCount = A.getRowCount();
    dists = new valType[vertexCount];

    singleSource = 0;
    dists[singleSource] = 0;

    for (vType v = 0; v != vertexCount; ++v)
    {
        if (v != singleSource)
        {
            dists[v] = UINT32_MAX;
        }
    }
}

void Bellman_Ford::functionBody(const SparseMatrix &A, int iterNumber)
{
    vType vertexCount = A.getRowCount();
    for (vType i = 0; i != vertexCount - 1 ; ++i)
    {
        bool relaxed = false;
        for (vType source = 0; source != vertexCount; ++source)
        {
            for (vType neighbour = ptrs[source]; neighbour != ptrs[source + 1]; ++neighbour)
            {
                vType destination = inds[neighbour];
                valType weight = vals[neighbour];

                if (dists[destination] > dists[source] + weight)
                {
                    relaxed = true;
                    dists[destination] = dists[source] + weight;
                }
            }
        }
        if (!relaxed)
        {
            break;
        }
    }

    for (vType source = 0; source != vertexCount; ++source)
    {
        for (vType neighbour = ptrs[source]; neighbour != ptrs[source + 1]; ++neighbour)
        {
            vType destination = inds[neighbour];
            valType weight = vals[neighbour];

            if (dists[destination] > dists[source] + weight)
            {
                std::cout << "Your graph contains negative weight cycle. Bellman_Ford would not yield correct results." << std::endl;
            }
        }
    }
}
