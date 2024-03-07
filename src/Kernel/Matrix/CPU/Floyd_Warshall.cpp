//
// Created on 2/11/24.
//

#include "Floyd_Warshall.h"


bool Floyd_Warshall::init(const SparseMatrix &A)
{
    if (!A.isSquare())
    {
        return false;
    }
    vertexCount = A.getRowCount();
    ptrs = A.getPtr();
    inds = A.getInd();
    vals = A.getValues();

    dists = nullptr;

    return true;
}

Floyd_Warshall::~Floyd_Warshall()
{
    for (vType i = 0; i != vertexCount; ++i)
    {
        delete[] dists[i];
    }
    delete[] dists;
}

void Floyd_Warshall::preprocess(const SparseMatrix &A)
{
    if (dists)
    {
        for (vType i = 0; i != vertexCount; ++i)
        {
            delete[] dists[i];
        }
        delete[] dists;
    }

    dists = new unsigned*[vertexCount];
    for (vType i = 0; i != vertexCount; ++i)
    {
        dists[i] = new unsigned[vertexCount];
    }

    for (vType s = 0; s != vertexCount; ++s)
    {
        dists[s][s] = 0;
    }

    for (vType s = 0; s != vertexCount; ++s)
    {
        for (vType neighbour = ptrs[s]; neighbour != ptrs[s + 1]; ++neighbour)
        {
            vType d = inds[neighbour];
            unsigned w = vals[neighbour];

            dists[s][d] = w;
        }
    }
}

void Floyd_Warshall::functionBody(const SparseMatrix &A, int iterNumber)
{
    for (vType k = 0; k != vertexCount - 1; ++k)
    {
        for (vType i = 0; i != vertexCount - 1; ++i)
        {
            for (vType j = 0; j != vertexCount - 1; ++j)
            {
                if (dists[i][k] + dists[k][j] < dists[i][j])
                {
                    dists[i][j] = dists[i][k] + dists[k][j];
                }
            }
        }
    }
}
