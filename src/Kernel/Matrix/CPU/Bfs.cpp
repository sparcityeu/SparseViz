//
// Created on 1/18/24.
//

#include "Bfs.h"


bool BFS::init(const SparseMatrix &A)
{
    if (!A.isPatternSymmetric())
    {
        std::cout << "BFS Kernel is terminating: matrix needs to be symettric" << std::endl;
        return false;
    }

    m = A.getRowCount();
    ptrs = A.getPtr();
    ids = A.getInd();
    level = new valType[m];

    for(vType i = 0; i < m; i++) level[i] = -1;

    return true;
}

BFS::~BFS()
{
    delete[] level;
}

void BFS::preprocess(const SparseMatrix& A) {
    for(int i = 0; i < m; i++) {
        level[i] = -1;
    }
}

void BFS::functionBody(const SparseMatrix& A, int iterNumber)
{
    vType root = (iterNumber * 492366587) % m;
    if(root < 0) root += m;
   
    level[root] = 0;
    int clevel = 0;
    bool conti = true;

    //kernel starts
    while(conti) {
        conti = false;
#pragma omp parallel for
        for (vType v = 0; v < m; v++)  {
            if(level[v] == clevel) {
                for (vType ptr = ptrs[v]; ptr < ptrs[v + 1]; ptr++) {
                    vType nbr = ids[ptr];
                    if(level[nbr] == -1) {
                        level[nbr] = clevel + 1;
                        conti = true;
                    }
                }
            }
        }
        clevel++;
    }
}

void BFS::postprocess(const SparseMatrix& A, int iter) {

    int count = 0;
    double sum = 0;
    int maxlevel = 0;
    for(int i = 0; i < m; i++) {
        if(level[i] != -1) {
            count++;
            sum += level[i];
            maxlevel = std::max(maxlevel, (int)level[i]);
        }
    }
    logger->makeSilentLog("Parallel BFS Log - matrix: " + A.getName() + " - iter: " + std::to_string(iter) + " - " + std::to_string(count) + " " + std::to_string(sum/count) + " " + std::to_string(maxlevel));
}
