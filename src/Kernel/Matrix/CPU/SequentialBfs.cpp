//
// Created on 1/19/24.
//

#include "SequentialBfs.h"


bool SequentialBFS::init(const SparseMatrix &A)
{
    if (!A.isPatternSymmetric()) {
        std::cout << "SequentialBFS Kernel is terminating: matrix needs to be symettric" << std::endl;
        return false;
    }

    m = A.getRowCount();
    ptrs = A.getPtr();
    ids = A.getInd();
    level = new valType[m];
    queue = new valType[m];

    for(vType i = 0; i < m; i++) level[i] = -1;

    return true;
}

void SequentialBFS::preprocess(const SparseMatrix& A) {
    for(int i = 0; i < m; i++) {
        level[i] = -1;
    }
}

SequentialBFS::~SequentialBFS()
{
    delete [] level;
    delete [] queue;
}

void SequentialBFS::functionBody(const SparseMatrix &A, int iterNumber)
{
    vType root = (iterNumber * 492366587) % m;
    if(root < 0) root += m;

    level[root] = 0;
    queue[0] = root;
    vType qr = 0, qw = 1;

    //kernel starts
    while(qr < qw) {
        vType v = queue[qr++];
        int clevel = level[v];
        //std::cout << v << " " << ptrs[v] << " " << ptrs[v + 1] << std::endl;
        for (vType ptr = ptrs[v]; ptr < ptrs[v + 1]; ptr++) {
            //if(ptr - ptrs[v] < 5)
                //std::cout << "\t" << " " << ids[ptr] << std::endl;
            vType nbr = ids[ptr];
            if(level[nbr] == -1) {
                level[nbr] = clevel + 1;
                queue[qw++] = nbr;
            }
        }
    }
}

void SequentialBFS::postprocess(const SparseMatrix& A, int iter) {

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
    logger->makeSilentLog("Sequential BFS Log - matrix: " + A.getName() + " - iter: " + std::to_string(iter) + " - " + std::to_string(count) + " " + std::to_string(sum/count) + " " + std::to_string(maxlevel));
}