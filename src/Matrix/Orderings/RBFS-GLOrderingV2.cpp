//
// RBFS-GLOrderingV2.cpp — optimized RBFS-GL ordering
// Key changes vs V1:
//   1. level[] allocated once; per-component reset via touched-list (O(comp) not O(n))
//   2. Min-degree vertex chosen as BFS root per component (better ordering quality)
//
#include "RBFS-GLOrdering.h"
#include <queue>
using std::vector;

namespace {

struct RBFSGLNode {
    int      level;
    unsigned degree;
    unsigned id;
};

struct RBFSGLNodeCompare {
    bool operator()(const RBFSGLNode& a, const RBFSGLNode& b) const {
        if (a.level  != b.level)  return a.level  > b.level;
        if (a.degree != b.degree) return a.degree > b.degree;
        return a.id > b.id;
    }
};

struct RBFSGLScratch {
    vector<int>      level;    // size n, all -1; reset via touched_level
    vector<unsigned> touched_level;

    void resize(unsigned n) {
        level.assign(n, -1);
        touched_level.reserve(n);
    }

    void reset() {
        for (unsigned v : touched_level) level[v] = -1;
        touched_level.clear();
    }
};

// BFS from root using Gibbs-Lewis priority queue.
// Uses pre-allocated scratch for the level array.
static void bfsGL_impl(
    unsigned root,
    const vType* ptr, const vType* ind,
    const vector<unsigned>& degree,
    vector<char>& assigned,
    vector<unsigned>& visitOrder,
    RBFSGLScratch& sc
) {
    sc.reset();

    std::priority_queue<RBFSGLNode, vector<RBFSGLNode>, RBFSGLNodeCompare> pq;

    sc.level[root] = 0;
    sc.touched_level.push_back(root);
    assigned[root]  = 1;
    pq.push(RBFSGLNode{0, degree[root], root});

    while (!pq.empty()) {
        const RBFSGLNode cur = pq.top();
        pq.pop();

        const unsigned v  = cur.id;
        visitOrder.push_back(v);
        const int lv = cur.level;

        for (vType kk = ptr[v]; kk < ptr[v + 1]; ++kk) {
            const unsigned u = static_cast<unsigned>(ind[kk]);
            if (assigned[u]) continue;

            assigned[u]      = 1;
            sc.level[u]      = lv + 1;
            sc.touched_level.push_back(u);
            pq.push(RBFSGLNode{lv + 1, degree[u], u});
        }
    }
}

} // namespace

void RBFSGLOrdering::orderingFunction() {
    const SparseMatrix& A = this->getMatrix();
    const unsigned n = A.getRowCount();

    const vType* ptr = A.getPtr();
    const vType* ind = A.getInd();

    rowIPermutation = new vType[n];
    colIPermutation = new vType[n];

    vector<unsigned> degree(n);
    for (unsigned v = 0; v < n; ++v)
        degree[v] = static_cast<unsigned>(ptr[v + 1] - ptr[v]);

    vector<char>     assigned(n, 0);
    vector<unsigned> visitOrder;
    visitOrder.reserve(n);

    RBFSGLScratch sc;
    sc.resize(n);

    for (unsigned v = 0; v < n; ++v) {
        if (assigned[v]) continue;

        // Pick root as the minimum-degree neighbor of v (single-hop heuristic).
        // Guaranteed to be in the same component; O(degree[v]) cost per component.
        // The GL priority queue already orders by degree within levels, so this
        // mainly improves the starting point of the traversal.
        unsigned root   = v;
        unsigned minDeg = degree[v];
        for (vType kk = ptr[v]; kk < ptr[v + 1]; ++kk) {
            const unsigned u = static_cast<unsigned>(ind[kk]);
            if (degree[u] < minDeg) { minDeg = degree[u]; root = u; }
        }

        bfsGL_impl(root, ptr, ind, degree, assigned, visitOrder, sc);
    }

    // RBFS: reverse the visit order
    for (unsigned newPos = 0; newPos < n; ++newPos) {
        const unsigned v = visitOrder[n - 1 - newPos];
        rowIPermutation[v] = static_cast<vType>(newPos);
        colIPermutation[v] = static_cast<vType>(newPos);
    }
}

// Keep V1's bfsGL defined (declared in header) — delegates to the new impl.
void RBFSGLOrdering::bfsGL(unsigned root,
                            const vector<unsigned>& degree,
                            vector<char>& assigned,
                            vector<unsigned>& visitOrder)
{
    const SparseMatrix& A = this->getMatrix();
    const vType* ptr = A.getPtr();
    const vType* ind = A.getInd();

    RBFSGLScratch sc;
    sc.resize(A.getRowCount());
    bfsGL_impl(root, ptr, ind, degree, assigned, visitOrder, sc);
}
