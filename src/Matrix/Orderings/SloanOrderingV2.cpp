//
// SloanOrderingV2.cpp — optimized Sloan ordering
// Key changes vs V1:
//   1. Lazy-deletion max-heap in Numbering (O(n^2) -> O((n+m) log n))
//   2. Touched-list BFS resets in buildLevelStructure (O(n) per call -> O(component))
//   3. Flat vector BFS queue instead of std::queue
//   4. Monotone seed cursor in orderingFunction (O(n) total vs O(n) per component)
//
#include "SloanOrdering.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <vector>
#include "SparseMatrix.h"

// ── scratch struct + BFS helper (anonymous, this TU only) ────────────────────
namespace {

struct SloanScratch {
    std::vector<int>                   level;
    std::vector<unsigned>              touched_level;
    std::vector<unsigned>              bfs_q;
    std::vector<std::vector<unsigned>> byLevel;
    std::vector<unsigned>              byLevel_touched;

    void resize(unsigned n) {
        level.assign(n, -1);
        touched_level.reserve(n);
        bfs_q.resize(n);
    }

    void reset_level() {
        for (unsigned v : touched_level) level[v] = -1;
        touched_level.clear();
    }

    void reset_byLevel() {
        for (unsigned lv : byLevel_touched) byLevel[lv].clear();
        byLevel_touched.clear();
    }
};

// BFS from root, filling sc.level / sc.byLevel.
// Early-exits (with partial results) if width > widthMin.
static void bls_impl(
    unsigned root,
    const vType* ptr, const vType* ind, unsigned n,
    SloanScratch& sc,
    unsigned& out_depth, unsigned& out_width,
    unsigned widthMin
) {
    sc.reset_level();
    sc.reset_byLevel();
    out_depth = 0;
    out_width = 0;

    sc.level[root] = 0;
    sc.touched_level.push_back(root);

    if (sc.byLevel.empty()) sc.byLevel.emplace_back();
    sc.byLevel[0].push_back(root);
    sc.byLevel_touched.push_back(0);
    out_width = 1;

    unsigned head = 0, tail = 0;
    sc.bfs_q[tail++] = root;

    while (head < tail) {
        const unsigned v  = sc.bfs_q[head++];
        const unsigned nL = static_cast<unsigned>(sc.level[v]) + 1;

        for (vType k = ptr[v]; k < ptr[v + 1]; ++k) {
            const unsigned u = static_cast<unsigned>(ind[k]);
            if (u >= n || sc.level[u] != -1) continue;

            sc.level[u] = static_cast<int>(nL);
            sc.touched_level.push_back(u);

            if (sc.byLevel.size() <= nL) sc.byLevel.resize(nL + 1);
            if (sc.byLevel[nL].empty())  sc.byLevel_touched.push_back(nL);
            sc.byLevel[nL].push_back(u);
            sc.bfs_q[tail++] = u;

            const unsigned sz = static_cast<unsigned>(sc.byLevel[nL].size());
            if (sz > out_width) out_width = sz;
            if (out_width > widthMin) {
                out_depth = static_cast<unsigned>(sc.byLevel.size());
                return;
            }
        }
    }

    out_depth = static_cast<unsigned>(sc.byLevel.size());
    while (out_depth > 0 && sc.byLevel[out_depth - 1].empty()) --out_depth;
}

// Pseudo-peripheral endpoint search, reusing a shared scratch for all BFS calls.
static void find_endpoints_impl(
    unsigned s,
    const vType* ptr, const vType* ind, unsigned n,
    const std::vector<unsigned>& degree,
    SloanScratch& sc,
    unsigned& sOut, unsigned& eOut
) {
    sOut = s; eOut = s;
    if (n == 0) return;

    std::vector<unsigned> Q_cands;

    while (true) {
        unsigned depth_s, width_s;
        bls_impl(s, ptr, ind, n, sc, depth_s, width_s,
                 std::numeric_limits<unsigned>::max());

        if (depth_s == 0) { sOut = s; eOut = s; return; }

        const std::vector<unsigned>& last = sc.byLevel[depth_s - 1];
        if (last.empty())  { sOut = s; eOut = s; return; }

        Q_cands = last;
        const size_t m = Q_cands.size();
        std::sort(Q_cands.begin(), Q_cands.end(), [&](unsigned a, unsigned b) {
            return degree[a] != degree[b] ? degree[a] < degree[b] : a < b;
        });
        Q_cands.resize((m + 2) / 2);

        unsigned wMin  = std::numeric_limits<unsigned>::max();
        const unsigned hMax = depth_s;
        unsigned e     = s;
        bool     restart = false;

        for (unsigned i : Q_cands) {
            unsigned depth_i, width_i;
            bls_impl(i, ptr, ind, n, sc, depth_i, width_i, wMin);

            if (depth_i > hMax && width_i < wMin) {
                s = i; restart = true; break;
            }
            if (width_i < wMin) { e = i; wMin = width_i; }
        }

        if (restart) continue;
        sOut = s; eOut = e;
        return;
    }
}

} // namespace

// ── SloanOrdering class method implementations ───────────────────────────────

void SloanOrdering::buildLevelStructure(
    unsigned root,
    const vType* ptr, const vType* ind, unsigned n,
    LevelStructure& out, unsigned widthMin
) {
    SloanScratch sc;
    sc.resize(n);
    unsigned depth, width;
    bls_impl(root, ptr, ind, n, sc, depth, width, widthMin);

    out.depth = depth;
    out.width = width;
    out.level.assign(n, -1);
    for (unsigned v : sc.touched_level) out.level[v] = sc.level[v];
    out.byLevel.assign(depth, {});
    for (unsigned lv : sc.byLevel_touched)
        if (lv < depth) out.byLevel[lv] = sc.byLevel[lv];
}

void SloanOrdering::findPseudoPeripheralEndpoints(
    unsigned s,
    const vType* ptr, const vType* ind, unsigned n,
    const std::vector<unsigned>& degree,
    unsigned& sOut, unsigned& eOut
) {
    SloanScratch sc;
    sc.resize(n);
    find_endpoints_impl(s, ptr, ind, n, degree, sc, sOut, eOut);
}

void SloanOrdering::initializeNodeInfo(
    std::vector<NodeInfo>& info,
    const LevelStructure& Le,
    unsigned n,
    const std::vector<unsigned>& degree,
    const int& W1, const int& W2
) {
    info.assign(n, NodeInfo{});
    const std::vector<int>& dist = Le.level;

    for (unsigned v = 0; v < n; ++v) {
        NodeInfo& ni   = info[v];
        ni.degree      = degree[v];
        ni.currentDegree = ni.degree + 1;
        ni.distToEnd   = dist[v];
        ni.priority    = (ni.distToEnd < 0)
            ? std::numeric_limits<int>::min()
            : static_cast<int>(n - ni.currentDegree) * W1 + ni.distToEnd * W2;
    }
}

void SloanOrdering::Numbering(
    unsigned s,
    const vType* ptr, const vType* ind, unsigned n,
    int W1,
    std::vector<NodeInfo>& info,
    std::vector<unsigned>& /* eligible — unused in V2 */,
    std::vector<unsigned>& labels
) {
    labels.assign(n, std::numeric_limits<unsigned>::max());

    using Pair = std::pair<int, unsigned>;
    std::priority_queue<Pair> heap;

    info[s].status = Status::Preactive;
    heap.push({info[s].priority, s});

    unsigned nextLabel = 0;

    while (!heap.empty()) {
        const auto [p, i] = heap.top();
        heap.pop();

        if (info[i].status == Status::Postactive) continue;
        if (p != info[i].priority)                continue; // stale entry

        const bool wasPreactive = (info[i].status == Status::Preactive);

        // Step 7: if i was Preactive, update all neighbors
        if (wasPreactive) {
            for (vType kk = ptr[i]; kk < ptr[i + 1]; ++kk) {
                const unsigned j = static_cast<unsigned>(ind[kk]);
                if (j >= n) continue;
                info[j].priority += W1;
                if (info[j].status == Status::Inactive) {
                    info[j].status = Status::Preactive;
                    heap.push({info[j].priority, j});
                } else if (info[j].status != Status::Postactive) {
                    heap.push({info[j].priority, j}); // lazy re-push for Active/Preactive
                }
            }
        }

        // Step 8: label i -> Postactive
        labels[i]      = nextLabel++;
        info[i].status = Status::Postactive;

        // Step 9: promote each Preactive neighbor of i to Active
        for (vType kk = ptr[i]; kk < ptr[i + 1]; ++kk) {
            const unsigned j = static_cast<unsigned>(ind[kk]);
            if (j >= n || info[j].status != Status::Preactive) continue;

            info[j].priority += W1;
            info[j].status    = Status::Active;
            heap.push({info[j].priority, j});

            for (vType kk2 = ptr[j]; kk2 < ptr[j + 1]; ++kk2) {
                const unsigned k = static_cast<unsigned>(ind[kk2]);
                if (k >= n) continue;
                if (info[k].status == Status::Active ||
                    info[k].status == Status::Preactive) {
                    info[k].priority += W1;
                    heap.push({info[k].priority, k});
                } else if (info[k].status == Status::Inactive) {
                    info[k].priority += W1;
                    info[k].status    = Status::Preactive;
                    heap.push({info[k].priority, k});
                }
            }
        }
    }
}

void SloanOrdering::orderingFunction() {
    const SparseMatrix& A = this->getMatrix();
    const unsigned n = A.getRowCount();
    if (n == 0) return;

    const vType* ptr = A.getPtr();
    const vType* ind = A.getInd();

    rowIPermutation = new vType[n];
    colIPermutation = new vType[n];

    const unsigned UNLABELED = std::numeric_limits<unsigned>::max();

    std::vector<unsigned> degree(n);
    for (unsigned v = 0; v < n; ++v)
        degree[v] = static_cast<unsigned>(ptr[v + 1] - ptr[v]);

    std::vector<unsigned> globalLabels(n, UNLABELED);
    unsigned globalLabel = 0;

    SloanScratch sc;
    sc.resize(n);

    // Reused across components to avoid repeated allocations
    std::vector<NodeInfo> info;
    std::vector<unsigned> eligible;
    std::vector<unsigned> localLabels;

    unsigned seed_cursor = 0; // monotone — O(n) total scan

    while (globalLabel < n) {
        while (seed_cursor < n && globalLabels[seed_cursor] != UNLABELED)
            ++seed_cursor;
        if (seed_cursor >= n) break;
        const unsigned seed = seed_cursor;

        if (degree[seed] == 0) {
            globalLabels[seed] = globalLabel++;
            ++seed_cursor;
            continue;
        }

        // Find pseudo-peripheral endpoints using shared scratch
        unsigned s = seed, e = seed;
        find_endpoints_impl(seed, ptr, ind, n, degree, sc, s, e);

        // Build L(e) — needed for initializeNodeInfo
        unsigned depth_e, width_e;
        bls_impl(e, ptr, ind, n, sc, depth_e, width_e,
                 std::numeric_limits<unsigned>::max());

        // Populate LevelStructure from scratch (O(n) for Le.level — once per component)
        LevelStructure Le;
        Le.depth = depth_e;
        Le.width = width_e;
        Le.level.assign(n, -1);
        for (unsigned v : sc.touched_level) Le.level[v] = sc.level[v];

        initializeNodeInfo(info, Le, n, degree, W1, W2);

        Numbering(s, ptr, ind, n, W1, info, eligible, localLabels);

        // Commit labels (iterate only component vertices via sc.touched_level)
        unsigned labeledThisRun = 0;
        for (unsigned v : sc.touched_level) {
            if (localLabels[v] != UNLABELED && globalLabels[v] == UNLABELED) {
                globalLabels[v] = globalLabel + localLabels[v];
                ++labeledThisRun;
            }
        }
        globalLabel += labeledThisRun;
    }

    for (unsigned v = 0; v < n; ++v) {
        rowIPermutation[v] = static_cast<vType>(globalLabels[v]);
        colIPermutation[v] = static_cast<vType>(globalLabels[v]);
    }
}
