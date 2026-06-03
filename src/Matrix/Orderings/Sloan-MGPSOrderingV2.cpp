//
// Sloan-MGPSOrderingV2.cpp — optimized Sloan-MGPS ordering
// Key changes vs V1:
//   1. Inherits heap-based Numbering from SloanOrderingV2 (O(n^2) -> O((n+m) log n))
//   2. Touched-list BFS resets (O(n) per call -> O(component))
//   3. Flat vector BFS queue
//   4. isNeighborOfAnyPicked uses flat bool array (O(1) lookup vs O(|picked|))
//   5. Monotone seed cursor
//
#include "Sloan-MGPSOrdering.h"
#include <algorithm>
#include <limits>
#include <vector>
#include "SparseMatrix.h"

// ── scratch + BFS helper (same pattern as SloanOrderingV2, duplicated per-TU) ─
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

// MGPS endpoint search — limits to 5 non-adjacent candidates.
// Uses a flat bool array (is_picked) for O(1) adjacency checks.
static void find_endpoints_mgps_impl(
    unsigned s,
    const vType* ptr, const vType* ind, unsigned n,
    const std::vector<unsigned>& degree,
    SloanScratch& sc,
    std::vector<bool>& is_picked,   // size n, all false on entry and exit
    unsigned& sOut, unsigned& eOut
) {
    sOut = s; eOut = s;
    if (n == 0) return;

    std::vector<unsigned> Q_sorted;
    std::vector<unsigned> Qpicked;
    Qpicked.reserve(5);

    while (true) {
        unsigned depth_s, width_s;
        bls_impl(s, ptr, ind, n, sc, depth_s, width_s,
                 std::numeric_limits<unsigned>::max());

        if (depth_s == 0) { sOut = s; eOut = s; return; }

        const std::vector<unsigned>& last = sc.byLevel[depth_s - 1];
        if (last.empty()) { sOut = s; eOut = s; return; }

        // Sort last level by ascending degree, then pick up to 5 non-adjacent
        Q_sorted = last;
        std::sort(Q_sorted.begin(), Q_sorted.end(), [&](unsigned a, unsigned b) {
            return degree[a] != degree[b] ? degree[a] < degree[b] : a < b;
        });

        Qpicked.clear();
        for (unsigned v : Q_sorted) {
            if (Qpicked.size() == 5) break;
            // O(degree[v]) adjacency check using flat bool array
            bool adjacent = false;
            for (vType k = ptr[v]; k < ptr[v + 1]; ++k) {
                if (is_picked[static_cast<unsigned>(ind[k])]) {
                    adjacent = true; break;
                }
            }
            if (adjacent) continue;
            is_picked[v] = true;
            Qpicked.push_back(v);
        }
        // Clean up is_picked (only the ≤5 entries we set)
        for (unsigned v : Qpicked) is_picked[v] = false;

        unsigned depth_Li = 0, width_Li = 0;
        unsigned wMin    = std::numeric_limits<unsigned>::max();
        const unsigned hMax = depth_s;
        unsigned e       = s;
        bool     restart = false;

        for (unsigned i : Qpicked) {
            bls_impl(i, ptr, ind, n, sc, depth_Li, width_Li, wMin);

            if (depth_Li > hMax && width_Li < wMin) {
                s = i; restart = true; break;
            }
            if (width_Li < wMin) { e = i; wMin = width_Li; }
        }

        if (restart) continue;

        // MGPS: choose narrower structure as starting node
        if (width_Li > width_s) {
            sOut = e; eOut = s;
        } else {
            sOut = s; eOut = e;
        }
        return;
    }
}

} // namespace

// ── MGPSSloanOrdering class method implementations ───────────────────────────

// buildLevelStructure: MGPS has its own static version (hides parent's).
// V2 uses scratch internally, exposing the same external API.
void MGPSSloanOrdering::buildLevelStructure(
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

bool MGPSSloanOrdering::isNeighborOfAnyPicked(
    unsigned v,
    const vType* ptr, const vType* ind,
    const std::vector<unsigned>& picked
) {
    // V1 fallback (only called externally if at all; hot path uses is_picked bool array)
    for (vType k = ptr[v]; k < ptr[v + 1]; ++k) {
        const unsigned u = static_cast<unsigned>(ind[k]);
        for (unsigned t : picked) if (u == t) return true;
    }
    return false;
}

void MGPSSloanOrdering::findPseudoPeripheralEndpointsMGPS(
    unsigned s,
    const vType* ptr, const vType* ind, unsigned n,
    const std::vector<unsigned>& degree,
    unsigned& sOut, unsigned& eOut
) {
    SloanScratch sc;
    sc.resize(n);
    std::vector<bool> is_picked(n, false);
    find_endpoints_mgps_impl(s, ptr, ind, n, degree, sc, is_picked, sOut, eOut);
}

void MGPSSloanOrdering::orderingFunction() {
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
    std::vector<bool> is_picked(n, false); // reused across components

    std::vector<NodeInfo> info;
    std::vector<unsigned> eligible;
    std::vector<unsigned> localLabels;

    unsigned seed_cursor = 0;

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

        unsigned s = seed, e = seed;
        find_endpoints_mgps_impl(seed, ptr, ind, n, degree, sc, is_picked, s, e);

        // Build L(e) for initializeNodeInfo
        unsigned depth_e, width_e;
        bls_impl(e, ptr, ind, n, sc, depth_e, width_e,
                 std::numeric_limits<unsigned>::max());

        LevelStructure Le;
        Le.depth = depth_e;
        Le.width = width_e;
        Le.level.assign(n, -1);
        for (unsigned v : sc.touched_level) Le.level[v] = sc.level[v];

        initializeNodeInfo(info, Le, n, degree, W1, W2);

        // Numbering resolves to SloanOrderingV2's heap-based implementation
        Numbering(s, ptr, ind, n, W1, info, eligible, localLabels);

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
