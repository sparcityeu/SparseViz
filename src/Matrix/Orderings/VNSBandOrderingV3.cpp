// Created by grkmshn on 11/20/25.

#include "VNSBandOrdering.h"
#include <vector>
#include <random>
#include <cstddef>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>

// ============================================================
//  Timing infrastructure
//  Set VNS_TIMING to 0 (or compile with -DVNS_TIMING=0) to
//  remove all overhead for production runs.
// ============================================================
#ifndef VNS_TIMING
#define VNS_TIMING 1
#endif

#if VNS_TIMING
    #define VTIME_START(tag)     auto _vt_##tag = std::chrono::steady_clock::now()
    #define VTIME_STOP(acc, tag) (acc) += std::chrono::steady_clock::now() - _vt_##tag
#else
    #define VTIME_START(tag)     ((void)0)
    #define VTIME_STOP(acc, tag) ((void)0)
#endif

using Dur = std::chrono::nanoseconds;

// All timing/counter state for one full VNS run
struct VNSTimings {
    // --- time buckets ---
    Dur initLabeling    {0};   // initialVNSBandLabeling
    Dur computeBW       {0};   // computeLocalAndGlobalBandwidth (outside IHC)
    Dur localSearch     {0};   // LocalSearchIHC total
    Dur shakeTotal      {0};   // entire shake() call

    // breakdown inside shake1
    Dur s1_shakeData    {0};   // fused computeShakeData (full pass, once per shake1 call)
    Dur s1_incrUpdate   {0};   // incremental update after each swap
    Dur s1_selectW      {0};   // selectW inside shake1

    // breakdown inside LocalSearchIHC
    Dur ihc_computeNP   {0};   // computeNPrime inside IHC
    Dur ihc_localBW     {0};   // localBandwidth calls inside IHC
    Dur ihc_fullBW      {0};   // full computeBandwidth inside IHC (rare)

    Dur moveCheck       {0};   // Move() function

    // --- counters ---
    unsigned long long vnsTotalIter    {0};  // VNS inner loop iterations
    unsigned long long tRestarts       {0};  // outer t-loop restarts
    unsigned long long moveAccepted    {0};
    unsigned long long moveRejected    {0};
    unsigned long long shake1Calls     {0};
    unsigned long long shake2Calls     {0};
    unsigned long long computeBWCalls  {0};  // outside IHC
    unsigned long long ihcCalls        {0};  // LocalSearchIHC invocations
    unsigned long long ihcOuterPasses  {0};  // work-list vertex visits (non-stale)
    unsigned long long ihcSwapAttempts {0};
    unsigned long long ihcSwapAccepted {0};
    unsigned long long ihcFullBWCalls  {0};

    void print(std::chrono::steady_clock::time_point wallStart) const {
        using ms = std::chrono::duration<double, std::milli>;

        auto toMs = [](Dur d) { return std::chrono::duration_cast<ms>(d).count(); };
        double wallMs = toMs(std::chrono::duration_cast<Dur>(
            std::chrono::steady_clock::now() - wallStart));

        auto pct = [&](Dur d) -> double {
            return wallMs > 0 ? (toMs(d) / wallMs) * 100.0 : 0.0;
        };

        std::cout << "\n========== VNS-Band V4 Timing Report ==========\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Wall time (total ordering)              : " << wallMs        << " ms\n\n";

        std::cout << "  --- Top-level time buckets ---\n";
        std::cout << "  initLabeling                            : " << toMs(initLabeling)
                  << " ms  (" << pct(initLabeling)   << " %)\n";
        std::cout << "  computeLocalAndGlobalBW (outside IHC)  : " << toMs(computeBW)
                  << " ms  (" << pct(computeBW)       << " %)  x" << computeBWCalls << "\n";
        std::cout << "  LocalSearchIHC (total)                 : " << toMs(localSearch)
                  << " ms  (" << pct(localSearch)     << " %)  x" << ihcCalls << "\n";
        std::cout << "  shake (total)                          : " << toMs(shakeTotal)
                  << " ms  (" << pct(shakeTotal)      << " %)\n";
        std::cout << "  Move()                                 : " << toMs(moveCheck)
                  << " ms  (" << pct(moveCheck)       << " %)\n\n";

        std::cout << "  --- Inside shake ---\n";
        std::cout << "    shake1 calls                         : " << shake1Calls << "\n";
        std::cout << "    shake2 calls                         : " << shake2Calls << "\n";
        std::cout << "    computeShakeData (full, once/call)   : " << toMs(s1_shakeData)
                  << " ms  (" << pct(s1_shakeData)   << " %)\n";
        std::cout << "    incremental update (after swap)      : " << toMs(s1_incrUpdate)
                  << " ms  (" << pct(s1_incrUpdate)  << " %)\n";
        std::cout << "    selectW                              : " << toMs(s1_selectW)
                  << " ms  (" << pct(s1_selectW)     << " %)\n\n";

        std::cout << "  --- Inside LocalSearchIHC ---\n";
        std::cout << "    outer while-loop passes              : " << ihcOuterPasses << "\n";
        std::cout << "    swap attempts                        : " << ihcSwapAttempts << "\n";
        std::cout << "    swap accepted                        : " << ihcSwapAccepted << "\n";
        if (ihcSwapAttempts > 0)
            std::cout << "    accept rate                          : "
                      << 100.0 * ihcSwapAccepted / ihcSwapAttempts << " %\n";
        std::cout << "    computeNPrime                        : " << toMs(ihc_computeNP)
                  << " ms  (" << pct(ihc_computeNP)  << " %)\n";
        std::cout << "    localBandwidth calls                 : " << toMs(ihc_localBW)
                  << " ms  (" << pct(ihc_localBW)    << " %)\n";
        std::cout << "    computeBandwidth (full, rare)        : " << toMs(ihc_fullBW)
                  << " ms  (" << pct(ihc_fullBW)     << " %)  x" << ihcFullBWCalls << "\n\n";

        std::cout << "  --- VNS loop counters ---\n";
        std::cout << "    t-restarts                           : " << tRestarts      << "\n";
        std::cout << "    VNS inner iterations                 : " << vnsTotalIter   << "\n";
        std::cout << "    Move accepted                        : " << moveAccepted   << "\n";
        std::cout << "    Move rejected                        : " << moveRejected   << "\n";
        std::cout << "================================================\n\n";
    }
};


//constants that should be changed to work with user input later
//these constants are currently the same as what was used in the original paper
constexpr unsigned KMIN        = 5;
constexpr unsigned KSTEP       = 5;
constexpr unsigned KPRIME_MAX  = 100;
constexpr unsigned K_MAX       = 200;

// compute bandwidth B(f)
static unsigned computeBandwidth(const std::vector<unsigned>& f,
                                 const unsigned* ptr,
                                 const unsigned* ind,
                                 unsigned n)
{
    unsigned B = 0;
    for (unsigned v = 0; v < n; ++v) {
        const unsigned fv = f[v];
        for (unsigned p = ptr[v]; p < ptr[v + 1]; ++p) {
            unsigned u = ind[p];
            if (u == v) continue;
            unsigned diff = std::abs(static_cast<int>(fv) - static_cast<int>(f[u]));
            if (diff > B) B = diff;
        }
    }
    return B;
}


static unsigned computeLocalAndGlobalBandwidth(
    const std::vector<unsigned>& f,
    const unsigned* ptr,
    const unsigned* ind,
    unsigned n,
    std::vector<unsigned>& B_local)
{
    unsigned B = 0;
    for (unsigned v = 0; v < n; ++v) {
        const unsigned fv = f[v];
        unsigned localMax = 0;
        for (unsigned p = ptr[v]; p < ptr[v + 1]; ++p) {
            unsigned u = ind[p];
            unsigned diff = std::abs(static_cast<int>(fv) - static_cast<int>(f[u]));
            if (diff > localMax) localMax = diff;
        }
        B_local[v] = localMax;
        if (localMax > B) B = localMax;
    }
    return B;
}


//takes an array of a current permutation and marks critical vertices
static void findCriticalVertices(const std::vector<unsigned>& f,
                                      const unsigned* ptr,
                                      const unsigned* ind,
                                      unsigned n,
                                      unsigned Bf,
                                      std::vector<char> &isCritical) {
    isCritical.assign(n, 0);
    for (unsigned v = 0; v < n; ++v) {
        const unsigned fv = f[v];
        for (unsigned p = ptr[v]; p < ptr[v + 1]; ++p) {
            unsigned u = ind[p];
            if (u == v) continue;
            unsigned diff = std::abs(static_cast<int>(fv) - static_cast<int>(f[u]));
            if (diff == Bf) {
                isCritical[v] = 1;
                isCritical[u] = 1;
            }
        }
    }
}

// |V_C(f)| = number of critical vertices under labeling f
static unsigned countCriticalVertices(const std::vector<unsigned>& f,
                                      const unsigned* ptr,
                                      const unsigned* ind,
                                      unsigned n,
                                      unsigned Bf,
                                      std::vector<char> &isCritical)
{
    findCriticalVertices(f, ptr, ind, n, Bf, isCritical);
    unsigned count = 0;
    for (char c : isCritical) count += static_cast<unsigned>(c != 0);
    return count;
}

// takes two permutations a and b, and calculates permutation difference
static unsigned hammingDistance(const std::vector<unsigned>& a,
                                const std::vector<unsigned>& b) {
    const unsigned n = static_cast<unsigned>(a.size());
    unsigned rho = 0;
    for (unsigned i = 0; i < n; ++i)
        if (a[i] != b[i]) ++rho;
    return rho;
}

// gives the initial labeling using randomized BFS with per-level rotation
std::vector<unsigned> initialVNSBandLabeling(
    const unsigned* ptr,
    const unsigned* ind,
    unsigned n,
    std::mt19937& rng)
{
    std::vector<unsigned> labels(n, 0);
    std::vector<bool> visited(n, false);


    // used for randomly choosing starting vertex, required for disconnected matrices
    auto pickRandomUnvisited = [&]() -> int {
        for (;;) {
            unsigned v = rng() % n;
            if (!visited[v]) return v;
        }
    };

    std::vector<unsigned> currentLevel;
    std::vector<unsigned> nextLevel;
    unsigned nextLabel = 0;

    while (nextLabel < n) {

        if (currentLevel.empty()) {
            int root = pickRandomUnvisited();
            visited[root] = true;
            currentLevel.push_back(root);
        }

        nextLevel.clear();
        for (unsigned v : currentLevel) {
            for (unsigned p = ptr[v]; p < ptr[v + 1]; ++p) {
                unsigned u = ind[p];
                if (u == v) continue;
                if (!visited[u]) {
                    visited[u] = true;
                    nextLevel.push_back(u);
                }
            }
        }

        const size_t currentCount = currentLevel.size();
        if (currentCount > 0) {
            // random rotation of current level
            std::uniform_int_distribution<size_t> distRot(0, currentCount - 1);
            size_t rotStart = distRot(rng);

            for (size_t k = 0; k < currentCount; ++k) {
                unsigned v = currentLevel[rotStart];
                labels[v] = nextLabel;
                ++nextLabel;

                ++rotStart;
                if (rotStart == currentCount) rotStart = 0;
            }
        }

        currentLevel.swap(nextLevel);
    }

    return labels;
}

// Finds local bandwidth
// B(f, v): local bandwidth of vertex v
// f[v]   : label (position) of vertex v
unsigned localBandwidth(
    const std::vector<unsigned>& f,
    const unsigned* ptr,
    const unsigned* ind,
    unsigned v
)
{
    unsigned lbl_v = f[v];
    unsigned maxDiff = 0;

    for (unsigned p = ptr[v]; p < ptr[v + 1]; ++p) {
        unsigned u = ind[p];
        if (u == v) continue;
        auto diff = static_cast<unsigned>(
            std::abs(static_cast<int>(f[u]) - static_cast<int>(lbl_v))
        );
        if (diff > maxDiff) maxDiff = diff;
    }

    return maxDiff;   // 0 if no neighbors or all same label
}

// Fused computation of B_local, fmin, fmax, K, and Bprime in a single O(m) pass.
// Replaces separate computeKandBprime + computeFMinFMax to halve adjacency traversals.
//
// All scratch buffers (B_local, sortedVals, fmin, fmax, K) are pre-allocated by
// the caller and reused across iterations to avoid heap allocation pressure.
void computeShakeData(
    const std::vector<unsigned>& f,
    const unsigned* ptr,
    const unsigned* ind,
    unsigned n,
    unsigned k,
    std::vector<unsigned>& K,
    std::vector<unsigned>& B_local,
    std::vector<unsigned>& sortedVals,
    std::vector<unsigned>& fmin,
    std::vector<unsigned>& fmax,
    unsigned& Bprime_out
)
{
    // --- 1. Single pass: compute B_local[v], fmin[v], fmax[v] for all vertices ---
    unsigned globalMax = 0;
    for (unsigned v = 0; v < n; ++v) {
        const unsigned fv = f[v];
        unsigned localMax = 0;
        bool first = true;
        unsigned mn = fv, mx = fv;

        for (unsigned p = ptr[v]; p < ptr[v + 1]; ++p) {
            unsigned u = ind[p];
            if (u == v) continue;
            unsigned fu = f[u];
            unsigned diff = std::abs(static_cast<int>(fv) - static_cast<int>(fu));
            if (diff > localMax) localMax = diff;
            if (first) {
                mn = fu; mx = fu; first = false;
            } else {
                if (fu < mn) mn = fu;
                if (fu > mx) mx = fu;
            }
        }
        B_local[v] = localMax;
        fmin[v] = mn;
        fmax[v] = mx;
        if (localMax > globalMax) globalMax = localMax;
    }

    if (globalMax == 0) {
        K.clear();
        Bprime_out = 0;
        return;
    }

    // --- 2. Compute B' as the k-th largest B_local value ---
    // Reuse sortedVals scratch buffer (copy B_local into it)
    std::copy(B_local.begin(), B_local.begin() + n, sortedVals.begin());
    unsigned idx = std::min<unsigned>(k - 1, n - 1);
    std::nth_element(sortedVals.begin(), sortedVals.begin() + idx, sortedVals.end(),
                     std::greater<unsigned>());
    unsigned Bprime = sortedVals[idx];

    // --- 3. Build K = { v | B_local[v] >= B' } ---
    K.clear();
    for (unsigned v = 0; v < n; ++v) {
        if (B_local[v] >= Bprime) {
            K.push_back(v);
        }
    }

    Bprime_out = Bprime;
}

// Selects w according to the Shake-1 criterion.
// cost(w) = max(|fmax[w]-lbl_v|, |lbl_v-fmin[w]|) — minimised over [fmin_u, fmax_u].
//
// Optimizations vs. the original linear scan:
//   1. Bidirectional outward scan from lbl_v: spatial locality after IHC means
//      candidates near lbl_v have low cost, so bestCost drops early.
//   2. c1-pruning: cost >= c1 = |fmax[w]-lbl_v|; skip fmin[w] load if c1 >= bestCost.
//   3. Early exit when bestCost == 0 (global minimum, cannot improve).
int selectW(
    unsigned u,
    unsigned v,
    const std::vector<unsigned>& f,
    const std::vector<unsigned>& fmin,
    const std::vector<unsigned>& fmax,
    const std::vector<unsigned>& pos
)
{
    const unsigned fmin_u = fmin[u];
    const unsigned fmax_u = fmax[u];

    if (fmin_u >= fmax_u)
        return -1;

    const unsigned lbl_v = f[v];

    // Clamp start to valid range — lbl_v may be slightly outside [fmin_u, fmax_u]
    // due to label staleness between incremental updates.
    const int range_lo = static_cast<int>(fmin_u);
    const int range_hi = static_cast<int>(fmax_u);
    int start = static_cast<int>(lbl_v);
    if (start < range_lo) start = range_lo;
    if (start > range_hi) start = range_hi;

    int bestW = -1;
    unsigned bestCost = std::numeric_limits<unsigned>::max();

    // lo scans downward from start; hi scans upward from start+1.
    // Uses signed int to avoid unsigned underflow when lo crosses zero.
    int lo = start;
    int hi = start + 1;

    while (lo >= range_lo || hi <= range_hi) {
        if (lo >= range_lo) {
            const unsigned w = pos[static_cast<unsigned>(lo)];
            if (w != u && w != v) {
                const unsigned c1 = fmax[w] > lbl_v ? fmax[w] - lbl_v : lbl_v - fmax[w];
                if (c1 < bestCost) {
                    const unsigned c2 = lbl_v > fmin[w] ? lbl_v - fmin[w] : fmin[w] - lbl_v;
                    if (c2 < bestCost) {
                        const unsigned cost = c1 > c2 ? c1 : c2;
                        if (cost < bestCost) {
                            bestCost = cost;
                            bestW    = static_cast<int>(w);
                            if (bestCost == 0) return bestW;
                        }
                    }
                }
            }
            --lo;
        }

        if (hi <= range_hi) {
            const unsigned w = pos[static_cast<unsigned>(hi)];
            if (w != u && w != v) {
                const unsigned c1 = fmax[w] > lbl_v ? fmax[w] - lbl_v : lbl_v - fmax[w];
                if (c1 < bestCost) {
                    const unsigned c2 = lbl_v > fmin[w] ? lbl_v - fmin[w] : fmin[w] - lbl_v;
                    if (c2 < bestCost) {
                        const unsigned cost = c1 > c2 ? c1 : c2;
                        if (cost < bestCost) {
                            bestCost = cost;
                            bestW    = static_cast<int>(w);
                            if (bestCost == 0) return bestW;
                        }
                    }
                }
            }
            ++hi;
        }
    }

    return bestW;
}

// actual Shake-1
void shake1(
    std::vector<unsigned>& f,
    const unsigned* ptr,
    const unsigned* ind,
    unsigned n,
    unsigned k,
    std::mt19937& rng,
    VNSTimings& T
)
{
    if (n == 0 || k == 0) return;

    T.shake1Calls++;

    // Pre-allocate all scratch buffers
    std::vector<unsigned> K;
    std::vector<unsigned> B_local(n, 0);
    std::vector<unsigned> sortedVals(n);
    std::vector<unsigned> fmin(n, 0);
    std::vector<unsigned> fmax(n, 0);
    std::vector<unsigned> maxCandidates;
    unsigned Bprime = 0;

    // pos[label] = vertex; updated O(1) after each swap
    std::vector<unsigned> pos(n);
    for (unsigned v = 0; v < n; ++v)
        pos[f[v]] = v;

    // --- Full O(m) pass once before the loop ---
    {
        VTIME_START(sd);
        computeShakeData(f, ptr, ind, n, k, K, B_local, sortedVals, fmin, fmax, Bprime);
        VTIME_STOP(T.s1_shakeData, sd);
    }

    // cnt[b] = number of vertices with B_local[v] == b; enables O(1) updates
    std::vector<unsigned> cnt(n, 0);
    for (unsigned v = 0; v < n; ++v)
        cnt[B_local[v]]++;

    // globalMax: highest b with cnt[b] > 0
    unsigned globalMax = 0;
    for (unsigned b = n; b-- > 0; )
        if (cnt[b] > 0) { globalMax = b; break; }

    // inK[x]=1 iff x is currently in K; posInK[x]=its index in K (valid when inK[x]=1).
    // Maintained incrementally so K can be updated in O(|affected|) when Bprime is stable.
    std::vector<char> inK(n, 0);
    std::vector<unsigned> posInK(n, 0);
    for (unsigned i = 0; i < static_cast<unsigned>(K.size()); ++i) {
        inK[K[i]] = 1;
        posInK[K[i]] = i;
    }

    // visited[]: deduplicates the affected set; reset only for touched entries
    std::vector<char> visited(n, 0);
    std::vector<unsigned> affected;
    affected.reserve(256);

    // processVertex: rescan x's neighbors to update B_local/fmin/fmax and cnt
    auto processVertex = [&](unsigned x) {
        if (visited[x]) return;
        visited[x] = 1;
        affected.push_back(x);

        const unsigned old_b = B_local[x];
        const unsigned fx    = f[x];
        unsigned localMax = 0;
        bool first = true;
        unsigned mn = fx, mx = fx;

        for (unsigned p = ptr[x]; p < ptr[x + 1]; ++p) {
            unsigned nb = ind[p];
            if (nb == x) continue;
            const unsigned fnb  = f[nb];
            const unsigned diff = (fx > fnb) ? (fx - fnb) : (fnb - fx);
            if (diff > localMax) localMax = diff;
            if (first) { mn = fnb; mx = fnb; first = false; }
            else { if (fnb < mn) mn = fnb; if (fnb > mx) mx = fnb; }
        }

        B_local[x] = localMax;
        fmin[x]    = mn;
        fmax[x]    = mx;

        if (old_b != localMax) {
            cnt[old_b]--;
            cnt[localMax]++;
        }
    };

    for (unsigned iter = 0; iter < k; ++iter) {

        if (K.empty() || Bprime == 0) break;

        // --- 1. Pick u uniformly at random from K ---
        std::uniform_int_distribution<unsigned> distU(0, static_cast<unsigned>(K.size() - 1));
        unsigned u = K[distU(rng)];

        // --- 2. Find v: worst neighbor of u ---
        unsigned lbl_u = f[u];
        maxCandidates.clear();
        unsigned maxDiffUV = 0;

        for (unsigned p = ptr[u]; p < ptr[u + 1]; ++p) {
            unsigned cand = ind[p];
            if (cand == u) continue;
            unsigned diff = (f[cand] > lbl_u) ? (f[cand] - lbl_u) : (lbl_u - f[cand]);
            if (diff > maxDiffUV) {
                maxDiffUV = diff;
                maxCandidates.clear();
                maxCandidates.push_back(cand);
            } else if (diff == maxDiffUV) {
                maxCandidates.push_back(cand);
            }
        }

        if (maxCandidates.empty() || maxDiffUV == 0) continue;

        unsigned v;
        if (maxCandidates.size() == 1) {
            v = maxCandidates[0];
        } else {
            std::uniform_int_distribution<unsigned> distV(0, static_cast<unsigned>(maxCandidates.size() - 1));
            v = maxCandidates[distV(rng)];
        }

        // --- 3. Choose w ---
        int bestW;
        {
            VTIME_START(sw);
            bestW = selectW(u, v, f, fmin, fmax, pos);
            VTIME_STOP(T.s1_selectW, sw);
        }
        if (bestW < 0) continue; // no swap — data still valid for next iteration

        // --- 4. Swap labels of v and w; update pos ---
        unsigned w = static_cast<unsigned>(bestW);
        std::swap(f[v], f[w]);
        pos[f[v]] = v;
        pos[f[w]] = w;

        // --- 5. Incremental update of B_local/fmin/fmax/cnt/Bprime/K ---
        // Only vertices in {v,w} ∪ N(v) ∪ N(w) have changed local data.
        // O(Σ deg(x) for x in affected) + O(n) for K rebuild — replaces O(m) full pass.
        {
            VTIME_START(iu);

            // Rescan v, neighbors of v, w, neighbors of w (deduplicated via visited[])
            processVertex(v);
            for (unsigned p = ptr[v]; p < ptr[v + 1]; ++p) {
                unsigned nb = ind[p];
                if (nb != v) processVertex(nb);
            }
            processVertex(w);
            for (unsigned p = ptr[w]; p < ptr[w + 1]; ++p) {
                unsigned nb = ind[p];
                if (nb != w) processVertex(nb);
            }

            // Reset visited flags; keep `affected` alive until after K update
            for (unsigned x : affected) visited[x] = 0;

            // Update globalMax: scan down until a non-zero cnt bucket
            while (globalMax > 0 && cnt[globalMax] == 0) --globalMax;

            // Recompute Bprime: k-th largest B_local (scan cnt from top)
            const unsigned oldBprime = Bprime;
            if (globalMax == 0) {
                Bprime = 0;
            } else {
                const unsigned idx = std::min<unsigned>(k - 1, n - 1);
                unsigned cumsum = 0;
                Bprime = 0;
                for (unsigned b = globalMax; ; --b) {
                    cumsum += cnt[b];
                    if (cumsum > idx) { Bprime = b; break; }
                    if (b == 0) break;
                }
            }

            // Update K
            if (Bprime == oldBprime) {
                // Bprime unchanged: only vertices in `affected` can have changed
                // membership. Update K in O(|affected|) using inK/posInK.
                for (unsigned x : affected) {
                    const bool shouldBeIn = (B_local[x] >= Bprime);
                    if (shouldBeIn && !inK[x]) {
                        posInK[x] = static_cast<unsigned>(K.size());
                        K.push_back(x);
                        inK[x] = 1;
                    } else if (!shouldBeIn && inK[x]) {
                        // O(1) swap-erase
                        const unsigned pos_x = posInK[x];
                        const unsigned last  = K.back();
                        K[pos_x]    = last;
                        posInK[last] = pos_x;
                        K.pop_back();
                        inK[x] = 0;
                    }
                }
            } else {
                // Bprime changed: full O(n) rebuild required
                K.clear();
                std::fill(inK.begin(), inK.end(), 0);
                for (unsigned x = 0; x < n; ++x) {
                    if (B_local[x] >= Bprime) {
                        posInK[x] = static_cast<unsigned>(K.size());
                        K.push_back(x);
                        inK[x] = 1;
                    }
                }
            }

            affected.clear();

            VTIME_STOP(T.s1_incrUpdate, iu);
        }
    }
}

// `Shake` step 2: randomly rotates an interval repeated k times
void shake2(
    std::vector<unsigned>& f,
    unsigned Bf,
    unsigned k,
    std::mt19937& rng,
    VNSTimings& T
) {
    T.shake2Calls++;
    if (k==0) return;
    const unsigned n = f.size();
    if (n < 3) return;

    // Build inverse permutation p[label] = vertex
    std::vector<unsigned> p(n);
    for (unsigned v = 0; v < n; ++v)
        p[f[v]] = v;
    for (unsigned iter = 0; iter < k; ++iter)
    {
        unsigned beginning = std::uniform_int_distribution<unsigned>(0, n - 1)(rng);

        // guard against too small Bf so upper bound >= 2
        unsigned maxLen = (Bf > 3 ? std::min(20u, Bf / 2) : 2u);
        if (maxLen < 2) continue;

        unsigned ending = beginning + std::uniform_int_distribution<unsigned>(2, maxLen)(rng);

        if (ending > n - 1)
            ending = n - 1;

        if (ending <= beginning + 1)
            continue; // skip this iteration

        unsigned innerLen = ending - beginning - 1;
        if (innerLen == 0) continue;

        unsigned middle = std::uniform_int_distribution<unsigned>(1, innerLen)(rng);

        // Rotate labels in-place and do partial pos update
        for (unsigned j = beginning; j <= ending; ++j) {
            unsigned vertex = p[j];
            if (j >= beginning + middle) {
                f[vertex] = j - middle;
            } else {
                f[vertex] = j + ending - beginning - middle + 1;
            }
        }

        // OPT: partial pos update — only fix entries for affected range
        for (unsigned j = beginning; j <= ending; ++j) {
            unsigned vertex = p[j];       // vertex that had label j
            p[f[vertex]] = vertex;        // update its new position
        }
    }
}

// SHAKE PROPER
void shake(
    std::vector<unsigned>& f,
    const unsigned* ptr,
    const unsigned* ind,
    unsigned n,
    unsigned k,
    unsigned k_min,
    unsigned k_step,
    unsigned kprime_max,
    unsigned Bf,
    std::mt19937& rng,
    VNSTimings& T
) {
    if (k <= kprime_max) {
        shake1(f, ptr, ind, n, k, rng, T);
    } else {
        unsigned kprime = (k - k_min) / k_step;
        shake2(f, Bf, kprime, rng, T);
    }
}

// Move or not function: decide if fPrime is "better" than f
bool Move(const std::vector<unsigned>& f,
          const std::vector<unsigned>& fPrime,
          const unsigned* ptr,
          const unsigned* ind,
          unsigned n,
          unsigned alpha,
          const unsigned& Bf,
          const unsigned& Bfp)
{
    if (Bfp < Bf) return true;

    if (Bfp == Bf) {

        const unsigned rho = hammingDistance(f, fPrime);
        if (rho > alpha) return true;

        std::vector<char> isCriticalF(n,0);
        std::vector<char> isCriticalFprime(n,0);
        const unsigned VCf  = countCriticalVertices(f,      ptr, ind, n, Bf, isCriticalF);
        const unsigned VCfp = countCriticalVertices(fPrime, ptr, ind, n, Bfp, isCriticalFprime);

        if (VCfp < VCf) return true;
    }

    return false;
}


void computeNPrime(
    const std::vector<unsigned>& f,
    const unsigned* ptr,
    const unsigned* ind,
    unsigned n,
    unsigned v,
    std::vector<unsigned>& nprime
) {
    nprime.clear();
    if (n == 0 || v >= n) return;

    unsigned mn = f[v], mx = f[v];
    bool first = true;
    for (unsigned p = ptr[v]; p < ptr[v + 1]; ++p) {
        unsigned u = ind[p];
        if (u == v) continue;
        if (first) { mn = f[u]; mx = f[u]; first = false; }
        else {
            if (f[u] < mn) mn = f[u];
            if (f[u] > mx) mx = f[u];
        }
    }

    unsigned mid = (mn + mx) / 2;
    unsigned dv = (mid > f[v]) ? (mid - f[v]) : (f[v] - mid);

    for (unsigned p = ptr[v]; p < ptr[v + 1]; ++p) {
        unsigned u = ind[p];
        if (u == v) continue;
        unsigned du = (mid > f[u]) ? (mid - f[u]) : (f[u] - mid);
        if (du < dv) {
            nprime.push_back(u);
        }
    }

    std::sort(nprime.begin(), nprime.end(),
          [&](unsigned a, unsigned b) {
              unsigned da = (mid > f[a]) ? (mid - f[a]) : (f[a] - mid);
              unsigned db = (mid > f[b]) ? (mid - f[b]) : (f[b] - mid);
              return da < db;
          });
}


void LocalSearchIHC(std::vector<unsigned>& f, unsigned n, unsigned& Bf,
                    const unsigned* ptr, const unsigned* ind,
                    std::vector<unsigned>& B_local,
                    VNSTimings& T)
{
    T.ihcCalls++;
    bool canImprove = true;
    std::vector<char> criticalVertices;
    findCriticalVertices(f, ptr, ind, n, Bf, criticalVertices);
    unsigned criticalCount = 0;
    for (char c : criticalVertices) criticalCount += static_cast<unsigned>(c != 0);
    std::vector<unsigned> nPrimeV;  // pre-allocate outside loops

    // cnt[b] = number of vertices with B_local[v] == b; lets us find max(B_local) without
    // a full O(m) edge scan — same technique as shake1's incremental update.
    std::vector<unsigned> cnt(n, 0);
    for (unsigned w = 0; w < n; ++w)
        cnt[B_local[w]]++;
    unsigned globalBW = Bf;  // tracks max(B_local), equals Bf initially

    while (canImprove) {
        canImprove = false;
        T.ihcOuterPasses++;
        for (unsigned v = 0; v < n; ++v)
        {
            if (criticalVertices[v] != 1) continue;

            {
                VTIME_START(cnp);
                computeNPrime(f, ptr, ind, n, v, nPrimeV);
                VTIME_STOP(T.ihc_computeNP, cnp);
            }
            if (nPrimeV.empty()) continue;

            for (unsigned i = 0; i < nPrimeV.size(); ++i)
            {
                T.ihcSwapAttempts++;
                unsigned u = nPrimeV[i];

                // tentatively swap in-place
                std::swap(f[v], f[u]);

                unsigned newBW_v, newBW_u;
                {
                    VTIME_START(lbw);
                    newBW_v = localBandwidth(f, ptr, ind, v);
                    VTIME_STOP(T.ihc_localBW, lbw);
                }
                if (newBW_v > Bf) { std::swap(f[v], f[u]); continue; }

                {
                    VTIME_START(lbw2);
                    newBW_u = localBandwidth(f, ptr, ind, u);
                    VTIME_STOP(T.ihc_localBW, lbw2);
                }
                if (newBW_u > Bf) { std::swap(f[v], f[u]); continue; }

                // Theorem 1: compute C' values
                // C'(x) = 2 if local BW > Bf (already rejected above)
                // C'(x) = 1 if local BW == Bf
                // C'(x) = 0 if local BW < Bf
                unsigned Cprime_v = (newBW_v == Bf) ? 1 : 0;
                unsigned Cprime_u = (newBW_u == Bf) ? 1 : 0;
                unsigned C_v = static_cast<unsigned char>(criticalVertices[v]);
                unsigned C_u = static_cast<unsigned char>(criticalVertices[u]);

                if (Cprime_u <= C_u &&
                    Cprime_v <= C_v &&
                    Cprime_u + Cprime_v < C_u + C_v)
                {
                    T.ihcSwapAccepted++;
                    // accepted — update Bf and critical vertices

                    // update B_local for affected vertices; keep cnt in sync
                    cnt[B_local[v]]--; B_local[v] = newBW_v; cnt[newBW_v]++;
                    cnt[B_local[u]]--; B_local[u] = newBW_u; cnt[newBW_u]++;
                    for (unsigned p = ptr[v]; p < ptr[v+1]; ++p) {
                        unsigned w = ind[p];
                        if (w != v) {
                            VTIME_START(lbwN);
                            unsigned newB = localBandwidth(f, ptr, ind, w);
                            VTIME_STOP(T.ihc_localBW, lbwN);
                            if (B_local[w] != newB) { cnt[B_local[w]]--; cnt[newB]++; }
                            B_local[w] = newB;
                        }
                    }
                    for (unsigned p = ptr[u]; p < ptr[u+1]; ++p) {
                        unsigned w = ind[p];
                        if (w != u) {
                            VTIME_START(lbwN2);
                            unsigned newB = localBandwidth(f, ptr, ind, w);
                            VTIME_STOP(T.ihc_localBW, lbwN2);
                            if (B_local[w] != newB) { cnt[B_local[w]]--; cnt[newB]++; }
                            B_local[w] = newB;
                        }
                    }
                    // advance globalBW down to the new max(B_local)
                    while (globalBW > 0 && cnt[globalBW] == 0) --globalBW;

                    // check if bandwidth actually decreased
                    if (Cprime_v == 0 && Cprime_u == 0 && criticalCount == 2) {
                        T.ihcFullBWCalls++;
                        VTIME_START(fbw);
                        // globalBW = max(B_local) maintained by cnt — no O(m) edge scan needed
                        unsigned newBf = globalBW;
                        VTIME_STOP(T.ihc_fullBW, fbw);
                        if (newBf < Bf) {
                            Bf = newBf;  // globalBW already equals newBf from the scan above
                            // Bf decreased: vertices far from v,u may be newly critical.
                            // Rebuild criticalVertices and criticalCount from stored B_local
                            // (O(n), no edge traversal — B_local is correct for all vertices).
                            criticalCount = 0;
                            for (unsigned w = 0; w < n; ++w) {
                                criticalVertices[w] = (B_local[w] == Bf) ? 1 : 0;
                                criticalCount += criticalVertices[w];
                            }
                            canImprove = true;
                            break;
                        }
                    }

                    // update criticalVertices and criticalCount for v, u, and their neighbors
                    auto updateCrit = [&](unsigned w) {
                        char newCrit = (B_local[w] == Bf) ? 1 : 0;
                        if (newCrit != criticalVertices[w]) {
                            criticalCount += newCrit - criticalVertices[w];
                            criticalVertices[w] = newCrit;
                        }
                    };
                    for (unsigned w : {v, u}) {
                        updateCrit(w);
                        for (unsigned p = ptr[w]; p < ptr[w+1]; ++p) {
                            unsigned x = ind[p];
                            if (x == w) continue;
                            updateCrit(x);
                        }
                    }
                    canImprove = true;
                    break;
                }
                // rejected. swap back
                std::swap(f[v], f[u]);
            }
        }
    }
    Bf = globalBW;
}


// VNS-Band main loop
void vnsBand(
    const unsigned* ptr,
    const unsigned* ind,
    unsigned n,
    unsigned t_max,
    unsigned alpha,
    std::vector<unsigned>& bestF,
    std::mt19937& rng,
    unsigned long long& iCount,
    std::chrono::steady_clock::time_point deadline
) {
    if (n == 0) return;

    VNSTimings T;
    auto wallStart = std::chrono::steady_clock::now();

    // best bandwidth seen so far: B*
    unsigned B_star = std::numeric_limits<unsigned>::max();

    const unsigned I_MAX = (K_MAX - KMIN) / KSTEP;
    while (std::chrono::steady_clock::now() < deadline)
    {
        for (unsigned t = 0; t < t_max; ++t) {
            T.tRestarts++;

            std::vector<unsigned> f;
            {
                VTIME_START(init);
                f = initialVNSBandLabeling(ptr, ind, n, rng);
                VTIME_STOP(T.initLabeling, init);
            }

            std::vector<unsigned> B_local(n, 0);
            unsigned Bf;
            {
                VTIME_START(bw0);
                Bf = computeLocalAndGlobalBandwidth(f, ptr, ind, n, B_local);
                VTIME_STOP(T.computeBW, bw0);
                T.computeBWCalls++;
            }

            {
                VTIME_START(ihc0);
                LocalSearchIHC(f, n, Bf, ptr, ind, B_local, T);
                VTIME_STOP(T.localSearch, ihc0);
            }

            unsigned i = 0;
            unsigned k = KMIN;

            // OPT: pre-allocate fPrime and Bf_prime_local outside the inner loop
            // to avoid repeated O(n) heap allocations per iteration.
            std::vector<unsigned> fPrime(n);
            std::vector<unsigned> Bf_prime_local(n);

            while (i <= I_MAX && std::chrono::steady_clock::now() < deadline) {
                T.vnsTotalIter++;
                iCount++;
                if (iCount % 100 == 0)
                    std::cout << "VNS V3 iteration i= " << iCount << std::endl;

                // reuse pre-allocated buffers
                fPrime.assign(f.begin(), f.end());
                {
                    VTIME_START(shk);
                    shake(fPrime, ptr, ind, n, k, KMIN, KSTEP, KPRIME_MAX, Bf, rng, T);
                    VTIME_STOP(T.shakeTotal, shk);
                }

                unsigned Bf_prime;
                {
                    VTIME_START(bw1);
                    Bf_prime = computeLocalAndGlobalBandwidth(fPrime, ptr, ind, n, Bf_prime_local);
                    VTIME_STOP(T.computeBW, bw1);
                    T.computeBWCalls++;
                }

                {
                    VTIME_START(ihc1);
                    LocalSearchIHC(fPrime, n, Bf_prime, ptr, ind, Bf_prime_local, T);
                    VTIME_STOP(T.localSearch, ihc1);
                }

                bool moved;
                {
                    VTIME_START(mv);
                    moved = Move(f, fPrime, ptr, ind, n, alpha, Bf, Bf_prime);
                    VTIME_STOP(T.moveCheck, mv);
                }

                if (moved) {
                    T.moveAccepted++;
                    f.swap(fPrime);
                    Bf = Bf_prime;
                    k = KMIN;
                    i = 0;
                } else {
                    T.moveRejected++;
                    k += KSTEP;
                    ++i;
                }
            }

            if (Bf < B_star) {
                B_star = Bf;
                bestF = f;
            }
        }
    }

    T.print(wallStart);
}


void VNSBandOrdering::orderingFunction() {

    const SparseMatrix& M = this->getMatrix();
    const unsigned nr = M.getRowCount();
    const unsigned nc = M.getColCount();

    if (nr == 0) return;

    const unsigned* ptrRaw = M.getPtr();
    const unsigned* indRaw = M.getInd();

    std::vector<unsigned> bestF;

    // Later user input
    const unsigned T_MAX_ITERS = 20;
    const unsigned ALPHA       = 10;
    thread_local std::mt19937 rng(std::random_device{}());
    auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(3);

    unsigned long long iCount = 0;

    vnsBand(ptrRaw, indRaw, nr, T_MAX_ITERS, ALPHA, bestF, rng, iCount, deadline);

    rowIPermutation = new vType[nr];
    for (unsigned v = 0; v < nr; ++v) {
        rowIPermutation[v] = bestF[v];
    }

    colIPermutation = new vType[nc];
    if (nr == nc) {
        for (unsigned v = 0; v < nc; ++v)
            colIPermutation[v] = rowIPermutation[v];
    } else {
        for (unsigned j = 0; j < nc; ++j)
            colIPermutation[j] = j;
    }
}
