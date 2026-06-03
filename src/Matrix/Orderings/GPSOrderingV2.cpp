
#include "GPSOrdering.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <utility>
#include <vector>

namespace {

// ---------- scratch memory (allocated once per orderingFunction() call) ----------
struct GPSScratch {
    unsigned nr = 0;

    // BFS queue backing store (capacity nr). Reused by every BFS in every phase.
    std::vector<unsigned> bfs_q;

    // Three level-index arrays. During Alg I the roles are:
    //   lv_level : currently held Lv (persistent through Alg II)
    //   lu_level : final Lu (persistent through Alg II)
    //   tmp_level: transient scratch for step-C exploratory BFSes (reset after each)
    // During Alg II / III, tmp_level is unused.
    std::vector<int> lv_level;
    std::vector<int> lu_level;
    std::vector<int> tmp_level;

    // Parallel touched-vertex lists for the three level arrays. Enables per-
    // component resets proportional to component size (not nr).
    std::vector<unsigned> touched_lv;
    std::vector<unsigned> touched_lu;
    std::vector<unsigned> touched_tmp;

    // Alg I working buffer: last-level vertex set of Lv, sorted by (deg, id).
    std::vector<unsigned> lastLevel;

    // Alg II scratch.
    std::vector<int>      i_of;             // i(w) from GPS paper; -1 outside component
    std::vector<int>      j_of;             // j(w) = k+1 - level_in_Lu(w)
    std::vector<char>     placed;           // 1 if w placed on diagonal (i==j)
    std::vector<char>     rem;              // 1 if w remaining (in_comp && !placed)
    std::vector<unsigned> touched_ij;       // vertices with i_of/j_of/placed/rem set

    std::vector<int>      counts;           // counts[r] = current |N[r]|
    std::vector<unsigned> counts_touched;   // levels with nonzero counts[] (for reset)

    std::vector<int>      Hbuf;             // per-subcomponent H[r]
    std::vector<int>      Lbuf;             // per-subcomponent L[r]

    // Alg II inner CC (connected components of `rem`).
    std::vector<unsigned> cc_mark;          // cc_mark[v] = subcomp index, UINT_MAX if unset
    std::vector<unsigned> touched_cc;       // vertices with cc_mark != UINT_MAX
    std::vector<unsigned> cc_verts;         // flat concatenated subcomp vertex lists
    std::vector<unsigned> cc_start;         // CSR-style starts, size num_subcomps+1
    std::vector<unsigned> cc_seeds;         // cc_seeds[k] = first vertex added to subcomp k
    std::vector<unsigned> cc_order;         // processing order: sort by (size desc, seed asc)

    // Alg III.
    std::vector<int>      final_level;      // N-level of each component vertex
    std::vector<unsigned> touched_final;
    std::vector<char>     labeled;          // 1 if vertex is in `order` (replaces std::find)
    std::vector<unsigned> touched_labeled;
    std::vector<unsigned> cand;             // candidate neighbors (reused per parent)
    std::vector<unsigned> order;            // numbered vertices for current component
    std::vector<unsigned> level_start_in_order; // level_start[i] = index where level i begins

    // Combined N: flat CSR layout derived from final_level via counting sort.
    std::vector<unsigned> N_off;
    std::vector<unsigned> N_flat;
    std::vector<unsigned> N_cursor;

    // Outer CC decomposition (one pass over the whole graph up front).
    std::vector<unsigned> comp_id;          // comp_id[v] = outer-component index
    std::vector<unsigned> comp_start;       // CSR starts
    std::vector<unsigned> comp_verts;       // vertices per comp, sorted by ascending id
    std::vector<unsigned> outer_cursor;     // scatter cursor during comp_verts construction

    // Global output.
    std::vector<unsigned> total_order;

    void resize(unsigned n) {
        nr = n;
        bfs_q.resize(n);

        lv_level.assign(n, -1);
        lu_level.assign(n, -1);
        tmp_level.assign(n, -1);
        touched_lv.reserve(n);
        touched_lu.reserve(n);
        touched_tmp.reserve(n);

        lastLevel.reserve(n);

        i_of.assign(n, -1);
        j_of.assign(n, -1);
        placed.assign(n, 0);
        rem.assign(n, 0);
        touched_ij.reserve(n);

        counts.assign(n + 2, 0);
        counts_touched.reserve(64);
        Hbuf.assign(n + 2, 0);
        Lbuf.assign(n + 2, 0);

        cc_mark.assign(n, UINT_MAX);
        touched_cc.reserve(n);
        cc_verts.reserve(n);
        cc_start.reserve(64);
        cc_seeds.reserve(64);
        cc_order.reserve(64);

        final_level.assign(n, -1);
        touched_final.reserve(n);
        labeled.assign(n, 0);
        touched_labeled.reserve(n);
        cand.reserve(128);
        order.reserve(n);
        level_start_in_order.reserve(64);

        N_off.reserve(64);
        N_flat.resize(n);
        N_cursor.reserve(64);

        comp_id.assign(n, UINT_MAX);
        comp_start.reserve(64);
        comp_verts.resize(n);
        outer_cursor.reserve(64);

        total_order.reserve(n);
    }
};

unsigned gps_deg(const unsigned* __restrict ptr, unsigned v) {
    return ptr[v + 1] - ptr[v];
}

// BFS from `root`, filling `level[]` (which must be all -1 for reached vertices
// beforehand; caller's touched-list reset handles this). Records touched
// vertices. Returns (depth, width) computed inline via level-wave traversal.
std::pair<int, unsigned> gps_bfs(
    unsigned root,
    const unsigned* __restrict ptr,
    const unsigned* __restrict ind,
    int* __restrict level,
    unsigned* __restrict bfs_q,
    std::vector<unsigned>& touched)
{
    unsigned qs = 0, qe = 0;
    level[root] = 1;
    touched.push_back(root);
    bfs_q[qe++] = root;
    int depth = 1;
    unsigned width = 1;
    while (qs < qe) {
        const unsigned level_end = qe;
        const unsigned cur_width = level_end - qs;
        if (cur_width > width) width = cur_width;
        const int cur_lvl = level[bfs_q[qs]];
        if (cur_lvl > depth) depth = cur_lvl;
        while (qs < level_end) {
            const unsigned u = bfs_q[qs++];
            for (unsigned p = ptr[u]; p < ptr[u + 1]; ++p) {
                const unsigned v = ind[p];
                if (level[v] == -1) {
                    level[v] = cur_lvl + 1;
                    touched.push_back(v);
                    bfs_q[qe++] = v;
                }
            }
        }
    }
    return {depth, width};
}

// Reset a level array to -1 at every touched entry, and clear the touched list.
void gps_reset_level(int* level, std::vector<unsigned>& touched) {
    for (unsigned v : touched) level[v] = -1;
    touched.clear();
}

// Sort a candidate list by (degree asc, id asc). V1's `by_deg`.
void gps_sort_by_deg(std::vector<unsigned>& xs,
                                   const unsigned* __restrict ptr) {
    std::ranges::sort(xs.begin(), xs.end(), [ptr](unsigned const a, unsigned const b) {
        const unsigned da = gps_deg(ptr, a);
        const unsigned db = gps_deg(ptr, b);
        if (da != db) return da < db;
        return a < b;
    });
}

} // namespace


void GPSOrdering::orderingFunction() {
    if (rowIPermutation || colIPermutation) return;

    const unsigned nr = getMatrix().getRowCount();
    const unsigned nc = getMatrix().getColCount();
    const unsigned* __restrict ptr = getMatrix().getPtr();
    const unsigned* __restrict ind = getMatrix().getInd();

    if (nr == 0) {
        rowIPermutation = new vType[0];
        colIPermutation = new vType[nc];
        for (unsigned j = 0; j < nc; ++j) colIPermutation[j] = static_cast<vType>(j);
        return;
    }

    GPSScratch sc;
    sc.resize(nr);

    // ------------------------------------------------------------------
    // Outer CC decomposition: label every vertex with its component id via
    // BFS from each unassigned seed in ascending id. Then build comp_verts
    // sorted by ascending id within each component.
    // ------------------------------------------------------------------
    unsigned num_comps = 0;
    for (unsigned s = 0; s < nr; ++s) {
        if (sc.comp_id[s] != UINT_MAX) continue;
        unsigned qs = 0, qe = 0;
        sc.bfs_q[qe++] = s;
        sc.comp_id[s] = num_comps;
        while (qs < qe) {
            const unsigned u = sc.bfs_q[qs++];
            for (unsigned p = ptr[u]; p < ptr[u + 1]; ++p) {
                const unsigned v = ind[p];
                if (sc.comp_id[v] == UINT_MAX) {
                    sc.comp_id[v] = num_comps;
                    sc.bfs_q[qe++] = v;
                }
            }
        }
        ++num_comps;
    }

    sc.comp_start.assign(num_comps + 1, 0);
    for (unsigned v = 0; v < nr; ++v) sc.comp_start[sc.comp_id[v] + 1]++;
    for (unsigned c = 1; c <= num_comps; ++c) sc.comp_start[c] += sc.comp_start[c - 1];

    sc.outer_cursor.assign(sc.comp_start.begin(), sc.comp_start.end());
    for (unsigned v = 0; v < nr; ++v) {
        const unsigned cid = sc.comp_id[v];
        sc.comp_verts[sc.outer_cursor[cid]++] = v;
    }

    // ------------------------------------------------------------------
    // Process each component in ascending-seed-id order (matches V1's
    // for-s-in-0..nr outer loop visit order).
    // ------------------------------------------------------------------
    for (unsigned cid = 0; cid < num_comps; ++cid) {
        const unsigned comp_begin = sc.comp_start[cid];
        const unsigned comp_end   = sc.comp_start[cid + 1];
        const unsigned n_comp     = comp_end - comp_begin;

        // Isolated / singleton component -> append seed directly (V1 fast path).
        if (n_comp == 1) {
            sc.total_order.push_back(sc.comp_verts[comp_begin]);
            continue;
        }

        // --------------------------------------------------------------
        // Algorithm I: pseudo-diameter endpoints.
        // --------------------------------------------------------------

        // A. v := min-degree vertex in component (iterated in ascending id;
        unsigned v_cur = sc.comp_verts[comp_begin];
        unsigned v_deg = gps_deg(ptr, v_cur);
        for (unsigned i = comp_begin + 1; i < comp_end; ++i) {
            const unsigned c = sc.comp_verts[i];
            const unsigned d = gps_deg(ptr, c);
            if (d < v_deg) { v_deg = d; v_cur = c; }
        }

        // B. Build Lv.
        auto Lv_dw = gps_bfs(v_cur, ptr, ind,
                             sc.lv_level.data(), sc.bfs_q.data(), sc.touched_lv);
        int      Lv_depth = Lv_dw.first;
        unsigned Lv_width = Lv_dw.second;

        // C/D fused: iterate sorted S once tracking (min_width, best_u). If any
        // s improves depth, promote and restart outer. Otherwise one final BFS
        // from best_u materializes Lu.
        unsigned u_best = v_cur;
        int      Lu_depth = Lv_depth;
        unsigned Lu_width = Lv_width;
        while (true) {
            // Build S from touched_lv, filtered to last level.
            sc.lastLevel.clear();
            for (unsigned w : sc.touched_lv) {
                if (sc.lv_level[w] == Lv_depth) sc.lastLevel.push_back(w);
            }
            gps_sort_by_deg(sc.lastLevel, ptr);

            unsigned best_u_local = sc.lastLevel.empty() ? v_cur : sc.lastLevel[0];
            unsigned best_w_local = UINT_MAX;
            bool progressed = false;
            for (unsigned s : sc.lastLevel) {
                auto dw = gps_bfs(s, ptr, ind,
                                  sc.tmp_level.data(), sc.bfs_q.data(),
                                  sc.touched_tmp);
                const int      s_depth = dw.first;
                const unsigned s_width = dw.second;
                if (s_depth > Lv_depth) {
                    // Promote s to the new v. Swap tmp <-> lv storage so the
                    // new Lv data lives in lv_level without an O(n) copy.
                    gps_reset_level(sc.lv_level.data(), sc.touched_lv);
                    sc.lv_level.swap(sc.tmp_level);
                    sc.touched_lv.swap(sc.touched_tmp);
                    v_cur    = s;
                    Lv_depth = s_depth;
                    Lv_width = s_width;
                    progressed = true;
                    break;
                }
                if (s_width < best_w_local) {
                    best_w_local = s_width;
                    best_u_local = s;
                }
                gps_reset_level(sc.tmp_level.data(), sc.touched_tmp);
            }
            if (!progressed) {
                u_best = best_u_local;
                auto dw = gps_bfs(u_best, ptr, ind,
                                  sc.lu_level.data(), sc.bfs_q.data(),
                                  sc.touched_lu);
                Lu_depth = dw.first;
                Lu_width = dw.second;
                break;
            }
        }

        const unsigned u_vert = u_best;
        const unsigned v_vert = v_cur;

        // --------------------------------------------------------------
        // Algorithm II: combine Lv and Lu into a single level structure N.
        // --------------------------------------------------------------
        const int k_total = std::max(Lv_depth, Lu_depth);

        // 1. i_of(w) = lv_level[w]; j_of(w) = k+1 - lu_level[w].
        sc.touched_ij.clear();
        for (unsigned i = comp_begin; i < comp_end; ++i) {
            const unsigned w = sc.comp_verts[i];
            const int iv = sc.lv_level[w];
            const int ju = sc.lu_level[w];
            if (iv > 0 && ju > 0) {
                sc.i_of[w] = iv;
                sc.j_of[w] = k_total + 1 - ju;
                sc.touched_ij.push_back(w);
            }
        }

        // 2. Diagonal placement.
        sc.touched_final.clear();
        sc.counts_touched.clear();
        for (unsigned w : sc.touched_ij) {
            if (sc.i_of[w] == sc.j_of[w]) {
                sc.placed[w] = 1;
                sc.final_level[w] = sc.i_of[w];
                sc.touched_final.push_back(w);
                const int r = sc.i_of[w];
                if (sc.counts[r] == 0) sc.counts_touched.push_back(static_cast<unsigned>(r));
                sc.counts[r]++;
            }
        }

        // 3. rem = in_comp && !placed (within touched_ij, which is exactly the
        // set of component vertices that had valid i_of/j_of).
        for (unsigned w : sc.touched_ij) {
            if (!sc.placed[w]) sc.rem[w] = 1;
        }

        // 4. Inner CC over rem. Seed order: ascending vertex id within component,
        // which makes cc_seeds[k] equal V1's per-subcomponent first vertex in
        // BFS order (seed == lowest-id vertex in the subcomponent).
        sc.touched_cc.clear();
        sc.cc_verts.clear();
        sc.cc_start.clear();
        sc.cc_seeds.clear();
        sc.cc_order.clear();
        sc.cc_start.push_back(0);

        for (unsigned i = comp_begin; i < comp_end; ++i) {
            const unsigned seed = sc.comp_verts[i];
            if (!sc.rem[seed] || sc.cc_mark[seed] != UINT_MAX) continue;

            const auto sub_id = static_cast<unsigned>(sc.cc_seeds.size());
            unsigned qs = 0, qe = 0;
            sc.bfs_q[qe++] = seed;
            sc.cc_mark[seed] = sub_id;
            sc.touched_cc.push_back(seed);
            sc.cc_verts.push_back(seed);
            while (qs < qe) {
                const unsigned u = sc.bfs_q[qs++];
                for (unsigned p = ptr[u]; p < ptr[u + 1]; ++p) {
                    const unsigned w = ind[p];
                    if (sc.rem[w] && sc.cc_mark[w] == UINT_MAX) {
                        sc.cc_mark[w] = sub_id;
                        sc.touched_cc.push_back(w);
                        sc.bfs_q[qe++] = w;
                        sc.cc_verts.push_back(w);
                    }
                }
            }
            sc.cc_seeds.push_back(seed);
            sc.cc_start.push_back(static_cast<unsigned>(sc.cc_verts.size()));
        }
        const auto num_sub = static_cast<unsigned>(sc.cc_seeds.size());

        // 5. Sort subcomponents by (size desc, seed asc).
        sc.cc_order.resize(num_sub);
        for (unsigned i = 0; i < num_sub; ++i) sc.cc_order[i] = i;
        std::ranges::sort(sc.cc_order.begin(), sc.cc_order.end(),
            [&](unsigned a, unsigned b) {
                const unsigned sa = sc.cc_start[a + 1] - sc.cc_start[a];
                const unsigned sb = sc.cc_start[b + 1] - sc.cc_start[b];
                if (sa != sb) return sa > sb;
                return sc.cc_seeds[a] < sc.cc_seeds[b];
            });

        // 6. For each subcomponent, compute H/L, decide FIRST vs SECOND, assign
        // final_level[w], update counts[r]. Record firstChoice from the first
        // processed subcomponent (matches V1).
        bool firstChoiceFirst = true;
        bool first_choice_set = false;

        for (unsigned idx : sc.cc_order) {
            const unsigned start = sc.cc_start[idx];
            const unsigned end   = sc.cc_start[idx + 1];

            // Populate H[r], L[r] for this subcomponent.
            for (unsigned i = start; i < end; ++i) {
                const unsigned w = sc.cc_verts[i];
                sc.Hbuf[sc.i_of[w]]++;
                sc.Lbuf[sc.j_of[w]]++;
            }

            // V1 computes max over ALL r in 1..k_total (no "H[r]>0" guard, despite
            // the paper's description). Preserve exactly.
            int h0 = 0, l0 = 0;
            for (int r = 1; r <= k_total; ++r) {
                const int vh = sc.counts[r] + sc.Hbuf[r];
                const int vl = sc.counts[r] + sc.Lbuf[r];
                if (vh > h0) h0 = vh;
                if (vl > l0) l0 = vl;
            }

            bool choose_first;
            if (h0 < l0)      choose_first = true;
            else if (l0 < h0) choose_first = false;
            else {
                // Tie on min-max: break by Lv vs Lu width. Tie-of-ties -> FIRST.
                if (Lv_width < Lu_width)      choose_first = true;
                else if (Lu_width < Lv_width) choose_first = false;
                else                           choose_first = true;
            }

            if (!first_choice_set) {
                firstChoiceFirst = choose_first;
                first_choice_set = true;
            }

            // Assign final_level + bump counts.
            if (choose_first) {
                for (unsigned i = start; i < end; ++i) {
                    const unsigned w = sc.cc_verts[i];
                    const int r = sc.i_of[w];
                    sc.final_level[w] = r;
                    sc.touched_final.push_back(w);
                    if (sc.counts[r] == 0) sc.counts_touched.push_back(static_cast<unsigned>(r));
                    sc.counts[r]++;
                }
            } else {
                for (unsigned i = start; i < end; ++i) {
                    const unsigned w = sc.cc_verts[i];
                    const int r = sc.j_of[w];
                    sc.final_level[w] = r;
                    sc.touched_final.push_back(w);
                    if (sc.counts[r] == 0) sc.counts_touched.push_back(static_cast<unsigned>(r));
                    sc.counts[r]++;
                }
            }

            // Zero H, L for next subcomponent using this subcomponent's vertex
            // list (idempotent: setting to 0 is safe even for repeated r).
            for (unsigned i = start; i < end; ++i) {
                const unsigned w = sc.cc_verts[i];
                sc.Hbuf[sc.i_of[w]] = 0;
                sc.Lbuf[sc.j_of[w]] = 0;
            }
        }

        // --------------------------------------------------------------
        // Algorithm III: numbering.
        // --------------------------------------------------------------

        // A. Swap u/v and reverse level indices if deg(u) < deg(v).
        unsigned u_eff = u_vert, v_eff = v_vert;
        bool swapped = false;
        if (gps_deg(ptr, u_eff) < gps_deg(ptr, v_eff)) {
            std::swap(u_eff, v_eff);
            swapped = true;

            // Flip level indices in-place for every placed vertex.
            for (unsigned w : sc.touched_final) {
                sc.final_level[w] = k_total + 1 - sc.final_level[w];
            }

            // Rebuild counts from scratch (index r becomes k+1-r).
            for (unsigned r : sc.counts_touched) sc.counts[r] = 0;
            sc.counts_touched.clear();
            for (unsigned w : sc.touched_final) {
                const int r = sc.final_level[w];
                if (sc.counts[r] == 0) sc.counts_touched.push_back(static_cast<unsigned>(r));
                sc.counts[r]++;
            }
        }

        // Build N_off + N_flat from final_level via counting sort.
        sc.N_off.assign(static_cast<size_t>(k_total + 2), 0);
        for (unsigned r : sc.counts_touched) sc.N_off[r] = static_cast<unsigned>(sc.counts[r]);

        unsigned acc = 0;
        for (int r = 1; r <= k_total; ++r) {
            const unsigned sz = sc.N_off[r];
            sc.N_off[r] = acc;
            acc += sz;
        }
        sc.N_off[k_total + 1] = acc;

        sc.N_cursor.assign(sc.N_off.begin(), sc.N_off.end());
        for (unsigned i = comp_begin; i < comp_end; ++i) {
            const unsigned w = sc.comp_verts[i];
            const int r = sc.final_level[w];
            if (r >= 1 && r <= k_total) {
                sc.N_flat[sc.N_cursor[r]++] = w;
            }
        }

        // B. Number level N_1.
        sc.order.clear();
        sc.touched_labeled.clear();
        sc.level_start_in_order.assign(static_cast<size_t>(k_total + 2), 0);
        sc.level_start_in_order[1] = 0;

        //  B.1 Seed with v_eff if it lives in level 1 (V1: `if level_of[v]==1`).
        if (sc.final_level[v_eff] == 1) {
            sc.order.push_back(v_eff);
            sc.labeled[v_eff] = 1;
            sc.touched_labeled.push_back(v_eff);
        }

        //  B.2 + B.3: forward-walk intra-level-1 expansion + leftover pickup.
        //  V1's semantics for level 1 are full B.2 iteration (every level-1
        //  vertex that gains unnumbered level-1 neighbors gets expanded)
        //  followed by leftover seed and loop.
        size_t scan = 0;
        while (true) {
            while (scan < sc.order.size()) {
                const unsigned w = sc.order[scan++];
                if (sc.final_level[w] != 1) continue;
                sc.cand.clear();
                for (unsigned p = ptr[w]; p < ptr[w + 1]; ++p) {
                    const unsigned x = ind[p];
                    if (sc.final_level[x] == 1 && !sc.labeled[x]) sc.cand.push_back(x);
                }
                if (sc.cand.empty()) continue;
                gps_sort_by_deg(sc.cand, ptr);
                for (unsigned x : sc.cand) {
                    sc.labeled[x] = 1;
                    sc.touched_labeled.push_back(x);
                    sc.order.push_back(x);
                }
            }
            //  Leftover pickup: min-(deg,id) unlabeled vertex in N_1.
            unsigned lo_best = UINT_MAX;
            unsigned lo_deg  = UINT_MAX;
            for (unsigned idx = sc.N_off[1]; idx < sc.N_off[2]; ++idx) {
                const unsigned x = sc.N_flat[idx];
                if (sc.labeled[x]) continue;
                const unsigned d = gps_deg(ptr, x);
                if (d < lo_deg || (d == lo_deg && x < lo_best)) {
                    lo_deg  = d;
                    lo_best = x;
                }
            }
            if (lo_best == UINT_MAX) break;
            sc.labeled[lo_best] = 1;
            sc.touched_labeled.push_back(lo_best);
            sc.order.push_back(lo_best);
            // Re-enter forward walk; scan continues from where it was, which
            // is now pointing at (or just past) the just-appended leftover.
        }
        sc.level_start_in_order[2] = static_cast<unsigned>(sc.order.size());

        // C. Number levels N_2 .. N_k.
        for (int i = 2; i <= k_total; ++i) {
            const unsigned li_m1_start = sc.level_start_in_order[i - 1];
            const unsigned li_m1_end   = sc.level_start_in_order[i];

            // C.1 Single forward walk across level-(i-1) parents in `order`.
            //     The natural iteration order gives V1's "lowest-numbered
            //     parent with unnumbered level-i neighbors".
            for (unsigned si = li_m1_start; si < li_m1_end; ++si) {
                const unsigned parent = sc.order[si];
                sc.cand.clear();
                for (unsigned p = ptr[parent]; p < ptr[parent + 1]; ++p) {
                    const unsigned x = ind[p];
                    if (sc.final_level[x] == i && !sc.labeled[x]) sc.cand.push_back(x);
                }
                if (sc.cand.empty()) continue;
                gps_sort_by_deg(sc.cand, ptr);
                for (unsigned x : sc.cand) {
                    sc.labeled[x] = 1;
                    sc.touched_labeled.push_back(x);
                    sc.order.push_back(x);
                }
            }

            // C.2 Leftover pickup. NOTE: V1 deliberately does NOT perform the
            //     full B.2 intra-level iteration for level i >= 2 -- only a
            //     seed + one-hop expansion. Preserve that quirk exactly.
            while (true) {
                unsigned lo_best = UINT_MAX;
                unsigned lo_deg  = UINT_MAX;
                for (unsigned idx = sc.N_off[i]; idx < sc.N_off[i + 1]; ++idx) {
                    const unsigned x = sc.N_flat[idx];
                    if (sc.labeled[x]) continue;
                    const unsigned d = gps_deg(ptr, x);
                    if (d < lo_deg || (d == lo_deg && x < lo_best)) {
                        lo_deg  = d;
                        lo_best = x;
                    }
                }
                if (lo_best == UINT_MAX) break;
                sc.labeled[lo_best] = 1;
                sc.touched_labeled.push_back(lo_best);
                sc.order.push_back(lo_best);

                // One-hop expansion from lo_best within level i.
                sc.cand.clear();
                for (unsigned p = ptr[lo_best]; p < ptr[lo_best + 1]; ++p) {
                    const unsigned x = ind[p];
                    if (sc.final_level[x] == i && !sc.labeled[x]) sc.cand.push_back(x);
                }
                if (!sc.cand.empty()) {
                    gps_sort_by_deg(sc.cand, ptr);
                    for (unsigned x : sc.cand) {
                        sc.labeled[x] = 1;
                        sc.touched_labeled.push_back(x);
                        sc.order.push_back(x);
                    }
                }
            }
            sc.level_start_in_order[i + 1] = static_cast<unsigned>(sc.order.size());
        }

        // D. GPS reversal rule.
        const bool needReverse =
            ( swapped && !firstChoiceFirst) ||
            (!swapped &&  firstChoiceFirst);
        if (needReverse) {
            std::ranges::reverse(sc.order.begin(), sc.order.end());
        }

        // Append this component's order into the global total.
        for (unsigned w : sc.order) sc.total_order.push_back(w);

        // --------------------------------------------------------------
        // Per-component scratch reset (proportional to component size).
        // --------------------------------------------------------------
        gps_reset_level(sc.lv_level.data(),  sc.touched_lv);
        gps_reset_level(sc.lu_level.data(),  sc.touched_lu);
        gps_reset_level(sc.tmp_level.data(), sc.touched_tmp);

        for (unsigned w : sc.touched_ij) {
            sc.i_of[w]   = -1;
            sc.j_of[w]   = -1;
            sc.placed[w] = 0;
            sc.rem[w]    = 0;
        }
        sc.touched_ij.clear();

        for (unsigned r : sc.counts_touched) sc.counts[r] = 0;
        sc.counts_touched.clear();

        for (unsigned w : sc.touched_final) sc.final_level[w] = -1;
        sc.touched_final.clear();

        for (unsigned w : sc.touched_labeled) sc.labeled[w] = 0;
        sc.touched_labeled.clear();

        for (unsigned w : sc.touched_cc) sc.cc_mark[w] = UINT_MAX;
        sc.touched_cc.clear();
    }

    // V1-compatible safety: append any vertex never written to total_order
    // (should not occur with correct CC, but preserve behavior).
    if (sc.total_order.size() != nr) {
        std::vector<char> seen(nr, 0);
        for (unsigned w : sc.total_order) seen[w] = 1;
        for (unsigned v = 0; v < nr; ++v) {
            if (!seen[v]) sc.total_order.push_back(v);
        }
    }
    assert(sc.total_order.size() == nr);

    // Write row permutation: newPos -> vertex, so rowIPermutation[v] = newPos.
    rowIPermutation = new vType[nr];
    for (unsigned newPos = 0; newPos < nr; ++newPos) {
        rowIPermutation[sc.total_order[newPos]] = static_cast<vType>(newPos);
    }

    // Column permutation: mirror rows when square, else identity.
    colIPermutation = new vType[nc];
    if (nr == nc) {
        for (unsigned v = 0; v < nc; ++v) colIPermutation[v] = rowIPermutation[v];
    } else {
        for (unsigned j = 0; j < nc; ++j) colIPermutation[j] = static_cast<vType>(j);
    }
}
