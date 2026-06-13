#include "DRSAOrdering.h"
#include <cmath>
#include <algorithm>
#include <iostream>

DRSAOrdering::DRSAOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters)
    : MatrixOrdering(matrix, orderingName, false, false, 500) // rectangularSupport=false, patternUnsymmetricSupport=false
{
    // Parameters determined via the full factorial tuning in the paper
    m_T0 = 1000.0;
    m_Tf = 1e-7;
    m_L0 = 40.0;
    
    std::random_device rd;
    m_Rng = std::mt19937(rd());
}


void DRSAOrdering::orderingFunction()
{
    vType n = getMatrix().getRowCount();
    vType m = std::max((vType)1, getMatrix().getNNZCount() / 2);

    m_Alpha = (n < 5000) ? 0.97 : 0.99;
    m_Lf = std::min(10.0 * n * m, 500.0 * n);
    
    double r_sa = (std::log(m_Tf) - std::log(m_T0)) / std::log(m_Alpha);
    double gamma = std::exp((std::log(m_Lf) - std::log(m_L0)) / r_sa);

    double T = m_T0;
    double L = m_L0;

    std::vector<vType> current_pi(n);
    std::vector<vType> current_rho(n);
    for (vType i = 0; i < n; ++i) {
        current_pi[i] = i;
        current_rho[i] = i;
    }
    std::shuffle(current_pi.begin(), current_pi.end(), m_Rng);
    for (vType i = 0; i < n; ++i) {
        current_rho[current_pi[i]] = i;
    }

    vType beta = 0;
    std::vector<vType> d(n + 1, 0);
    const vType* ptr = getMatrix().getPtr();
    const vType* ind = getMatrix().getInd();

    double current_delta = 0.0;

    // Initial calculation
    for (vType u = 0; u < n; ++u) {
        for (vType j = ptr[u]; j < ptr[u + 1]; ++j) {
            vType v = ind[j];
            if (u < v) {
                vType diff = std::abs(static_cast<long long>(current_pi[u]) - current_pi[v]);
                if (diff > beta) beta = diff;
                d[diff]++;
            }
        }
    }

    for (vType i = 0; i <= beta; ++i) {
        current_delta += static_cast<double>(d[i]) / (n - i + 1);
    }

    std::vector<vType> w_pi = current_pi;
    double f_w = static_cast<double>(beta) + current_delta;
    double f_x = f_w;

    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    std::uniform_int_distribution<vType> dist_n(0, n - 1);

    std::vector<vType> changed_vertices;
    changed_vertices.reserve(10);

    // Fast byte-level boolean array to replace sorting/unique operations
    std::vector<char> in_changed(n, 0);

    int stay_count = 0;
    const int MAX_STAYS = 3;

    while (T > m_Tf) {
        bool improvement = false;
        int L_int = static_cast<int>(std::min(L, static_cast<double>(5 * n)));

        for (int step = 0; step < L_int; ++step) {

            double p = p_dist(m_Rng);
            int op = 0;
            vType u = 0, v = 0;
            int rot_i = 0, rot_j = 0;

            if (p <= 0.6) {
                op = 0;
                u = dist_n(m_Rng); v = dist_n(m_Rng);
                if (u != v) {
                    changed_vertices.push_back(u);
                    changed_vertices.push_back(v);
                }
            } else if (p <= 0.8) {
                op = 1;
                u = dist_n(m_Rng);
                vType degree = ptr[u + 1] - ptr[u];
                if (degree > 0) {
                    std::uniform_int_distribution<vType> nbr_dist(ptr[u], ptr[u + 1] - 1);
                    v = ind[nbr_dist(m_Rng)];
                    if (u != v) {
                        changed_vertices.push_back(u);
                        changed_vertices.push_back(v);
                    }
                }
            } else {
                op = 2;
                int max_r = std::min(5, static_cast<int>(n) - 1);
                if (max_r >= 1) {
                    std::uniform_int_distribution<int> dist_r(1, max_r);
                    int r = dist_r(m_Rng);
                    std::uniform_int_distribution<int> dist_i(0, static_cast<int>(n) - 1 - r);
                    rot_i = dist_i(m_Rng);
                    rot_j = rot_i + r;
                    for (int k = rot_i; k <= rot_j; ++k) {
                        changed_vertices.push_back(current_rho[k]);
                    }
                }
            }

            if (changed_vertices.empty()) continue;

            // Mark vertices for instant O(1) existence checks
            for (vType v_c : changed_vertices) in_changed[v_c] = 1;

            // 1. Instantly subtract old edge distances
            for (vType node : changed_vertices) {
                for (vType j = ptr[node]; j < ptr[node + 1]; ++j) {
                    vType nbr = ind[j];
                    // Skip if the neighbor is ALSO in changed_vertices AND we already processed it
                    if (in_changed[nbr] && node > nbr) continue;

                    vType diff = std::abs((long long)current_pi[node] - current_pi[nbr]);
                    d[diff]--;
                    current_delta -= 1.0 / (n - diff + 1);
                }
            }

            // 2. Swap array layouts
            if (op == 0 || op == 1) {
                std::swap(current_pi[u], current_pi[v]);
                current_rho[current_pi[u]] = u;
                current_rho[current_pi[v]] = v;
            } else {
                vType temp_vertex = current_rho[rot_i];
                for (int k = rot_i; k < rot_j; ++k) {
                    current_rho[k] = current_rho[k + 1];
                    current_pi[current_rho[k]] = k;
                }
                current_rho[rot_j] = temp_vertex;
                current_pi[temp_vertex] = rot_j;
            }

            // 3. Instantly add new edge distances
            vType max_new_diff = 0;
            for (vType node : changed_vertices) {
                for (vType j = ptr[node]; j < ptr[node + 1]; ++j) {
                    vType nbr = ind[j];
                    if (in_changed[nbr] && node > nbr) continue;

                    vType diff = std::abs((long long)current_pi[node] - current_pi[nbr]);
                    d[diff]++;
                    current_delta += 1.0 / (n - diff + 1);
                    if (diff > max_new_diff) max_new_diff = diff;
                }
            }

            vType old_beta = beta;
            if (max_new_diff > beta) {
                beta = max_new_diff;
            } else {
                while (beta > 0 && d[beta] == 0) {
                    beta--;
                }
            }

            // O(1) Evaluation!
            double f_y = static_cast<double>(beta) + current_delta;

            // 4. Check Acceptance
            if (f_y < f_x || p_dist(m_Rng) < std::exp(-(f_y - f_x) / T)) {
                f_x = f_y;
                if (f_x < f_w) {
                    w_pi = current_pi;
                    f_w = f_x;
                    improvement = true;
                }
            } else {
                // 5. Revert Changes (Subtract new distances)
                for (vType node : changed_vertices) {
                    for (vType j = ptr[node]; j < ptr[node + 1]; ++j) {
                        vType nbr = ind[j];
                        if (in_changed[nbr] && node > nbr) continue;
                        vType diff = std::abs((long long)current_pi[node] - current_pi[nbr]);
                        d[diff]--;
                        current_delta -= 1.0 / (n - diff + 1);
                    }
                }

                // Un-swap layouts
                if (op == 0 || op == 1) {
                    std::swap(current_pi[u], current_pi[v]);
                    current_rho[current_pi[u]] = u;
                    current_rho[current_pi[v]] = v;
                } else {
                    vType temp_vertex = current_rho[rot_j];
                    for (int k = rot_j; k > rot_i; --k) {
                        current_rho[k] = current_rho[k - 1];
                        current_pi[current_rho[k]] = k;
                    }
                    current_rho[rot_i] = temp_vertex;
                    current_pi[temp_vertex] = rot_i;
                }

                // Re-add old distances
                for (vType node : changed_vertices) {
                    for (vType j = ptr[node]; j < ptr[node + 1]; ++j) {
                        vType nbr = ind[j];
                        if (in_changed[nbr] && node > nbr) continue;
                        vType diff = std::abs((long long)current_pi[node] - current_pi[nbr]);
                        d[diff]++;
                        current_delta += 1.0 / (n - diff + 1);
                    }
                }
                beta = old_beta;
            }

            // Unmark tracking array for the next step
            for (vType v_c : changed_vertices) in_changed[v_c] = 0;
            changed_vertices.clear();

            // Prevent floating point drift accumulating over millions of loop iterations
            if (step > 0 && step % 10000 == 0) {
                double exact_delta = 0.0;
                for (vType i = 0; i <= beta; ++i) {
                    exact_delta += static_cast<double>(d[i]) / (n - i + 1);
                }
                current_delta = exact_delta;
            }
        }

        if (!improvement || stay_count >= MAX_STAYS) {
            T *= m_Alpha;
            L = std::min(L * gamma, m_Lf);
            stay_count = 0;
        } else {
            stay_count++;
        }
    }

    rowIPermutation = new vType[n];
    colIPermutation = new vType[n];
    for (vType i = 0; i < n; ++i) {
        rowIPermutation[i] = w_pi[i];
        colIPermutation[i] = w_pi[i];
    }
}
