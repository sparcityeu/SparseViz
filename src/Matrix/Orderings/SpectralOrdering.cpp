#include "SpectralOrdering.h"
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <random>
#include <algorithm>
#include <limits>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;


static bool tqli(int n, vector<double>& d, vector<double>& e,
                 vector<vector<double>>& z) {
    for (int i = 1; i < n; ++i) e[i - 1] = e[i];
    e[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0, m;
        do {
            for (m = l; m < n - 1; ++m) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) + dd == dd) break;
            }
            if (m != l) {
                if (iter++ == 60) {
                    cerr << "tqli: no convergence for eigenvalue " << l << "\n";
                    return false;
                }
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = hypot(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + copysign(r, g));

                double s = 1.0, c = 1.0, p = 0.0;
                for (int i = m - 1; i >= l; --i) {
                    double f = s * e[i];
                    double b = c * e[i];
                    e[i + 1] = r = hypot(f, g);
                    if (r == 0.0) { d[i + 1] -= p; e[m] = 0.0; break; }
                    s = f / r;  c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p = s * r);
                    g = c * r - b;

                    for (int k = 0; k < n; ++k) {
                        double tmp  = z[k][i + 1];
                        z[k][i + 1] = s * z[k][i] + c * tmp;
                        z[k][i]     = c * z[k][i] - s * tmp;
                    }
                }
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
    return true;
}


SpectralOrdering::SpectralOrdering(SparseMatrix& m, string name, string params)
    : MatrixOrdering(m, move(name), false, false, 8) {}

void SpectralOrdering::orderingFunction() {
    const SparseMatrix& mat = getMatrix();
    const int n = mat.getRowCount();
    const auto* ptr = mat.getPtr();
    const auto* ind = mat.getInd();

    vector<int>    clean_ptr(n + 1, 0);
    vector<int>    clean_ind;
    clean_ind.reserve(ptr[n] - n);
    vector<int>    degrees(n, 0);
    int max_degree = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = ptr[i]; j < ptr[i + 1]; ++j) {
            if (ind[j] != i) {
                clean_ind.push_back(ind[j]);
                ++degrees[i];
            }
        }
        clean_ptr[i + 1] = static_cast<int>(clean_ind.size());
        if (degrees[i] > max_degree) max_degree = degrees[i];
    }

    const double sigma = 2.0 * max_degree + 1.0;
    vector<double> diag_vals(n);
    for (int i = 0; i < n; ++i)
        diag_vals[i] = sigma - degrees[i];

    const int*    __restrict__ cp = clean_ptr.data();
    const int*    __restrict__ ci = clean_ind.data();
    const double* __restrict__ dv = diag_vals.data();

    const int k = min(n, max(20, min(300, n)));

    vector<double> Vstore(size_t(k) * size_t(n));
    double* __restrict__ V = Vstore.data();
    const size_t nz = static_cast<size_t>(n);

    vector<double> alpha(k, 0.0);
    vector<double> beta(k > 0 ? k - 1 : 0, 0.0);
    vector<double> w(n);
    double* __restrict__ wp = w.data();

    
    constexpr int PARALLEL_THRESHOLD = 20000;
    int  actual_k = 0;
    bool stop_flag = false;


    double sum_w;
    double dot;
    double alpha_j;
    double norm2;


    #pragma omp parallel default(shared) if(n >= PARALLEL_THRESHOLD)
    {
        if (omp_get_thread_num() == 0) {
            int active_threads = omp_get_num_threads();
            printf("Running with %d active threads\n", active_threads);
        }
        #pragma omp single
        {
            mt19937 gen(42);
            uniform_real_distribution<double> dist(-1.0, 1.0);
            double s = 0.0;
            for (int i = 0; i < n; ++i) { V[i] = dist(gen); s += V[i]; }
            s /= n;
            double norm2_init = 0.0;
            for (int i = 0; i < n; ++i) { V[i] -= s; norm2_init += V[i] * V[i]; }
            double inv = 1.0 / sqrt(norm2_init);
            for (int i = 0; i < n; ++i) V[i] *= inv;
        }

        for (int j = 0; j < k; ++j) {
            if (stop_flag) break;

            const double* __restrict__ vj = V + size_t(j) * nz;

            #pragma omp single
            sum_w = 0.0;

            #pragma omp for reduction(+:sum_w) schedule(static)
            for (int i = 0; i < n; ++i) {
                const int e = cp[i + 1];
                double val = dv[i] * vj[i];
                for (int kk = cp[i]; kk < e; ++kk)
                    val += vj[ci[kk]];
                wp[i] = val;
                sum_w += val;
            }
            
            const double mean_w = sum_w / n;

            #pragma omp for simd schedule(static)
            for (int i = 0; i < n; ++i) wp[i] -= mean_w;

            for (int jj = 0; jj < j; ++jj) {
                const double* __restrict__ vjj = V + size_t(jj) * nz;

                #pragma omp single
                dot = 0.0;

                #pragma omp for simd reduction(+:dot) schedule(static)
                for (int i = 0; i < n; ++i) dot += wp[i] * vjj[i];

                #pragma omp for simd schedule(static)
                for (int i = 0; i < n; ++i) wp[i] -= dot * vjj[i];
            }

            #pragma omp single
            alpha_j = 0.0;

            #pragma omp for simd reduction(+:alpha_j) schedule(static)
            for (int i = 0; i < n; ++i) alpha_j += wp[i] * vj[i];

            #pragma omp single
            norm2 = 0.0;

            #pragma omp for simd reduction(+:norm2) schedule(static)
            for (int i = 0; i < n; ++i) {
                double v = wp[i] - alpha_j * vj[i];
                wp[i]    = v;
                norm2   += v * v;
            }
            const double bj = sqrt(norm2);

            #pragma omp single
            {
                alpha[j]  = alpha_j;
                actual_k  = j + 1;
                if (bj < 1e-10) {
                    cout << "Lanczos: invariant subspace at step " << j << "\n";
                    stop_flag = true;
                } else if (j < k - 1) {
                    beta[j] = bj;
                }
            }

            if (stop_flag) break;

            if (j < k - 1) {
                const double inv_b = 1.0 / bj;
                double* __restrict__ vjp1 = V + size_t(j + 1) * nz;
                #pragma omp for simd schedule(static)
                for (int i = 0; i < n; ++i) vjp1[i] = wp[i] * inv_b;
            }
        }
    } // end parallel region

    const int kk = actual_k;
    vector<double> td(kk), te(kk, 0.0);
    for (int i = 0; i < kk; ++i) td[i] = alpha[i];
    for (int i = 1; i < kk; ++i) te[i] = beta[i - 1];

    vector<vector<double>> Q(kk, vector<double>(kk, 0.0));
    for (int i = 0; i < kk; ++i) Q[i][i] = 1.0;

    if (!tqli(kk, td, te, Q)) {
        cerr << "SpectralOrdering: tqli did not converge – "
                "falling back to natural ordering.\n";
        this->rowIPermutation = new vType[n];
        this->colIPermutation = new vType[n];
        for (int i = 0; i < n; ++i) rowIPermutation[i] = colIPermutation[i] = i;
        return;
    }

    int fied_col = 0;
    for (int j = 1; j < kk; ++j)
        if (td[j] > td[fied_col]) fied_col = j;

    const double fiedler_eigenval_M = td[fied_col];
    const double fiedler_eigenval_L = sigma - fiedler_eigenval_M;
    cout << "Lanczos: Fiedler eigenvalue (Laplacian) = " << fiedler_eigenval_L
         << "  (in M = " << fiedler_eigenval_M << ", kk = " << kk << ")\n";


    vector<double> fiedler(n, 0.0);
    double* __restrict__ fp = fiedler.data();

    #pragma omp parallel if(n >= PARALLEL_THRESHOLD)
    {
        for (int j = 0; j < kk; ++j) {
            const double coeff = Q[j][fied_col];
            if (coeff == 0.0) continue;
            const double* __restrict__ vjr = V + size_t(j) * nz;
            #pragma omp for simd schedule(static)
            for (int i = 0; i < n; ++i) fp[i] += coeff * vjr[i];
        }
    }


    vector<int> p(n);
    iota(p.begin(), p.end(), 0);
    sort(p.begin(), p.end(),
         [&](int a, int b) { return fiedler[a] < fiedler[b]; });

    this->rowIPermutation = new vType[n];
    this->colIPermutation = new vType[n];
    for (int i = 0; i < n; ++i)
        this->rowIPermutation[p[i]] = this->colIPermutation[p[i]] = i;
}