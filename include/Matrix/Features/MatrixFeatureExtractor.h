#ifndef SPARSEVIZ_MATRIXFEATUREEXTRACTOR_H
#define SPARSEVIZ_MATRIXFEATUREEXTRACTOR_H

#include <string>
#include <vector>
#include <utility>
#include "config.h"


class SparseMatrix;

/*!
 * @brief Structural features of an (ordered) sparse matrix. Values are normalized so different matrices can be compared.
 * Insertion order is the column order in the csv.
 */
struct MatrixFeatures
{
    std::vector<std::pair<std::string, double>> values;

    void add(const std::string& name, double value) { values.emplace_back(name, value); }
};

/*!
 * @brief Extracts features from a SparseMatrix (should be the ordered one, i.e. what the kernels run on).
 * Groups: size/density, row length distribution, bandwidth/span, column locality,
 * a simple reuse distance model of the x accesses in SpMV (gives approximate L1/L2/L3 hit ratios), block fill.
 */
class MatrixFeatureExtractor
{
public:
    MatrixFeatureExtractor() = delete;

    static MatrixFeatures extract(const SparseMatrix& matrix);

    static std::string header(const MatrixFeatures& features, char separator = '\t');
    static std::string row(const MatrixFeatures& features, char separator = '\t');

    /*!
     * @brief cdf[k] = fraction of x accesses whose last touch of the same line was < 2^k accesses ago. First touches never hit.
     */
    static std::vector<double> reuseDistanceCDF(const SparseMatrix& matrix);
    static constexpr int REUSE_BUCKETS = 26;

private:
    static void sizeAndRowFeatures(const SparseMatrix& matrix, MatrixFeatures& features);
    static void bandFeatures(const SparseMatrix& matrix, MatrixFeatures& features);
    static void columnLocalityFeatures(const SparseMatrix& matrix, MatrixFeatures& features);
    static void reuseDistanceFeatures(const SparseMatrix& matrix, MatrixFeatures& features);
    static void reuseDistanceHistogram(const SparseMatrix& matrix, std::vector<double>& histogram, double& compulsory, double& total);
    static void blockFeatures(const SparseMatrix& matrix, MatrixFeatures& features, vType blockSize);

public:
    // 64 byte lines = 8 doubles, capacities in lines
    static constexpr vType DOUBLES_PER_LINE = 8;
    static constexpr unsigned long long L1_LINES = 32ull * 1024 / 64;        // 32 KB
    static constexpr unsigned long long L2_LINES = 512ull * 1024 / 64;       // 512 KB
    static constexpr unsigned long long L3_LINES = 32ull * 1024 * 1024 / 64; // 32 MB (one EPYC 7763 CCD)
};


#endif //SPARSEVIZ_MATRIXFEATUREEXTRACTOR_H
