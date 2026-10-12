#include "MatrixFeatureExtractor.h"
#include "SparseMatrix.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>


MatrixFeatures MatrixFeatureExtractor::extract(const SparseMatrix& matrix)
{
    MatrixFeatures features;
    sizeAndRowFeatures(matrix, features);
    bandFeatures(matrix, features);
    columnLocalityFeatures(matrix, features);
    reuseDistanceFeatures(matrix, features);
    blockFeatures(matrix, features, 8);
    blockFeatures(matrix, features, 64);
    return features;
}

std::string MatrixFeatureExtractor::header(const MatrixFeatures& features, char separator)
{
    std::string line;
    for (const auto& f: features.values)
    {
        line += f.first;
        line += separator;
    }
    return line;
}

std::string MatrixFeatureExtractor::row(const MatrixFeatures& features, char separator)
{
    std::ostringstream stream;
    stream << std::setprecision(10);
    for (const auto& f: features.values)
    {
        stream << f.second << separator;
    }
    return stream.str();
}

void MatrixFeatureExtractor::sizeAndRowFeatures(const SparseMatrix& matrix, MatrixFeatures& features)
{
    const vType n = matrix.getRowCount();
    const vType m = matrix.getColCount();
    const vType* ptrs = matrix.getPtr();
    const double nnz = ptrs[n];

    features.add("rows", n);
    features.add("cols", m);
    features.add("nnz", nnz);
    features.add("density", (n && m) ? nnz / ((double) n * (double) m) : 0);

    double mean = n ? nnz / n : 0;
    double var = 0;
    vType maxLen = 0;
    vType emptyRows = 0;
    for (vType i = 0; i < n; ++i)
    {
        vType len = ptrs[i + 1] - ptrs[i];
        var += (len - mean) * (len - mean);
        maxLen = std::max(maxLen, len);
        if (len == 0) ++emptyRows;
    }
    var = n ? var / n : 0;

    features.add("row_len_mean", mean);
    features.add("row_len_cv", mean > 0 ? std::sqrt(var) / mean : 0);
    features.add("row_len_max_over_mean", mean > 0 ? maxLen / mean : 0);
    features.add("empty_row_frac", n ? (double) emptyRows / n : 0);
}

void MatrixFeatureExtractor::bandFeatures(const SparseMatrix& matrix, MatrixFeatures& features)
{
    const vType n = matrix.getRowCount();
    const vType m = matrix.getColCount();
    const vType* ptrs = matrix.getPtr();
    const vType* ids = matrix.getInd();
    const double nnz = ptrs[n];

    // rectangular case: scale column index to row axis
    const double scale = (m > 0) ? (double) n / (double) m : 1.0;
    const double band1 = 0.01 * n;
    const double band10 = 0.10 * n;

    double maxDist = 0, sumDist = 0;
    double inBand1 = 0, inBand10 = 0;
    double spanSum = 0;
    vType nonEmptyRows = 0;

    for (vType i = 0; i < n; ++i)
    {
        if (ptrs[i + 1] == ptrs[i]) continue;
        ++nonEmptyRows;
        vType minCol = ids[ptrs[i]], maxCol = ids[ptrs[i]];
        for (vType p = ptrs[i]; p < ptrs[i + 1]; ++p)
        {
            vType j = ids[p];
            minCol = std::min(minCol, j);
            maxCol = std::max(maxCol, j);
            double dist = std::fabs((double) i - scale * (double) j);
            sumDist += dist;
            maxDist = std::max(maxDist, dist);
            if (dist <= band1) ++inBand1;
            if (dist <= band10) ++inBand10;
        }
        spanSum += (double) (maxCol - minCol) / (m ? m : 1);
    }

    features.add("bw_max_norm", n ? maxDist / n : 0);
    features.add("bw_avg_norm", (n && nnz) ? (sumDist / nnz) / n : 0);
    features.add("band_frac_1pct", nnz ? inBand1 / nnz : 0);
    features.add("band_frac_10pct", nnz ? inBand10 / nnz : 0);
    features.add("row_span_avg_norm", nonEmptyRows ? spanSum / nonEmptyRows : 0);
}

void MatrixFeatureExtractor::columnLocalityFeatures(const SparseMatrix& matrix, MatrixFeatures& features)
{
    // column ids in a row are sorted (also after ordering)
    const vType n = matrix.getRowCount();
    const vType* ptrs = matrix.getPtr();
    const vType* ids = matrix.getInd();
    const double nnz = ptrs[n];

    double distinctLines = 0;
    double gapSum = 0;
    double gapCount = 0;

    for (vType i = 0; i < n; ++i)
    {
        vType prevLine = 0;
        bool first = true;
        for (vType p = ptrs[i]; p < ptrs[i + 1]; ++p)
        {
            vType line = ids[p] / DOUBLES_PER_LINE;
            if (first)
            {
                first = false;
                ++distinctLines;
            }
            else
            {
                gapSum += (double) (line - prevLine);
                ++gapCount;
                if (line != prevLine) ++distinctLines;
            }
            prevLine = line;
        }
    }

    features.add("distinct_lines_per_nnz", nnz ? distinctLines / nnz : 0);
    features.add("col_gap_mean_lines", gapCount ? gapSum / gapCount : 0);
}

void MatrixFeatureExtractor::reuseDistanceHistogram(const SparseMatrix& matrix, std::vector<double>& histogram, double& compulsory, double& total)
{
    // simple cache model for the x accesses of a CSR SpMV: go over the nonzeros in order and
    // for every x[ids[p]] look how many accesses ago the same cache line was touched last time.
    // if that is smaller than the cache size (in lines) we count it as a hit. this is an
    // approximation, it counts accesses and not distinct lines, so it is a bit pessimistic.
    // histogram[k] = accesses whose distance d is 2^(k-1) <= d < 2^k, k = 0 is d = 0
    const vType n = matrix.getRowCount();
    const vType m = matrix.getColCount();
    const vType* ptrs = matrix.getPtr();
    const vType* ids = matrix.getInd();
    const size_t nnz = ptrs[n];

    const size_t lineCount = (m + DOUBLES_PER_LINE - 1) / DOUBLES_PER_LINE;
    std::vector<long long> lastAccess(lineCount, -1);

    histogram.assign(REUSE_BUCKETS, 0.0);
    compulsory = 0;
    total = nnz ? (double) nnz : 1;

    long long t = 0;
    for (vType i = 0; i < n; ++i)
    {
        for (vType p = ptrs[i]; p < ptrs[i + 1]; ++p, ++t)
        {
            size_t line = ids[p] / DOUBLES_PER_LINE;
            if (lastAccess[line] < 0)
            {
                ++compulsory;
            }
            else
            {
                long long distance = t - lastAccess[line] - 1;
                int bucket = 0;
                while (bucket + 1 < REUSE_BUCKETS && distance >= (1LL << bucket)) ++bucket;
                histogram[bucket] += 1;
            }
            lastAccess[line] = t;
        }
    }
}

std::vector<double> MatrixFeatureExtractor::reuseDistanceCDF(const SparseMatrix& matrix)
{
    std::vector<double> histogram;
    double compulsory, total;
    reuseDistanceHistogram(matrix, histogram, compulsory, total);

    std::vector<double> cdf(REUSE_BUCKETS, 0.0);
    double running = 0;
    for (int k = 0; k < REUSE_BUCKETS; ++k)
    {
        running += histogram[k];   // accesses with distance < 2^k
        cdf[k] = running / total;
    }
    return cdf;
}

void MatrixFeatureExtractor::reuseDistanceFeatures(const SparseMatrix& matrix, MatrixFeatures& features)
{
    std::vector<double> histogram;
    double compulsory, total;
    reuseDistanceHistogram(matrix, histogram, compulsory, total);

    auto hitsBelow = [&](unsigned long long capacityLines)
    {
        // buckets with 2^k <= capacity are all hits
        double hits = 0;
        for (int k = 0; k < REUSE_BUCKETS && (1ull << k) <= capacityLines; ++k) hits += histogram[k];
        return hits / total;
    };
    double logDistSum = 0, reuseCount = 0;
    for (int k = 0; k < REUSE_BUCKETS; ++k)
    {
        logDistSum += histogram[k] * k;  // bucket k ~ log2 distance k
        reuseCount += histogram[k];
    }

    features.add("x_compulsory_frac", compulsory / total);
    features.add("x_reuse_hit_l1", hitsBelow(L1_LINES));
    features.add("x_reuse_hit_l2", hitsBelow(L2_LINES));
    features.add("x_reuse_hit_l3", hitsBelow(L3_LINES));
    features.add("x_reuse_mean_log2_dist", reuseCount ? logDistSum / reuseCount : 0);
}

void MatrixFeatureExtractor::blockFeatures(const SparseMatrix& matrix, MatrixFeatures& features, vType blockSize)
{
    const vType n = matrix.getRowCount();
    const vType m = matrix.getColCount();
    const vType* ptrs = matrix.getPtr();
    const vType* ids = matrix.getInd();
    const double nnz = ptrs[n];

    const vType rowBlocks = (n + blockSize - 1) / blockSize;
    const vType colBlocks = (m + blockSize - 1) / blockSize;
    std::vector<vType> stamp(colBlocks, 0);
    vType currentStamp = 0;
    double nonEmptyBlocks = 0;

    for (vType rb = 0; rb < rowBlocks; ++rb)
    {
        ++currentStamp;
        vType rowEnd = std::min(n, (rb + 1) * blockSize);
        for (vType i = rb * blockSize; i < rowEnd; ++i)
        {
            for (vType p = ptrs[i]; p < ptrs[i + 1]; ++p)
            {
                vType cb = ids[p] / blockSize;
                if (stamp[cb] != currentStamp)
                {
                    stamp[cb] = currentStamp;
                    ++nonEmptyBlocks;
                }
            }
        }
    }

    const std::string suffix = "_" + std::to_string(blockSize);
    const double blockArea = (double) blockSize * (double) blockSize;
    features.add("block_fill" + suffix, nonEmptyBlocks ? nnz / (nonEmptyBlocks * blockArea) : 0);
    features.add("block_nonempty_frac" + suffix, (rowBlocks && colBlocks) ? nonEmptyBlocks / ((double) rowBlocks * (double) colBlocks) : 0);
}
