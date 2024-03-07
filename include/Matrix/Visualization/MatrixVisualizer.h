#ifndef SPARSEVIZ_MATRIXVISUALIZER_H
#define SPARSEVIZ_MATRIXVISUALIZER_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>  // for std::default_random_engine
#include <sstream>
#include <vector>
#include <string>
#include "helpers.h"
#include "json.hpp"
#include "config.h"
#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include "KernelResult.h"


const int SORT_CALC_AVG_DIFF_DEBUG = 0;
const bool EXPONENTIAL_COLORSCALE = false;
const int NROWBLOCKS = 3;
const int rowBlockSizes[NROWBLOCKS] = {32, 64, 128};

struct MatrixBin
{
    MatrixBin(): nonzeroCount(0), totalValues(0.0f), absTotalValues(0.0f) {}

    int nonzeroCount;
    float totalValues;
    float absTotalValues;
};

struct Statistic
{
public:
    Statistic(int no_kernels = 0, char **kernelNames = nullptr)
    :   no_kernels(no_kernels)
    {
        maxBandwidth = -1;
        maxRowSpan = -1;
        maxColSpan = -1;
        normalizedBandwidth = 0;
        normalizedColSpan = 0;
        normalizedRowSpan = 0;
        avgRowSpan = 0;
        avgColSpan = 0;
        avgBandwidth = 0;
        no_empty_bins = 0;
        geo_mean_nnz = 0;
        for (int i = 0; i < NROWBLOCKS; i++)
            rowBlockEfficiency[i] = 0;
        if (no_kernels > 0)
        {
            kernelTimes = new double[no_kernels];
        }
    }

    std::string matrixName;
    std::string orderingName;

    int no_bins;
    int no_empty_bins;
    double mean_nnz;
    int median_nnz;
    double geo_mean_nnz;

    int no_kernels;
    double *kernelTimes;

    int maxBandwidth;
    double avgBandwidth;
    double normalizedBandwidth;

    int maxRowSpan;
    double avgRowSpan;
    double normalizedRowSpan;

    int maxColSpan;
    double avgColSpan;
    double normalizedColSpan;

    double rowBlockEfficiency[NROWBLOCKS];
};

/*!
 * @brief Visualization function for generating an html file in which matrix orderings are listed for a single matrix.
 * @param orderings Pointer to an array of MatrixOrdering pointers, each pointing to a custom ordering derived from the abstract MatrixOrdering class.
 * @param norder The total number of matrix orderings to be visualized.
 */
void visualizeMatrixOrderings(MatrixOrdering** orderings, int norder);

/*!
 * @brief Visualization function for generating an html file in which matrices are listed for a single ordering.
 * @param orderings Pointer to an array of MatrixOrdering pointers, each pointing to a custom ordering derived from the abstract MatrixOrdering class.
 * @param norder The total number of matrices to be visualized.
 */
void visualizeMatrices(MatrixOrdering** orderings, int norder);


#endif //SPARSEVIZ_MATRIXVISUALIZER_H
