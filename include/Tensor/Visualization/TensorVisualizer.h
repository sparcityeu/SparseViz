#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random> // for std::default_random_engine
#include <chrono> // for seeding the random number generator
#include <cassert>
#include "json.hpp"
#include "KernelResult.h"
#include "TensorOrdering.h"
#include "helpers.h"
#include "config.h"

struct TStatistic
{
public:
    TStatistic(int no_kernels = 0, char **kernelNames = nullptr)
    :   no_kernels(no_kernels)
    {
        maxSpan[0] =  maxSpan[1] =  maxSpan[2] = -1;
        normSpan[0] = normSpan[1] = normSpan[2] = 0;
        avgSpan[0] = avgSpan[1] = avgSpan[2] = 0;
        fiberCounts[0] = fiberCounts[1] = fiberCounts[2] = 0;
        singleNNZfiberCounts[0] = singleNNZfiberCounts[1] = singleNNZfiberCounts[2] = 0;
        no_empty_bins = 0;
        geo_mean_nnz = 0;
        mean_nnz = 0;
        no_bins = 0;
        median_nnz = 0;
    }

    std::string tensorName;
    std::string orderingName;

    int no_bins;
    int no_empty_bins;
    double mean_nnz;
    int median_nnz;
    double geo_mean_nnz;

    int no_kernels;
    double *kernelTimes;

    int maxSpan[3];
    double avgSpan[3];
    double normSpan[3];

    int fiberCounts[3];
    int singleNNZfiberCounts[3];
};

struct TensorBin
{
    TensorBin(): nonzeroCount(0), totalValues(0.0f), absTotalValues(0.0f) {}

    vType nonzeroCount;
    valType totalValues;
    valType absTotalValues;
};

/*!
 * @brief Visualization function for generating an html file in which tensor orderings are listed for a single tensor.
 * @param orderings Pointer to an array of TensorOrdering pointers, each pointing to a custom ordering derived from the abstract TensorOrdering class.
 * @param norder The total number of tensor orderings to be visualized.
 */
void visualizeTensorOrderings(TensorOrdering** orderings, int norder);

/*!
 * @brief Visualization function for generating an html file in which tensors are listed for a single ordering.
 * @param orderings Pointer to an array of TensorOrdering pointers, each pointing to a custom ordering derived from the abstract TensorOrdering class.
 * @param norder The total number of tensors to be visualized.
 */
void visualizeTensors(TensorOrdering** orderings, int norder);

/*!
 * @brief Visualization function for generating an html file in which all mode combinations of a single ordering & tensor are listed.
 * @param orderings Pointer to a custom ordering object deriving from TensorOrdering class.
 */
void visualizeFullSparseTensor(TensorOrdering* ordering);
double *sortAndCalculateAverageDifference3D(vType *coords, vType nnz);
