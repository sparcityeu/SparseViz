#ifndef SPARSEVIZ_GRAYORDERING_H
#define SPARSEVIZ_GRAYORDERING_H

#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include <vector>
#include <algorithm>
#include "config.h"
#include "helpers.h"

typedef int IDType;
typedef int NNZType;
typedef double ValueType;
typedef std::pair<IDType, unsigned long> row_grey_pair;

enum BitMapSize{
  BitSize16 = 16,
  BitSize32 = 32/*,
  BitSize64 = 64*/ //at the moment, using 64 bits is not working as intended
};


class GrayOrdering : public MatrixOrdering
{
public:
    GrayOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters) :
      MatrixOrdering(matrix, orderingName, false, false), group_size(1024), bit_resolution(32), nnz_threshold(2048) {
       if(orderingParameters != "") {
            std::vector<std::string> parameters = split(orderingParameters, '/');

            if (parameters.size() > 0) {
                group_size = stoi(parameters[0]);
            } 
            if (parameters.size() > 1) {
                bit_resolution = stoi(parameters[1]);
            }
            if (parameters.size() > 2) {
                nnz_threshold = stoi(parameters[2]);
            }
        }
    }
  
private:
    virtual void orderingFunction() override;
    int group_size;
    int bit_resolution;
    int nnz_threshold;
    unsigned long grey_bin_to_dec(unsigned long n);
    void print_dec_in_bin(unsigned long n, int size);
    unsigned long bin_to_grey(unsigned long n);
    bool is_banded(int nnz, int n_cols, NNZType *row_ptr, IDType *cols, std::vector<IDType> order, int band_size = -1);
};

#endif // SPARSEVIZ_GRAYORDERING_H