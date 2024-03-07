//                                                                                                                                                                                                          
// Created on on 15 November 2023                                                                                                                                                                
//                                                                                                                                                                                                          

#include "GrayOrdering.h"
#include <vector>
#include <string>
#include "Parameters.h"

bool desc_comparator(const row_grey_pair &l, const row_grey_pair &r) {
  return l.second > r.second;
}

bool asc_comparator(const row_grey_pair &l, const row_grey_pair &r) {
  return l.second < r.second;
}

unsigned long GrayOrdering::grey_bin_to_dec(unsigned long n) {
  unsigned long inv = 0;
  for (; n; n = n >> 1) inv ^= n;
  return inv;
}

void GrayOrdering::print_dec_in_bin(unsigned long n, int size) {
  // array to store binary number
  int binaryNum[size];

  // counter for binary array
  int i = 0;
  while (n > 0) {
    // storing remainder in binary array
    binaryNum[i] = n % 2;
    n = n / 2;
    i++;
  }

  // printing binary array in reverse order
  std::string bin_nums = "";
  for (int j = i - 1; j >= 0; j--)
    bin_nums = bin_nums + std::to_string(binaryNum[j]);
}

// not sure if all IDTypes work for this
unsigned long GrayOrdering::bin_to_grey(unsigned long n) {
  /* Right Shift the number by 1
  taking xor with original number */
  return n ^ (n >> 1);
}

bool GrayOrdering::is_banded(int nnz, int n_cols, NNZType *row_ptr, IDType *cols, std::vector<IDType> order, int band_size) {
  if (band_size == -1) band_size = n_cols / 64;
  int band_count = 0;
  bool banded = false;

  for (int r = 0; r < order.size(); r++) {
    for (int i = row_ptr[order[r]]; i < row_ptr[order[r] + 1]; i++) {
      int col = cols[i];
      if (abs(col - r) <= band_size) band_count++;
    }
  }

  if (double(band_count) / nnz >= 0.3) {
    banded = true;
  }
  return banded;
}

void GrayOrdering::orderingFunction()
{
    int n_rows = this->getMatrix().getRowCount();
    int n_cols = this->getMatrix().getColCount();
    int n_nz = this->getMatrix().getNNZCount();
    int* ptrs = (int*) this->getMatrix().getPtr();
    int* ids = (int*)this->getMatrix().getInd();

    rowIPermutation = new unsigned[n_rows];
    colIPermutation = new unsigned[n_rows];
    for(int i = 0; i < n_rows; i++) {
        rowIPermutation[i] = colIPermutation[i] = i;
    }

    IDType *order = new IDType[n_rows]();

  int raise_to = 0;
  int adder = 0;
  int start_split_reorder, end_split_reorder;

  int last_row_nnz_count = 0;
  int threshold = 0;  // threshold used to set a bit in bitmap to 1
  bool decresc_grey_order = false;

  int group_count = 0;

  // Initializing row order
  std::vector<IDType> v_order;
  std::vector<IDType> sparse_v_order;
  std::vector<IDType> dense_v_order;

  // Splitting original matrix's rows in two submatrices
  IDType sparse_dense_split = 0;
  for (IDType i = 0; i < n_rows; i++) {
    if ((ptrs[i + 1] - ptrs[i]) <= nnz_threshold) {
      sparse_v_order.push_back(i);
      sparse_dense_split++;
    } else {
      dense_v_order.push_back(i);
    }
  }

  v_order.reserve(sparse_v_order.size() + dense_v_order.size());  // preallocate memory

  bool is_sparse_banded = is_banded(n_nz, n_cols, ptrs, ids, sparse_v_order);
  bool is_dense_banded = is_banded(n_nz, n_cols, ptrs, ids, dense_v_order);
  std::sort(sparse_v_order.begin(), sparse_v_order.end(),
            [&](int i, int j) -> bool {
              return (ptrs[i + 1] - ptrs[i]) <
                     (ptrs[j + 1] - ptrs[j]);
            });  // reorder sparse matrix into nnz amount

  // the bit resolution determines the width of the bitmap of each row
  if (n_rows < bit_resolution) {
    bit_resolution = n_rows;
  }

  int row_split = n_rows / bit_resolution;

  auto nnz_per_row_split = new IDType[bit_resolution];
  auto nnz_per_row_split_bin = new IDType[bit_resolution];

  unsigned long decimal_bit_map = 0;
  unsigned long dec_begin = 0;
  int dec_begin_ind = 0;

  std::vector<row_grey_pair> reorder_section;  // vector that contains a section to be reordered

  if (!is_sparse_banded) {  // if banded just row ordering by nnz count is
    // enough, else do bitmap reordering in groups

    for (int i = 0; i < sparse_v_order.size(); i++) {  // sparse sub matrix if not highly banded
      if (i == 0) {
        last_row_nnz_count = ptrs[sparse_v_order[i] + 1] - ptrs[sparse_v_order[i]];  // get nnz count in first
        // row
        start_split_reorder = 0;
      }  // check if nnz amount changes from last row
      if ((ptrs[sparse_v_order[i] + 1] -  ptrs[sparse_v_order[i]]) == 0) {  // for cases where rows are empty
        start_split_reorder = i + 1;
        last_row_nnz_count = ptrs[sparse_v_order[i + 1] + 1] - ptrs[sparse_v_order[i + 1]];
        continue;
      }

      // reset bitmap for this row
      for (int j = 0; j < bit_resolution; j++) nnz_per_row_split[j] = 0;
      for (int j = 0; j < bit_resolution; j++) nnz_per_row_split_bin[j] = 0;

      // get number of nnz in each bitmap section
      for (int k = ptrs[sparse_v_order[i]]; k < ptrs[sparse_v_order[i] + 1]; k++) {
        nnz_per_row_split[ids[k] / row_split]++;
      }

      // get bitmap of the row in decimal value (first rows are less significant
      // bits)
      decimal_bit_map = 0;
      for (int j = 0; j < bit_resolution; j++) {
        adder = 0;
        if (nnz_per_row_split[j] > threshold) {
          nnz_per_row_split_bin[j] = 1;
          raise_to = j;
          adder = pow(2, raise_to);

          decimal_bit_map = decimal_bit_map + adder;
        }
      }

      // if number of nnz changed from last row, increment group count, which
      // might trigger a reorder of the group
      if ((i != 0) && (last_row_nnz_count != (ptrs[sparse_v_order[i] + 1] - ptrs[sparse_v_order[i]]))) {
        group_count = group_count + 1;

        // update nnz count for current row
        last_row_nnz_count = ptrs[sparse_v_order[i] + 1] - ptrs[sparse_v_order[i]];

        // if group size achieved, start reordering section until this row
        if (group_count == group_size) {
          end_split_reorder = i;
          // start next split the split for processing

          // process and reorder the reordered_matrix array till this point
          // (ascending or descending alternately)
          if (!decresc_grey_order) {
            sort(reorder_section.begin(), reorder_section.end(), asc_comparator);
            decresc_grey_order = !decresc_grey_order;
          } else {
            sort(reorder_section.begin(), reorder_section.end(), desc_comparator);
            decresc_grey_order = !decresc_grey_order;
          }

          dec_begin = reorder_section[0].second;
          dec_begin_ind = start_split_reorder;

          // apply reordered
          for (int a = start_split_reorder; a < end_split_reorder; a++) {
            if ((dec_begin != reorder_section[a - start_split_reorder].second) && (a < 100000)) {
              // print_dec_in_bin(bin_to_grey(dec_begin));

              dec_begin = reorder_section[a - start_split_reorder].second;
              dec_begin_ind = a;
            }

            sparse_v_order[a] = reorder_section[a - start_split_reorder].first;
          }

          start_split_reorder = i;
          reorder_section.clear();
          group_count = 0;
        }
      }

      reorder_section.push_back(row_grey_pair(sparse_v_order[i], grey_bin_to_dec(decimal_bit_map)));

      // when reaching end of sparse submatrix, reorder section
      if (i == sparse_v_order.size() - 1) {
        end_split_reorder = sparse_v_order.size();
        if (!decresc_grey_order) {
          sort(reorder_section.begin(), reorder_section.end(), asc_comparator);
          decresc_grey_order = !decresc_grey_order;
        } else {
          sort(reorder_section.begin(), reorder_section.end(), desc_comparator);
          decresc_grey_order = !decresc_grey_order;
        }
        for (int a = start_split_reorder; a < end_split_reorder; a++) {
          sparse_v_order[a] = reorder_section[a - start_split_reorder].first;
        }
      }
    }

    reorder_section.clear();
  }

  if (!is_dense_banded) {
    for (int i = 0; i < dense_v_order.size(); i++) {
      // if first row, establish the nnz amount, and starting index
      for (int j = 0; j < bit_resolution; j++) nnz_per_row_split[j] = 0;

      for (int k = ptrs[dense_v_order[i]]; k < ptrs[dense_v_order[i] + 1]; k++) {
        nnz_per_row_split[ids[k] / row_split]++;
      }
      threshold = (ptrs[dense_v_order[i] + 1] - ptrs[dense_v_order[i]]) / bit_resolution;  // floor
      decimal_bit_map = 0;
      for (int j = 0; j < bit_resolution; j++) {
        adder = 0;
        if (nnz_per_row_split[j] > threshold) {
          raise_to = j;  // row 0 = lowest significant bit
          adder = pow(2, raise_to);

          decimal_bit_map = decimal_bit_map + adder;
        }
      }
      reorder_section.push_back(row_grey_pair(dense_v_order[i], grey_bin_to_dec(decimal_bit_map)));
    }
    std::sort(reorder_section.begin(), reorder_section.end(), asc_comparator);

    for (int a = 0; a < dense_v_order.size(); a++) {
      dense_v_order[a] = reorder_section[a].first;
    }

    reorder_section.clear();
  }

  v_order.insert(v_order.end(), sparse_v_order.begin(), sparse_v_order.end());
  v_order.insert(v_order.end(), dense_v_order.begin(), dense_v_order.end());

  /*This order array stores the inverse permutation vector such as order[0] =
   * 243 means that row 0 is placed at the row 243 of the reordered matrix*/
  // std::vector<IDType> v_order_inv(n_rows);
  for (int i = 0; i < n_rows; i++) {
    rowIPermutation[v_order[i]] = i;
  }
}
