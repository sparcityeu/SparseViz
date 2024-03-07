#include "BOBAOrdering.h"
#include <vector>
#include <unordered_set>
#include <queue>
#include "Parameters.h"

void BOBAOrdering::applyOrdering(unsigned* order) {
  unsigned n = this->getMatrix().getRowCount();
  for(unsigned i = 0; i < n; i++) {
    rowIPermutation[order[i]] = i;
  }
}

void BOBAOrdering::orderingFunction() {
  unsigned n = this->getMatrix().getRowCount();
  unsigned m = this->getMatrix().getColCount();

  unsigned* ptrs = this->getMatrix().getPtr();
  unsigned* ids = this->getMatrix().getInd();

  rowIPermutation = new unsigned[n];
  colIPermutation = new unsigned[n];
  for(unsigned i = 0; i < n; i++) rowIPermutation[i] = i;
  for(unsigned i = 0; i < m; i++) colIPermutation[i] = i;

  unsigned nodes = 0;
  unsigned nnzs = this->getMatrix().getNNZCount();

  if (n >= m) {
    nodes = n;
  } else {
    nodes = m;
  }

  unsigned *order = new unsigned[nodes];

  int k = 0;
  std::vector<std::pair<unsigned, unsigned>> cooMatrix(nnzs);
  for(unsigned i = 0; i < n; i++) {
    for (int ptr = ptrs[i]; ptr < ptrs[i+1]; ptr++) {
        cooMatrix[i] = std::make_pair(i, ids[ptr]);
    }
  }

  /* Sort the matrix based on columns */
  std::sort(cooMatrix.data(), cooMatrix.data() + cooMatrix.size(), [](const auto& a, const auto& b) {
    return (a.second != b.second) ? (a.second < b.second) : (a.first < b.first);
  });

  if (m_Execution == "SEQ") { //params_->sequential) {
    /* Goes through the values in the I array (rows) */
    std::unordered_set<unsigned> copiedValues;
    for (unsigned i = 0; i < nnzs; i++) {
      unsigned element_i = cooMatrix[i].first;
      if (copiedValues.find(element_i) == copiedValues.end()) {
        order[k++] = element_i;
        copiedValues.insert(element_i);
      }
    }
    if (k == nodes) {
      applyOrdering(order);
    } else {
        for (int i = 0; i < nnzs; i++) {
            unsigned element_j = cooMatrix[i].second;
            if (copiedValues.find(element_j) == copiedValues.end()) {
                order[k++] = element_j;
                copiedValues.insert(element_j);
            }
        }
        if (k == nodes) {
          applyOrdering(order);
        } else {
            /* Goes through the single nodes that don't have any edge */
            for (unsigned i = 0; i < nodes; i++) {
                if (copiedValues.find(i) == copiedValues.end()) {
                    order[k++] = i;
                    copiedValues.insert(i);
                }
            }
            applyOrdering(order);
        }
    }
  } else {
    std::priority_queue<std::pair<unsigned, unsigned>, std::vector<std::pair<unsigned, unsigned>>, std::greater<std::pair<unsigned, unsigned>>> PQ;

    for (unsigned i = 0; i < nodes; i++) {order[i] = nnzs*2;}

    /* Goes through the values in the I++J array */
#pragma omp parallel for schedule(dynamic, 256)
    for (unsigned i = 0; i < nnzs*2; i++) {
      if ((i < nnzs) && (i < order[cooMatrix[i].first]))
        order[cooMatrix[i].first] = i;
      else if ((i >= nnzs) && (i < order[cooMatrix[i - nnzs].second]))
        order[cooMatrix[i - nnzs].second] = i;
    }

    /* Places vertices by order of appearence */
    for (unsigned i = 0; i < nodes; i++) {
      PQ.push(std::make_pair(order[i], i));
    }
    for (unsigned i = 0; i < nodes; i++) {
     
      rowIPermutation[PQ.top().second] = i;
   
      PQ.pop();
    }
  }
}
