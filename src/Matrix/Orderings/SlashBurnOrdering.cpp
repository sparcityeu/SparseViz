//                                                                                                                                                                                                          
// Created on on 16 September 2023                                                                                                                                                                
//                                                                                                                                                                                                          

#include "SlashBurnOrdering.h"
#include <vector>
#include "Parameters.h"

#include <queue>
#include <stack>


/*SQ*/
/* 1ºst imp - greedy k-hub selection, update degree of nodes */
/* after removing one of the nodes.                          */
/* 2ºnd imp - order spokes by hube order and not by size     */

 /*params_struct->k_size = k_size;
  params_struct->greedy = greedy;
  params_struct->hub_order = hub_order;*/

unsigned* SlashBurnOrdering::computeDegree(unsigned* rptr , unsigned* col,  unsigned n, unsigned *v_flag, unsigned level) {

  unsigned *degree = new unsigned[n];

  /* Compute the degree of all nodes in the GCC */
  for (unsigned i = 0; i < n; i++) {
    degree[i] = 0;
    if (v_flag[i] == level) {
      for (unsigned ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
        unsigned node_id = col[ptr];
        if (v_flag[node_id] == level) degree[i]++;
      }
    } else {
      degree[i] = -1;
    }
  }
  return degree;
}

unsigned* SlashBurnOrdering::removeKHubsetGreedy(unsigned *rptr, unsigned *col, unsigned n, unsigned k, 
                                                 unsigned *v_flag, unsigned *order, unsigned *degree,
                                                 unsigned level, unsigned min_id) {
                                                                  
  std::priority_queue<std::pair<unsigned, unsigned>, std::vector<std::pair<unsigned, unsigned>>, std::greater<std::pair<unsigned, unsigned>>> PQ;
  unsigned *k_hub = new unsigned[k];

  /* Removes the highest degree node k times*/
  for (unsigned i = 0; i < k; i++) {
    /* Finds node of highest degree */
    PQ.push(std::make_pair(degree[0], 0));
    for (unsigned i = 0; i < n; i++) {
      if (v_flag[i] == level) {
        if (degree[i] > PQ.top().first) {
            PQ.pop();
            PQ.push(std::make_pair(degree[i], i));
        }
      }
    }
    
    /* Remove highest degree node from the graph (flag = 0) */
    unsigned u = PQ.top().second;
    order[min_id + i] = u;
    v_flag[u] = 0;
    k_hub[i] = u;
    degree[u] = -1;
    PQ.pop();
    
    /* Update degree of nodes */
    for (unsigned ptr = rptr[u]; ptr < rptr[u + 1]; ptr++) {
      unsigned node_id = col[ptr];
      if (v_flag[node_id] == level) degree[node_id]--;
    }
  }

  return k_hub;
}

unsigned* SlashBurnOrdering::removeKHubset( unsigned *rptr, unsigned *col, unsigned n, unsigned k, 
                                            unsigned *v_flag, unsigned *order, unsigned level, unsigned min_id) {

  std::priority_queue<std::pair<unsigned, unsigned>, std::vector<std::pair<unsigned, unsigned>>, std::greater<std::pair<unsigned, unsigned>>> PQ;
  unsigned *k_hub = new unsigned[k];
  unsigned i = 0, j = 0, qwp1 = 0;

  /* Place first k nodes in the stack */
  for (i = 0; i < n; i++) {
    if (v_flag[i] == level) {
      unsigned degree = 0;
      for (unsigned ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
        unsigned node_id = col[ptr];
        if (v_flag[node_id] == level) degree++;
      }
      PQ.push(std::make_pair(degree, i));
      j++;
    }
    if (j == k) break;
  }
  /* Check rest of the nodes to find high degree ones */
  for (i = i+1; i < n; i++) {
    if (v_flag[i] == level) {
      unsigned degree = 0;
      for (unsigned ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
        unsigned node_id = col[ptr];
        if (v_flag[node_id] == level) degree++;
      }
      if (degree > PQ.top().first) {
          PQ.pop();
          PQ.push(std::make_pair(degree, i));
      }
    }
  }
  /* Remove them from the graph (flag = 0) */
  while (!PQ.empty()) {
    unsigned node_id = PQ.top().second;
    order[min_id + k - 1 - qwp1] = node_id;
    v_flag[node_id] = 0;
    k_hub[k - 1 - qwp1] = node_id;
    qwp1++;
    PQ.pop();
  }
  return k_hub;
}

unsigned SlashBurnOrdering::findCC( unsigned* rptr,  unsigned* col, 
                                    unsigned *v_flag, unsigned level,
                                    unsigned root) {

  unsigned cc_count = 1;
  std::stack<unsigned> DFS;

  DFS.push(root);
  v_flag[root] = level + 1;
  
  /* Goes through all the nodes in the connected component */
  while (!DFS.empty()) {
    unsigned u = DFS.top();
    DFS.pop();
    
    for (unsigned ptr = rptr[u]; ptr < rptr[u + 1]; ptr++) {
      unsigned node_id = col[ptr];
      if (v_flag[node_id] == level) {
        DFS.push(node_id);
        v_flag[node_id] = level + 1;
        cc_count++;
      } 
    }
  }
  return cc_count;
}

unsigned SlashBurnOrdering::orderCC(unsigned* rptr,  unsigned* col, 
                                    unsigned *v_flag, unsigned *order,
                                    unsigned level, unsigned root,
                                    unsigned max_id) {
  unsigned qwp2 = 0;
  std::queue<unsigned> DFS;

  DFS.push(root);
  order[max_id - qwp2] = root;
  v_flag[root] = -level;
  qwp2++;

  /* Goes through all the nodes in the connected component and orders them */
  while (!DFS.empty()) {
    unsigned u = DFS.front();
    DFS.pop();
    for (unsigned ptr = rptr[u]; ptr < rptr[u + 1]; ptr++) {
      unsigned node_id = col[ptr];
      if (v_flag[node_id] == level + 1) {
        v_flag[node_id] = -level;
        DFS.push(node_id);
        order[max_id - qwp2] = node_id;
        qwp2++;
      }
    }
  }
  return qwp2;
}

void SlashBurnOrdering::slashloop(  unsigned *rptr, unsigned *col, unsigned n, unsigned k, 
                                    unsigned *v_flag, unsigned *order, unsigned level, unsigned max_id) {

  std::priority_queue<  std::tuple<unsigned, unsigned, unsigned>, 
                        std::vector<std::tuple<unsigned, unsigned, unsigned>>, 
                        std::greater<std::tuple<unsigned, unsigned, unsigned>>> PQ_CC_hub;
  unsigned cmp_counter = 0;
  unsigned *k_hub = NULL;

  while (true) {

    /* Removes k-hubset */
    if (GREEDY) {
      unsigned *degree = computeDegree(rptr, col, n, v_flag, level);
      k_hub = removeKHubsetGreedy(rptr, col, n, k, v_flag, order, degree, level, (level-2)*k);
    } else {
      k_hub = removeKHubset(rptr, col, n, k, v_flag, order, level, (level-2)*k);
    }
    unsigned gcc_count = 0, gcc_id = -1;
    cmp_counter = 0;

    /* Finds strong connected components */
    for (int i = k - 1; i >= 0; i--) {
      unsigned u = k_hub[i];  
      for (unsigned ptr = rptr[u]; ptr < rptr[u + 1]; ptr++) {
        unsigned node_id = col[ptr];
        if (v_flag[node_id] == level) {
          unsigned n_cc = findCC(rptr, col, v_flag, level, node_id);
          if (n_cc > gcc_count) {
            gcc_count = n_cc;
            gcc_id = node_id;
          }
          if (HUB_ORDER) {
            PQ_CC_hub.push(std::make_tuple(i, n_cc, node_id));
          } else {
            PQ_CC_hub.push(std::make_tuple(0, n_cc, node_id));
          }
          cmp_counter++;
        }
      }
    }
    /* When there are no more nodes left to reorder */
    if (cmp_counter == 0)
      break;
    
    /* Places spokes in the permutation vector */
    for (unsigned i = 0; i < cmp_counter; i++) {
      std::tuple<unsigned, unsigned, unsigned> root = PQ_CC_hub.top();
      if (std::get<2>(root) == gcc_id) {
        PQ_CC_hub.pop();
        continue;
      }
      PQ_CC_hub.pop();
      unsigned n_cc = orderCC(rptr, col, v_flag, order, level, std::get<2>(root), n-1-max_id);
      max_id += n_cc;
    }

    /* Checks size of GCC */
    if (gcc_count < k) {
      unsigned n_cc = orderCC(rptr, col, v_flag, order, level, gcc_id, n-1-max_id);
      break;
    } else {
      level++;
    }
  }
  return;
}
using namespace std;
void SlashBurnOrdering::orderingFunction() {

  std::priority_queue<std::pair<unsigned, unsigned>, std::vector<std::pair<unsigned, unsigned>>, std::greater<std::pair<unsigned, unsigned>>> PQ;
  unsigned *rptr = this->getMatrix().getPtr();
  unsigned *col = this->getMatrix().getInd();
  unsigned nodes = this->getMatrix().getRowCount();

  unsigned *t_row = new unsigned[nodes + 1];
  unsigned *t_count = new unsigned[nodes];
  unsigned *t_col = new unsigned[this->getMatrix().getNNZCount()];

  unsigned *order = new unsigned[nodes];
  unsigned *order2 = new unsigned[nodes];
  unsigned *v_flag = new unsigned[nodes];

  rowIPermutation = new unsigned[nodes];
  colIPermutation = new unsigned[nodes];
  unsigned cmp_counter = 0, max_id = 0;

  for (unsigned i = 0; i < nodes; i++) {
    v_flag[i] = 1;
    t_count[i] = 0;
  }

  /* Create CSC format matrix of the input CSR */
  for (unsigned i = 0; i < this->getMatrix().getNNZCount(); i++) {
    unsigned col_id = col[i];
    t_count[col_id]++;
  }

  t_row[0] = 0;
  for (unsigned i = 1; i < nodes; i++) {
    t_row[i] = t_row[i-1] + t_count[i-1];
  }
  t_row[nodes] = this->getMatrix().getNNZCount();

  for (unsigned i = 0; i < nodes; i++) {
    for (unsigned ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
      unsigned node_id = col[ptr];
      t_col[t_row[node_id] + t_count[node_id] - 1] = i;
      t_count[node_id]--;
    }
  }

  /* Create the symmetric version of the input matrix */
  unsigned *last_row = new unsigned[nodes + 1];
  unsigned *last_col = new unsigned[this->getMatrix().getNNZCount() * 2];
  unsigned *s_flag = new unsigned[nodes]();
  unsigned last_c = 0;

  for (unsigned i = 0; i < nodes; i++) {
    for (unsigned ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
      unsigned node_id = col[ptr];
      last_col[last_c] = node_id;
      s_flag[node_id] = i+1;
      last_c++;
    }
    for (unsigned ptr = t_row[i]; ptr < t_row[i + 1]; ptr++) {
      unsigned node_id = t_col[ptr];
      if (s_flag[node_id] != i+1) {
        last_col[last_c] = node_id;
        last_c++;
      }
    }
    last_row[i + 1] = last_c;
  }

  last_row[0] = 0;

  /* Free the auxiliary CSC format matrix*/
  delete[] s_flag;
  delete[] t_row;
  delete[] t_count;
  delete[] t_col;

  /* Orders the spokes in the original graph */
  for (unsigned i = 0; i < nodes; i++) {
    if (v_flag[i] == 1) {
      unsigned n_cc = findCC(last_row, last_col, v_flag, 1, i);
      PQ.push(std::make_pair(n_cc, i));
      cmp_counter++;
    }
  }
  for (unsigned i = 0; i < cmp_counter - 1; i++) {
    if(PQ.size() == 0)
      break;
    unsigned root = PQ.top().second;
    PQ.pop();
    unsigned n_cc = orderCC(last_row, last_col, v_flag, order, 1, root, nodes-1-max_id);
    max_id += n_cc;
  }

  if (PQ.size() > 0 && PQ.top().first < KSIZE) {
    unsigned root = PQ.top().second;
    PQ.pop();
    unsigned n_cc = orderCC(last_row, last_col, v_flag, order, 1, root, nodes-1-max_id);
  } else {
    PQ.pop();
    /* Call the slashburn loop algorithm to order the GCC */
    slashloop(last_row, last_col, nodes, KSIZE, v_flag, order, 2, max_id);
  }

  /* Convert to the sparsebase permutation format */
  for (unsigned i = 0; i < nodes; i++) {
    rowIPermutation[order[i]] = i;
    colIPermutation[order[i]] = i;
  }

  /* Free rest of allocated memory */
  delete[] order;
  delete[] v_flag;
}
