//                                                                                                                                                                                                          
// Created on 22 September 2023                                                                                                                                                                
//                                                                                                                                                                                                          

#include "RCMOrdering.h"
#include <vector>
#include "Parameters.h"

using namespace std;
bool RCMOrdering::findRoot (    unsigned& root, unsigned* queue,
                                int* level, int& max_level, unsigned& ccsize) {
  unsigned* ptrs = this->getMatrix().getPtr();
  unsigned* ids = this->getMatrix().getInd();
  unsigned n = this->getMatrix().getRowCount();
  for(int i = 0; i < n; i++) level[i] = -1;
  
  queue[0] = root;
  level[root] = 0;
  unsigned qs = 0;
  unsigned qe = 1;

  unsigned clevel = 0;
  while(qs < qe) {
    unsigned v = queue[qs++];
    unsigned clevel = level[v];
    for(unsigned ptr = ptrs[v]; ptr < ptrs[v+1]; ptr++) {
      unsigned nbr = ids[ptr];
      if(level[nbr] == -1) {
        level[nbr] = clevel + 1;
        queue[qe++] = nbr;
      }
    }
  }

  ccsize = qe;
  if(ccsize == 1) {
    max_level = 0;
    return false;
  }

  int final_level = level[queue[qe-1]];
  if(final_level <= max_level) {
    return false;
  } else {
    max_level = final_level;

    root = queue[qe - 1];
    unsigned i = qe - 2;
    while(i < n && level[queue[i]] == final_level) {
      unsigned v = queue[i];
      if((ptrs[v+1] - ptrs[v]) < (ptrs[root + 1] - ptrs[root])) {
        root = v;
      }
      i--;
    }   
    return true;
  }
}

void RCMOrdering::rcm (   unsigned& root, unsigned* queue,
                          double* sort_values,
                          bool* is_permuted) {
  unsigned* ptrs = this->getMatrix().getPtr();
  unsigned* ids = this->getMatrix().getInd();
  unsigned n = this->getMatrix().getRowCount();
  
  queue[0] = root;
  is_permuted[root] = true;

  unsigned qs = 0;
  unsigned qe = 1;
  while(qs < qe) {
    unsigned v = queue[qs++];
    unsigned start_v = qe;

    for(unsigned ptr = ptrs[v]; ptr < ptrs[v+1]; ptr++) {
      unsigned nbr = ids[ptr];

      if(!is_permuted[nbr]) {
        queue[qe++] = nbr;
        is_permuted[nbr] = true;
      } 
    }

    //sort the newly added vertices - can be better for larger lists
    for(unsigned i = start_v; i < qe; i++) {
      int j = i;
      while(j > start_v && sort_values[queue[j]] > sort_values[queue[j-1]]) {
          int temp = queue[j];
          queue[j] = queue[j-1];
          queue[j-1] = temp;
          j--;
      }
    }
  }

  return;
}

void RCMOrdering::orderingFunction() {
  unsigned n = this->getMatrix().getRowCount();
  unsigned* ptrs = this->getMatrix().getPtr();
  unsigned* ids = this->getMatrix().getInd();

  rowIPermutation = new unsigned[n];
  colIPermutation = new unsigned[n];

  unsigned* queue = new unsigned[n];
  int* level = new int[n];
  double* sort_values = new double[n];
  bool* is_permuted = new bool [n];
  for (unsigned i = 0; i < n; i++) is_permuted[i] = false;

  if(m_SortingCriteria == "DEG") {
    for (unsigned i = 0; i < n; i++) sort_values[i] = ptrs[i + 1] - ptrs[i];
  } 
  
  unsigned total_permuted = 0;
  for (unsigned i = 0; i < n; i++) { //traverse over all vertices                                                                                                                                           
    if ( !is_permuted[i] ) { //if i is not in a previously seen component                                                                                                                                     
      //std::cout << "start with " << i << std::endl;
      //this will find the peripheral vertex 
      unsigned root = i; //i is the first candidate 

      int current_level = -1;
      unsigned ccsize;     
      if(ptrs[i+1] == ptrs[i] || ((ptrs[i+1] == ptrs[i] + 1) && ids[ptrs[i]] == i)) {
        ccsize = 1;
        queue[total_permuted] = i;
        is_permuted[i] = true;
      } else {
        while(findRoot (root, queue + total_permuted, level, current_level, ccsize)) {
          //cout << "finding root " << root << " " << *level << " " << current_level << " " << ccsize << endl;
          if(!m_RootSelection) {
            root = i;
            break;
          }
        }

        //cout << root << " " << current_level <<  " " << ccsize <<  " -- " << ptrs[root] << " " << ptrs[root+1] << " " << ids[ptrs[root]] << endl;
        rcm (root, queue + total_permuted, sort_values, is_permuted); 
        //cout << "loop is done" << endl;
      }
     
      total_permuted += ccsize;    
    }
  }

  /*bool* marker = new bool[n];
  memset(marker, 0, sizeof(bool) * n);
  for(int i = 0; i < n; i++) {
    if(marker[queue[i]]) {
      std::cout << "duplicate " << i << " " << queue[i] << std::endl;
    } 
    marker[queue[i]] = true;
  }
  for(int i = 0 ; i < n; i++) {
    if(marker[i] == false) {
      std::cout << "missing " << i << std::endl;
    }
  }*/

  for(unsigned i = 0; i < n; i++) {
    colIPermutation[queue[i]] = n - i - 1;
    rowIPermutation[queue[i]] = n - i - 1;
  }

  delete [] queue;
  delete [] level;
  delete [] sort_values;
  delete [] is_permuted;
}


