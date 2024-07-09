
//                                                                                                                                                                                                          
// Created on 22 December 2023                                                                                                                                                                
//                                                                                                                                                                                                          

#include "MinDegOrdering.h"
#include "Parameters.h"
#include <vector>
#include <algorithm>
#include <omp.h>

void MinDegOrdering::orderingFunction() {
  unsigned n = this->getMatrix().getRowCount();
  vType* ptrs = this->getMatrix().getPtr();
  vType* ids = this->getMatrix().getInd();
  int DEG_THRESHOLD = 200;//(ptrs[n] / n) * 10;
  //if(DEG_THRESHOLD > 250) DEG_THRESHOLD = 250;

  rowIPermutation = new vType[n];
  colIPermutation = new vType[n];

  int min_deg = n;
  for(int i = 0; i < n; i++) {
    for(int p = ptrs[i]; p < ptrs[i+1]; p++) {		
      if(ptrs[i+1] - ptrs[i] < min_deg) min_deg = ptrs[i+1] - ptrs[i];
    }
  }

  Node* info =  new Node[n];

  int* clq_starts = new int[n];
  int* clq_ends = new int[n];
  std::vector<int> clq_ids;
  clq_ids.reserve((int)(n * log2(n)));

  std::vector< std::vector<int> > vtxes(n + 1);
  for(int i = n-1; i >= 0; i--) {
    int deg = ptrs[i+1] - ptrs[i];
    vtxes[deg].push_back(i);  
    info[i].cdeg = deg;
    info[i].cloc = vtxes[deg].size() - 1;
  }     

  int* dense = new int[n];
  for(int i = 0; i < n; i++) dense[i] = -1;
  int dense_marker = 0;

  double starttime = omp_get_wtime();
  int cdeg = min_deg;
  for(int i = 0; i < n; i++) {//that many iterations will be performed - every row must be permuted
    while(cdeg <= n && vtxes[cdeg].size() == 0) cdeg++; //find the next full entry
    //if(cdeg >= DEG_THRESHOLD) break;
    if(i == 100) break;
    int vtx = vtxes[cdeg].back(); //the last vertex in the vector
    vtxes[cdeg].pop_back(); //popped back from vtxes 
    info[vtx].cdeg = -1; //removed from the vtxes so update info accordingly
    rowIPermutation[vtx] = colIPermutation[vtx] = i; //permute it as the next row

    //generate the clique for this elimination. 
    clq_starts[i] = clq_ids.size(); //new clique starts from start
    for(int p = ptrs[vtx]; p < ptrs[vtx+1]; p++) { //add the normal edges first
      int nbr = ids[p]; //actual neighbor
      if(info[nbr].cdeg != -1) { //nbr is not permuted yet
        dense[nbr] = dense_marker; //mark it as added to the clique
        clq_ids.push_back(nbr);
        info[nbr].cliqueIDs.push_back(i);
      }
    }

    for(int cid : info[vtx].cliqueIDs) { //now lets traverse the cliques of vtx
      if(cid < 0) cid = -cid;
      for(int p = clq_starts[cid]; p < clq_ends[cid]; p++) {  //cid is the current clique id
        int nbr = clq_ids[p];
        if(nbr == vtx) {//remove vtx from the clique (move the last vertex in the clique to here) this
          //this branch must be taken exactly once
          clq_ids[p] = clq_ids[clq_ends[cid] - 1];
          clq_ends[cid]--;
          p--; //set p-- to revisit this node again (note that clq_ends also decreases so loop will terminat even if this is the last vertex in the clique ) 
          continue;
        } else if(dense[nbr] != dense_marker) { //if nbr is not added to the clique before
          dense[nbr] = dense_marker; //mark nbr as added
          clq_ids.push_back(nbr); //add nbr to clique
          info[nbr].cliqueIDs.push_back(i); //add clique to nbr's cliques
        } 
      }
    }  
    clq_ends[i] = clq_ids.size(); //finish the new clique
    if(i % 1 == 0) {
      double time_until = omp_get_wtime() - starttime;
      std::cout << "Vertex: " << i << " is " << vtx << " - cdeg: " << cdeg << " clique size: " << clq_ends[i] - clq_starts[i] << " " << time_until << std::endl;
    }

    dense_marker++;
    if(dense_marker == (1 << 30)) {
        for(int i = 0; i < n; i++) dense[i] = -1;
        dense_marker = 0;
    }

    //now degree updates must be done for every vertex in the new clique. 
    for(int p = clq_starts[i]; p < clq_ends[i]; p++) {    
      int vtx_to_update = clq_ids[p]; 

      int _nvisited = 0;
      int degree = 0;
      //start with the original edges
      for(int p2 = ptrs[vtx_to_update]; p2 < ptrs[vtx_to_update + 1]; p2++) { //add the normal edges first
        int nbr = ids[p2]; //an original neighbor of vtx_to_update
        _nvisited++;
        if(info[nbr].cdeg != -1) { //nbr is not permuted yet
          dense[nbr] = dense_marker; //mark it as seen
          degree++;
        }
      }
     
      //now continue with cliques for vtx_to_update
      int _nchecked_cliques = 0;
      for(int clq = info[vtx_to_update].cliqueIDs.size() - 1; clq >= 0; clq--) { 
        int cid = info[vtx_to_update].cliqueIDs[clq];
        if(cid < 0) continue;
        _nchecked_cliques++;
        bool added = false;
        for(int p = clq_starts[cid]; p < clq_ends[cid]; p++) {  //cid is the clique id
          int nbr = clq_ids[p];
          _nvisited++;

          if(nbr == vtx_to_update) {
            continue;
          } else if(dense[nbr] != dense_marker) { //if nbr has not been seen before
            dense[nbr] = dense_marker; //mark nbr as added
            degree++;
            added = true; //this clique contributes to the degree count
          } 
        }
        if(added == false) { //this clique is not important for vtx_to_update since the previous cliques handle it so skip it in next iterations
          info[vtx_to_update].cliqueIDs[clq] = -info[vtx_to_update].cliqueIDs[clq];
        } 
      }  
      //std::cout << i << " " << _nvisited << " " << degree << " " << info[vtx_to_update].cliqueIDs.size() << " " << _nchecked_cliques << std::endl;
      int old_degree = info[vtx_to_update].cdeg;
      if(degree != old_degree) {
        //a move is required to put vtx_to_update to the correct place
        //remove it from its vector
        int last = vtxes[old_degree].back();
        vtxes[old_degree].pop_back();

        if(last != vtx_to_update) { //nbr is not the last node so move the last to its location
          int nbrloc = info[vtx_to_update].cloc;
          vtxes[old_degree][nbrloc] = last; 
          info[last].cloc = nbrloc;
        }

        //move it to its new location and update the location
        vtxes[degree].push_back(vtx_to_update);
        info[vtx_to_update].cdeg = degree;
        info[vtx_to_update].cloc = vtxes[degree].size() - 1;
        if(degree < cdeg) { //degree is updated so if necessary cdeg
          cdeg = degree;
        }
      }

      dense_marker++;
      if(dense_marker == (1 << 30)) {
        for(int i = 0; i < n; i++) dense[i] = -1;
        dense_marker = 0;
      }
    }
  }

  dense_marker++;
  if(dense_marker == (1 << 30)) {
    for(int i = 0; i < n; i++) dense[i] = -1;
    dense_marker = 0;
  }
  int counter = 0;
  for(int i = 0; i < n; i++) {
    if(rowIPermutation[i] != colIPermutation[i]) std::cout << "PRoblem - not symmetric " << std::endl;
    if(dense[rowIPermutation[i]] != dense_marker) {
      counter++;
      dense[rowIPermutation[i]] = dense_marker;
    }
  }
  std::cout << "Counter " << counter << " " << n << std::endl;

  /*


  int perm_position_from_back = n-1;
  for(int i = n-1; i > 0; i--) {
    for(int v : vtxes[i]) {
      if(info[v].cdeg != -1) {
          rowIPermutation[v] = colIPermutation[v] = perm_position_from_back--; //permute it as the next row
      }
    }
  }
   */

  delete [] info;
  delete [] clq_starts;
  delete [] clq_ends;
  delete [] dense;     
}