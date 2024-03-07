//                                                                                                                                                                                                          
// Created on on 31 December 2023                                                                                                                                                                
//                                                                                                                                                                                                          

#include "DynaDegOrdering.h"
#include <vector>
#include "Parameters.h"

void DynaDegOrdering::orderingFunction() {
    int m = this->getMatrix().getRowCount();
    int* row_ptrs = (int*)this->getMatrix().getPtr();
    int* row_ids = (int*)this->getMatrix().getInd();

    int n = this->getMatrix().getColCount();
    int* col_ptrs;
    int* col_ids;
    int max_deg = 0; 
    int min_deg = m+n;
    int* degs = (int *) malloc(sizeof(int) * n);	
	memset(degs, 0, sizeof(int) * n);

	for(int i = 0; i < m; i++) {
	  for(int p = row_ptrs[i]; p < row_ptrs[i+1]; p++) {	
        if(row_ptrs[i+1] - row_ptrs[i] > max_deg) {
            max_deg = row_ptrs[i+1] - row_ptrs[i];
        }	
        if(row_ptrs[i+1] - row_ptrs[i] < min_deg) {
            min_deg = row_ptrs[i+1] - row_ptrs[i];
        }		
	    degs[row_ids[p]]++;
	  }
	}
    for(int i = 0; i < n; i++) {
        if(degs[i] > max_deg) {
            max_deg = degs[i];
        }
        if(degs[i] < min_deg) {
            min_deg = degs[i];
        }
    }

	col_ptrs = (int *) malloc(sizeof(int) * (n+1));
	col_ptrs[0] = 0;
	for(int i = 0; i < n; i++) {
		col_ptrs[i+1] = col_ptrs[i] + degs[i];
		degs[i] = col_ptrs[i];
	}  

	col_ids = (int*) malloc(sizeof(int) * col_ptrs[n]);	
	for(int i = 0; i < m; i++) {
		for(int p = row_ptrs[i]; p < row_ptrs[i+1]; p++) {
			col_ids[degs[row_ids[p]]++] = i;
		}
	}

    rowIPermutation = new unsigned[m];
    colIPermutation = new unsigned[n];

    Node* info =  new Node[m+n];
    std::vector< std::vector<int> > vtxes(max_deg + 1);
    for(int i = 0; i < m; i++) {
        int deg = row_ptrs[i+1] - row_ptrs[i];
        vtxes[deg].push_back(i);  
        info[i].cdeg = deg;
        info[i].cloc = vtxes[deg].size() - 1;
    }     
    for(int i = 0; i < n; i++) {
        int deg = col_ptrs[i+1] - col_ptrs[i];
        vtxes[deg].push_back(m + i);  
        info[m+i].cdeg = deg;
        info[m+i].cloc = vtxes[deg].size() - 1;
    }

    int cdeg = max_deg;
    if(m_Direction == "MIN") {
        cdeg = min_deg;
    } 
    int rpermuted = 0, cpermuted = 0; 
    for(int i = 0; i < m+n; i++) {//that many iterations 
        if(m_Direction == "MIN") {
            while(cdeg <= max_deg && vtxes[cdeg].size() == 0) cdeg++; //find the next full entry
        } else {
            while(cdeg >= 0 && vtxes[cdeg].size() == 0) cdeg--; //find the next full entry
        }
    
        int vtx = vtxes[cdeg].back();
        vtxes[cdeg].pop_back();

        if(vtx < m) { //the chosen one is row
            info[vtx].cdeg = -1; //removed from the vtxes so update info accordingly
            rowIPermutation[vtx] = rpermuted++; //permute it as the next row
            for(int p = row_ptrs[vtx]; p < row_ptrs[vtx+1]; p++) {
                int nbr = row_ids[p];
                int nbrdeg = info[m + nbr].cdeg;

                if(nbrdeg != -1) { //this column has not been permuted before so decrement its degree
                    //remove it from its vector
                    int last = vtxes[nbrdeg].back();
                    vtxes[nbrdeg].pop_back();

                    if(last != m + nbr) { //nbr is not the last node so move the last to its location
                        int nbrloc = info[m + nbr].cloc;
                        vtxes[nbrdeg][nbrloc] = last; 
                        info[last].cloc = nbrloc;
                    }

                    //move it to its new location and update the location
                    nbrdeg--;
                    vtxes[nbrdeg].push_back(m + nbr);
                    info[m + nbr].cdeg = nbrdeg;
                    info[m + nbr].cloc = vtxes[nbrdeg].size() - 1;
                    if(m_Direction == "MIN") {
                        if(nbrdeg < cdeg) {
                            cdeg = nbrdeg;
                        }
                    }
                }
            }
        } else {
            vtx = vtx - m; //chosen column
            info[m + vtx].cdeg = -1; //removed from the vtxes so update info accordingly
            colIPermutation[vtx] = cpermuted++; //permute it as the next column

            for(int p = col_ptrs[vtx]; p < col_ptrs[vtx+1]; p++) {
                int nbr = col_ids[p]; //neighbor row
                int nbrdeg = info[nbr].cdeg; //get its degree

                if(nbrdeg != -1) { //this is not permuted yet
                    //remove it from its vector
                    int last = vtxes[nbrdeg].back();
                    vtxes[nbrdeg].pop_back();

                    if(last != nbr) { //nbr is not the last node so move the last to its location
                        int nbrloc = info[nbr].cloc;
                        vtxes[nbrdeg][nbrloc] = last; 
                        info[last].cloc = nbrloc;
                    }

                    //move it to its new location and update the location
                    nbrdeg--;
                    vtxes[nbrdeg].push_back(nbr);
                    info[nbr].cdeg = nbrdeg;
                    info[nbr].cloc = vtxes[nbrdeg].size() - 1;
                    if(m_Direction == "MIN") {
                        if(nbrdeg < cdeg) {
                            cdeg = nbrdeg;
                        }
                    }
                }
            }
        }      
    }
    delete [] info;
    free(degs);
    free(col_ptrs);
    free(col_ids);
}


