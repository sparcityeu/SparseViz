//                                                                                                                                                                                                          
// Created on on 15 October 2023                                                                                                                                                                
//                                                                                                                                                                                                          

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "patoh.h"
#include "helpers.h"
#include "PatohOrdering.h"
#include "RCMOrdering.h"
#include "AMDOrdering.h"
#include "SlashBurnOrdering.h"
#include "DynaDegOrdering.h"
#include "BOBAOrdering.h"
#include "GrayOrdering.h"
#include <vector>
#include "Parameters.h"
#include <time.h>
#include <sys/time.h>
#include <unordered_map>
#include "amd.h"
#ifdef RABBIT_AVAILABLE
#include "RabbitOrdering.h"
#endif

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

omp_lock_t PatohOrdering::patoh_lock;


void PatohOrdering::wrapPaToH(	int _c, int _n, int *xpins, int *pins, int *cwghts, 
				int *nwghts, int *partvec, int no_cons, int patoh_no_parts) {

    	omp_set_lock(&patoh_lock);

		int i, ares;
		PaToH_Parameters args;
		int* partwghts;
		
		args._k = patoh_no_parts;
		if(_c + xpins[_n] > 100000000) {
			if(m_Metric == "CUT") {
				PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_SPEED );
			} else if(m_Metric == "CON") {
				PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_SPEED  );
			} else {
				std::cout << "Unknown metric for PatohOrdering : " << m_Metric << std::endl;
				exit(1);
			}
			args.doinitperm = 0; 
			args.MemMul_CellNet = (int) ceil(log2(1.0 * 4 * patoh_no_parts)); 
			args.MemMul_Pins = (int) ceil(log2(1.0 * 4 * patoh_no_parts)); 
			args.crs_coarsento = _c+1; 
			args.initp_alg = PATOH_IPA_GHGP; 
			args.initp_ghg_trybalance = 0; 
			args.ref_passcnt = 3; 
			args.initp_refalg = 0; 
			args.initp_runno = 1;
		} else {
			if(m_Metric == "CUT") {
				PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_DEFAULT );
			} else if(m_Metric == "CON") {
				PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT );
			} else {
				std::cout << "Unknown metric for PatohOrdering : " << m_Metric << std::endl;
				exit(1);
			}

			args.MemMul_Pins += 3;
			args.MemMul_CellNet += 3;
			args.init_imbal = 0.10;
			args.final_imbal = 0.10;
			args.ref_passcnt = 3;
			args.crs_coarsenper = 10;
		}

		ares = PaToH_Alloc(&args, _c, _n, no_cons, cwghts, nwghts, xpins, pins);
		if (ares) {
			printf("%s: wrapPaToH: error in allocating memory for PaToH %d\n", this->getMatrix().getName().c_str(), ares);
			fflush(stdout);
			exit(1);
		}

		int cut = 0;
		struct timeval tp;
		gettimeofday(&tp, NULL);
		args.seed = tp.tv_sec;
		partwghts = (int*) malloc(sizeof(int) * no_cons * patoh_no_parts);	

		PaToH_Part(&args, _c, _n, no_cons, 0, cwghts, nwghts, xpins, pins, NULL, partvec, partwghts, &cut);
		for(int i = 0; i < _c; i++) {
			if(partvec[i] == -1) partvec[i] = patoh_no_parts;
		}

		double avgwght = 0;
		for(i = 0; i < patoh_no_parts; i++) {
			avgwght += partwghts[i];
		}
		avgwght /= patoh_no_parts;

		double imbal = 0;
		for(i = 0; i < patoh_no_parts; i++) {
			imbal = max(imbal, ((double)(partwghts[i])/avgwght));
		}
		imbal -= 1;
		
		printf("%s: PaToH cut is %d\t_c=%d _n=%d\tpins=%d\timbal=%.3f\n", this->getMatrix().getName().c_str(), cut, _c, _n, xpins[_n], imbal);

		PaToH_Free();
		free(partwghts);
	    omp_unset_lock(&patoh_lock);
}

void PatohOrdering::innerOrder(int* ptrs, int* js, int m, int n, int* rpartv, int* cpartv) {
 	//Rows/columns are ordered - if an inner ordering is not enforced then we are done otherwise 
	//(1) generate symettric CRSs (for AMD, RCM, slashburn etc.) inner Patoh is not necessary, just use 2D partitioning with more parts 
	//(2) order these CRSs 
	//(3) and update the rowIPermutation and colIPermutation 

	std::vector<std::vector<int>> i_row_ids(m_NoParts);
	std::vector<std::vector<int>> i_col_ids(m_NoParts);

	//number of rows/columns in a part
	int* n_part_row = (int*)malloc(sizeof(int) * (m_NoParts));
	memset(n_part_row, 0, sizeof(int) * (m_NoParts));
	int* n_part_col = (int*)malloc(sizeof(int) * (m_NoParts)); 
	memset(n_part_col, 0, sizeof(int) * (m_NoParts));

	//mapping of real vertex ids to part vertex ids (rows/columns)
	int* row_part_ids =  (int*)malloc(sizeof(int) * m); 
	int* col_part_ids =  (int*)malloc(sizeof(int) * n); 
	for(int i = 0; i < m; i++) {
		if(rpartv[i] < m_NoParts) {
			row_part_ids[i] = n_part_row[rpartv[i]]++;
			i_row_ids[rpartv[i]].push_back(i);
		}
	}
	for(int i = 0; i < n; i++) {
		if(cpartv[i] < m_NoParts) {
			col_part_ids[i] = n_part_col[cpartv[i]]++;
			i_col_ids[cpartv[i]].push_back(i);
		}
	}
	
	std::vector<Entries> entries(m_NoParts + 1);
	for(int i = 0; i <= m_NoParts; i++) {
		entries[i].reserve(ptrs[m] / m_NoParts);
	}

	for(int i = 0; i < m; i++) {
		int row_part = rpartv[i];
		if(row_part >= m_NoParts) continue;
		int row_part_id = row_part_ids[i];
		for(int ptr = ptrs[i]; ptr < ptrs[i+1]; ptr++) {
			int nbr = js[ptr];
			if(cpartv[nbr] == row_part) {
				int nbr_part_id = col_part_ids[nbr];
				entry e(row_part_id, nbr_part_id);
				entries[row_part].push_back(e);
				if(row_part_id != nbr_part_id) {
					entry e2(nbr_part_id, row_part_id);
					entries[row_part].push_back(e2);
				}
			}
		}
	}

	int max_nnz = 0;
	for(int i = 0; i < m_NoParts; i++) {
		if(entries[i].size() > max_nnz) {
			max_nnz = entries[i].size();
		}
	}

	entry* sorted_entries = (entry*)malloc(sizeof(entry) * max_nnz);
	int maxmn = max(m,n);
	
	int* row_iperm = (int*)malloc(sizeof(int) * maxmn);
	int* col_iperm = (int*)malloc(sizeof(int) * maxmn);

	int *part_ptrs = (int*)malloc(sizeof(int) * (maxmn + 1));
	int *part_ids = (int*)malloc(sizeof(int) * max_nnz);
	int *last_ids = (int*)malloc(sizeof(int) * maxmn);

	SparseMatrix sp("Temporary", 0, 0, 0);
	sp.isSymmetric() = false;
   	sp.isPatternSymmetric() = false;
    sp.isSquare() = false;

	int* i_order = (int*) malloc(sizeof(int) * maxmn);

	int n_permuted_rows = 0;
	int n_permuted_cols = 0;
	for(int i = 0; i < m_NoParts; i++) {
		int matrix_dim = max(n_part_row[i], n_part_col[i]); 
		for(int j = 0; j < matrix_dim; j++) last_ids[j] = -1;

		memset(part_ptrs, 0, sizeof(int) * (matrix_dim + 1));
		for(auto e : entries[i]) part_ptrs[e.col + 1]++;
		for(int j = 1; j < matrix_dim; j++) part_ptrs[j] += part_ptrs[j - 1];
		for(auto e : entries[i]) {
			sorted_entries[part_ptrs[e.col]++] = e;
		}

		memset(part_ptrs, 0, sizeof(int) * (matrix_dim + 1));
		for(int j = 0; j < entries[i].size(); j++) {
			auto &e = sorted_entries[j];
			if(last_ids[e.row] != e.col) {
				part_ptrs[e.row + 1]++;
				last_ids[e.row] = e.col;
			} else {
				e.row = -1;
			}
		}
		for(int j = 1; j <= matrix_dim; j++) part_ptrs[j] += part_ptrs[j - 1];
		for(int j = 0; j < entries[i].size(); j++) {
			auto e = sorted_entries[j];
			if(e.row != -1) {
				part_ids[part_ptrs[e.row]++] = e.col;
			}
		}
		for(int j = matrix_dim; j > 0; j--) part_ptrs[j] = part_ptrs[j-1];
		part_ptrs[0] = 0;

		sp.isSquare() = sp.isPatternSymmetric() = true;
		sp.getRowCount() = sp.getColCount() = matrix_dim;
		sp.getNNZCount() = part_ptrs[matrix_dim];
		sp.getPtr() = (vType*) part_ptrs; //will not work if the byte sizes are not the same
		sp.getInd() = (vType*) part_ids;  //will not work if the byte sizes are not the same

 		MatrixOrdering* orderer = nullptr;
		if(m_Inner == "AMD") {
			orderer = new AMDOrdering(sp, "AMD_PatohPart_" + std::to_string(i));
		} else if(m_Inner == "RCM") {
			orderer = new RCMOrdering(sp, "RCM_PatohPart_" + std::to_string(i), "");
		} else if(m_Inner == "SBURN") {
			orderer = new SlashBurnOrdering(sp, "SBURN_PatohPart_" + std::to_string(i), "");
		}
#ifdef RABBIT_AVAILABLE
        else if(m_Inner == "RABBIT")
        {
			orderer = new RabbitOrdering(sp, "Rabbit - PatohPart " + std::to_string(i));
		}
#endif
        else if(m_Inner == "BOBA") {
			orderer = new BOBAOrdering(sp, "BOBA_PatohPart_" + std::to_string(i), "");
		} else if(m_Inner == "GRAY") {
			orderer = new GrayOrdering(sp, "GRAY _PatohPart_" + std::to_string(i), "");
		} else if(m_Inner == "DynaDeg") {
			orderer = new DynaDegOrdering(sp, "DynaDeg_PatohPart_" + std::to_string(i), "");
		} 
		
		if(orderer == nullptr) {
			#pragma omp critical 
			{
				throw std::runtime_error("Exception in PaToH Inter Ordering - Innter type is not defined");
			}
		}

		orderer->generateOrdering(false);

		const vType* row_order = orderer->getRowIPermutation();
		const vType* col_order = orderer->getColIPermutation();

		for(int i = 0; i < matrix_dim; i++) i_order[row_order[i]] = i;
		for(int j = 0; j < matrix_dim; j++) {
			if(i_order[j] < n_part_row[i]) {
				int org_row_id = i_row_ids[i][i_order[j]];
				rowIPermutation[org_row_id] = n_permuted_rows++;
			}
		}

		for(int i = 0; i < matrix_dim; i++) i_order[col_order[i]] = i;
		for(int j = 0; j < matrix_dim; j++) {
			if(i_order[j] < n_part_col[i]) {
				int org_col_id = i_col_ids[i][i_order[j]];
				colIPermutation[org_col_id] = n_permuted_cols++;
			}
		}
    	sp.getPtr() = nullptr;
    	sp.getInd() = nullptr;
	}

	free(i_order);
	free(part_ptrs);
	free(part_ids);
	free(row_part_ids);
	free(col_part_ids);
	free(n_part_row);
	free(n_part_col);
	free(last_ids);
}

void PatohOrdering::colNetOrder(int* ptrs, int* js, int m, int n, int* partv ) {
	int *degs, *xpins, *pins, *cwghts, *nwghts;
	int  i, p;

	degs = (int *)malloc(sizeof(int) * n);	
	memset(degs, 0, sizeof(int) * n);

	for(i = 0; i < m; i++) {
	  for(p = ptrs[i]; p < ptrs[i+1]; p++) {			
	    degs[js[p]]++;
	  }
	}

	nwghts = (int *) malloc(sizeof(int) * n);	
	for(i = 0; i < n; i++) nwghts[i] = 1;

	xpins = (int *) malloc(sizeof(int) * (n+1));
	xpins[0] = 0;
	for(i = 0; i < n; i++) {
		xpins[i+1] = xpins[i] + degs[i];
		degs[i] = xpins[i];
	}  

	pins = (int*) malloc(sizeof(int) * xpins[n]);	
	for(i = 0; i < m; i++) {
		for(p = ptrs[i]; p < ptrs[i+1]; p++) {
			pins[degs[js[p]]++] = i;
		}
	}

	cwghts = (int *) malloc(sizeof(int) * m);  
	for(i = 0; i < m; i++) {
		cwghts[i] = 1;
		if(m_CellWght == "NNZ") {
			cwghts[i] = ptrs[i+1] - ptrs[i];
		}
	}

	wrapPaToH(m, n, xpins, pins, cwghts, nwghts, partv, 1, m_NoParts);

    int* part_cnts = (int*) malloc(sizeof(int) * (2 * m_NoParts + 2)); 
	int* part_marker = (int*) malloc(sizeof(int) * (m_NoParts + 2));    
	int* cpartv = (int*) malloc(sizeof(int) * n);

    for(int i = 0; i <= m_NoParts + 1; i++) {part_cnts[i] = 0; part_marker[i] = -1;}
 	for(int i = 0; i < m; i++) part_cnts[partv[i] + 1]++;
	for(int i = 2; i < m_NoParts; i++) part_cnts[i] += part_cnts[i-1];

	//columns are nets... rows are partitioned. this information is primary
	//the column ordering will be based on this ordering
	//columns being in the cut and extra parameters provided.
	for(int i = 0; i < m; i++) {
		rowIPermutation[i] = part_cnts[partv[i]]++;
	}

	//if matrix is square (not necessarily symettric) and net placement is requested to be sym
	//then cols are distributed based on the row partitioning.
	//for unsymettric matrices - use MAX or END
	if((m == n) && (m_NetPlacement == "SYM")) { 
		for(int i = 0; i < n; i++) {
			colIPermutation[i] = rowIPermutation[i];
			cpartv[i] = partv[i];
		}
	} else {
		int conn = 0; int cut = 0;
		for(int i = 0; i <= m_NoParts; i++) {part_cnts[i] = 0;}
		for(int i = 0; i < n; i++) {
			int max_part = -1;
			for(int ptr = xpins[i]; ptr < xpins[i+1]; ptr++) {
				int part = partv[pins[ptr]];
				
				if(part_marker[part] != i) {
					//the first hit is given for every non-empty net
					conn++; //adds to connectivity
					part_cnts[part] = 1; //first time hitting the part - set the counts to 1. 
					part_marker[part] = i; //set the marker
				} else {
					//the second and later hits for this part only increases the counts
					part_cnts[part]++;
				}

				//if this is the first pin, or the current part has the most hits for this net
				if(max_part == -1 || part_cnts[part] > part_cnts[max_part]) {
					max_part = part;
				}
			}

			if(max_part == -1) {
				//empty net - goes to the end;
				cpartv[i] = 2 * m_NoParts;
			} else {
				conn--; //connectivity - 1's -1
				if(part_cnts[max_part] == xpins[i+1] - xpins[i]) {
					//internal net - goes inside the part (every hit is to the same part)
					cpartv[i] = 2 * max_part;
				} else {
					cut++; //cutnet - depending on m_NetPlacement it goes either in or to the end
					if(m_NetPlacement == "END") {
						cpartv[i] = 2 * m_NoParts;	
					} else {
						cpartv[i] = 2 * max_part + 1; //this is for making the cutnet goes to the end of the part
					}
				}
			}
		}
		printf("%s: ColNet Partitioning - CutNet: %d, Connectivity: %d\n", this->getMatrix().getName().c_str(), cut, conn); 
		

		//now columns also have parts - we can order them based on
		for(int i = 0; i <= 2 * m_NoParts + 1; i++) part_cnts[i] = 0;
		for(int i = 0; i < n; i++) part_cnts[cpartv[i] + 1]++;
		for(int i = 1; i <= 2 * m_NoParts + 1; i++) part_cnts[i] += part_cnts[i-1];		
		for(int i = 0; i < n; i++) {
			colIPermutation[i] = part_cnts[cpartv[i]]++;
			cpartv[i] /= 2;
		}
	}

	if(m_Inner != "NONE") {
		innerOrder(ptrs, js, m, n, partv, cpartv);
	}

	free(cpartv);
	free(part_cnts);
	free(part_marker);
	free(xpins);
	free(pins);
	free(nwghts);
	free(cwghts);
	free(degs);
}

void PatohOrdering::rowNetOrder(int* ptrs, int* js, int m, int n, int* partv) {
	int *xpins, *pins, *cwghts, *nwghts;
	int i, p;

	cwghts = (int *) malloc(sizeof(int) * n);
	memset(cwghts, 0, sizeof(int) * n);

	if(m_CellWght == "UNIT") {
		for(i = 0; i < n; i++) {
			cwghts[i] = 1;
		}
	} else {	
		for(i = 0; i < m; i++) {
			for(p = ptrs[i]; p < ptrs[i+1]; p++) {			
				cwghts[js[p]]++;
			}
		}
	}

	nwghts = (int *)malloc(sizeof(int) * m);	
	for(i = 0; i < m; i++) nwghts[i] = 1;

	xpins = (int *) malloc(sizeof(int) * (m+1));
	memcpy(xpins, ptrs, sizeof(int) * (m+1));

	pins = (int*) malloc(sizeof(int) * xpins[m]);	
	for(i = 0; i < m; i++) {
		memcpy(pins + xpins[i], js + ptrs[i], sizeof(int) * (ptrs[i+1] - ptrs[i]));
	}

	wrapPaToH(n, m, xpins, pins, cwghts, nwghts, partv, 1, m_NoParts);

    int* part_cnts = (int*) malloc(sizeof(int) * (m_NoParts + 2)); 
	int* part_marker = (int*) malloc(sizeof(int) * (m_NoParts + 2));    
	int* rpartv = (int*) malloc(sizeof(int) * m);

    for(int i = 0; i <= m_NoParts + 1; i++) {part_cnts[i] = 0; part_marker[i] = -1;}
 	for(int i = 0; i < n; i++) part_cnts[partv[i] + 1]++;
	for(int i = 2; i < m_NoParts; i++) part_cnts[i] += part_cnts[i-1];
	for(int i = 0; i < m; i++) {
		colIPermutation[i] = part_cnts[partv[i]]++;
	}

	if((m == n) && (m_NetPlacement == "SYM")) {
		for(int i = 0; i < n; i++) {
			rowIPermutation[i] = colIPermutation[i];
		}
	} else {
		int conn = 0; int cut = 0;
		for(int i = 0; i <= m_NoParts; i++) {part_cnts[i] = 0;}
		for(int i = 0; i < m; i++) {
			int max_part = -1;
			for(int ptr = ptrs[i]; ptr < ptrs[i+1]; ptr++) {
				int part = partv[js[ptr]];
				
				if(part_marker[part] != i) {
					conn++;
					part_cnts[part] = 1;
					part_marker[part] = i; 
				} else {
					part_cnts[part]++;
				}
				if(max_part == -1 || part_cnts[part] > part_cnts[max_part]) {
					max_part = part;
				}
			}
			if(max_part == -1) {
				//empty net - goes to the end;
				rpartv[i] = m_NoParts;
			} else {
				conn--; //connectivity - 1's -1
				if(part_cnts[max_part] == ptrs[i+1] - ptrs[i]) {
					//internal net - goes inside the part
					rpartv[i] = max_part;
				} else {
					cut++;
					//cutnet - depending on m_CPerm it goes either in or to the end
					if(m_NetPlacement == "END") {
						rpartv[i] = m_NoParts;	
					} else {
						rpartv[i] = max_part;
					}
				}
			}
		}
		printf("%s: RowNet Partitioning - CutNet: %d, Connectivity: %d\n", this->getMatrix().getName().c_str(), cut, conn); 

		for(int i = 0; i <= m_NoParts + 1; i++) part_cnts[i] = 0;
		for(int i = 0; i < m; i++) part_cnts[rpartv[i] + 1]++;
		for(int i = 2; i < m_NoParts + 1; i++) part_cnts[i] += part_cnts[i-1];		
		for(int i = 0; i < m; i++) {
			rowIPermutation[i] = part_cnts[rpartv[i]]++;
		}
	}

	if(m_Inner != "NONE") {
		innerOrder(ptrs, js, m, n, rpartv, partv);
	}

	free(rpartv);
	free(part_cnts);
	free(part_marker);
	free(xpins);
	free(pins);
	free(nwghts);
	free(cwghts);
}

void PatohOrdering::twoDimOrder(int* ptrs, int* I, int* J, int m, int n, int nz, int* partv) {
	int *rdegs, *cdegs, *xpins, *pins, *cwghts, *nwghts;
	int  i;
	
	nwghts = (int *) malloc(sizeof(int) * (m+n));	
	for(i = 0; i < (m+n); i++) nwghts[i] = 1;

	cwghts = (int *) malloc(sizeof(int) * nz);  
	for(i = 0; i < nz; i++) cwghts[i] = 1;

	cdegs = (int *)malloc(sizeof(int) * n);	
	memset(cdegs, 0, sizeof(int) * n);  
	rdegs = (int *)malloc(sizeof(int) * m);	
	memset(rdegs, 0, sizeof(int) * m);  
	for(i = 0; i < nz; i++) {
	  rdegs[I[i]]++;
	  cdegs[J[i]]++;
	}

	xpins = (int *) malloc(sizeof(int) * (m+n+1));
	xpins[0] = 0;
	for(i = 0; i < m; i++) {
		xpins[i+1] = xpins[i] + rdegs[i];
	}  
	for(i = m; i < m+n; i++) {
		xpins[i+1] = xpins[i] + cdegs[i-m];
	}

	for(i = 1; i < m-1; i++) rdegs[i] += rdegs[i-1];
	for(i = m-1; i > 0; i--) rdegs[i] = rdegs[i-1]; 
	rdegs[0] = 0;

	for(i = 1; i < n-1; i++) cdegs[i] += cdegs[i-1];
	for(i = n-1; i > 0; i--) cdegs[i] = nz + cdegs[i-1]; 
	cdegs[0] = nz;
	
	pins = (int*) malloc(sizeof(int) * xpins[m+n]);	
	for(i = 0; i < nz; i++) {
		pins[rdegs[I[i]]++] = i;    
		pins[cdegs[J[i]]++] = i;    
	}
 
	wrapPaToH(nz, n+m, xpins, pins, cwghts, nwghts, partv, 1, m_NoParts);

    int* part_cnts = (int*) malloc(sizeof(int) * (m_NoParts + 2)); 
	int* part_marker = (int*) malloc(sizeof(int) * (m_NoParts + 2));  
  
	int* rpartv = (int*) malloc(sizeof(int) * m);
	int* cpartv = (int*) malloc(sizeof(int) * n);

    for(int i = 0; i <= m_NoParts + 1; i++) {part_cnts[i] = 0; part_marker[i] = -1;}
	for(int i = 0; i < m; i++) {
		int max_part = -1;
		for(int ptr = xpins[i]; ptr < xpins[i+1]; ptr++) {
			int part = partv[pins[ptr]];
			
			if(part_marker[part] != i) {
				part_cnts[part] = 1;
				part_marker[part] = i; 
			} else {
				part_cnts[part]++;
			}
			if(max_part == -1 || part_cnts[part] > part_cnts[max_part]) {
				max_part = part;
			}
		}
		if(max_part == -1) {
			//empty net - goes to the end;
			rpartv[i] = m_NoParts;
		} else {
			if(part_cnts[max_part] == xpins[i+1] - xpins[i]) {
				//internal net - goes inside the part
				rpartv[i] = max_part;
			} else {
				//cutnet - depending on m_CPerm it goes either in or to the end
				if(m_NetPlacement == "END") {
					rpartv[i] = m_NoParts;	
				} else {
					rpartv[i] = max_part;
				}
			}
		}
	}
	for(int i = 0; i <= m_NoParts + 1; i++) part_cnts[i] = 0;
	for(int i = 0; i < m; i++) part_cnts[rpartv[i] + 1]++;
	for(int i = 2; i < m_NoParts + 1; i++) part_cnts[i] += part_cnts[i-1];		
	for(int i = 0; i < m; i++) {
		rowIPermutation[i] = part_cnts[rpartv[i]]++;
	}

	if((m == n) && (m_NetPlacement == "SYM")) {
		for(int i = 0; i < n; i++) {
			colIPermutation[i] = rowIPermutation[i];
		}
	} else {
		for(int i = 0; i <= m_NoParts + 1; i++) {part_cnts[i] = 0; part_marker[i] = -1;}
		for(int i = 0; i < n; i++) {
			int max_part = -1;
			for(int ptr = xpins[m+i]; ptr < xpins[m+i+1]; ptr++) {
				int part = partv[pins[ptr]];
				
				if(part_marker[part] != i) {
					part_cnts[part] = 1;
					part_marker[part] = i; 
				} else {
					part_cnts[part]++;
				}
				if(max_part == -1 || part_cnts[part] > part_cnts[max_part]) {
					max_part = part;
				}
			}
			if(max_part == -1) {
				//empty net - goes to the end;
				cpartv[i] = m_NoParts;
			} else {
				if(part_cnts[max_part] == xpins[m+i+1] - xpins[m+i]) {
					//internal net - goes inside the part
					cpartv[i] = max_part;
				} else {
					//cutnet - depending on m_CPerm it goes either in or to the end
					if(m_NetPlacement != "MAX") {
						cpartv[i] = m_NoParts;	
					} else {
						cpartv[i] = max_part;
					}
				}
			}
		}
		for(int i = 0; i <= m_NoParts + 1; i++) part_cnts[i] = 0;
		for(int i = 0; i < n; i++) part_cnts[cpartv[i] + 1]++;
		for(int i = 2; i < m_NoParts + 1; i++) part_cnts[i] += part_cnts[i-1];		
		for(int i = 0; i < n; i++) {
			colIPermutation[i] = part_cnts[cpartv[i]]++;
		}
	}

	if(m_Inner != "NONE") {
		innerOrder(ptrs, J, m, n, rpartv, cpartv);
	}

	free(cpartv);
	free(rpartv);
	free(xpins);
	free(pins);
	free(nwghts);
	free(cwghts);
	free(rdegs);
	free(cdegs);
}


void PatohOrdering::orderingFunction() 
{	
	int m = this->getMatrix().getRowCount();
    int n = this->getMatrix().getColCount(); 
    int nnz = this->getMatrix().getNNZCount();   
    int *js = (int*)(this->getMatrix().getInd());
    int *ptrs = (int*)(this->getMatrix().getPtr());

    rowIPermutation = new unsigned[m];
    colIPermutation = new unsigned[n];
    
	int _c = -1;
	if(m_Model == "CN") _c = m;
	else if(m_Model == "RN") _c = n;
	else if(m_Model == "2D") _c = nnz;
	else if(m_Model == "CB") _c = nnz;	
	
	int* partv = new int[_c]; 
		
    if(m_Model == "CN") colNetOrder(ptrs, js, m, n, partv);
    else if(m_Model == "RN") rowNetOrder(ptrs, js, m, n, partv);
    else if(m_Model == "2D") {
        int* I = new int[nnz];
        for(int i = 0; i < m; i++) {
            for(int j = ptrs[i]; j < ptrs[i+1]; j++) {
                I[j] = i;
            }
        }
       
		twoDimOrder(ptrs, I, js, m, n, nnz, partv);
		delete [] I;
	}
}