#include "OrderingHelpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int find_pseudo_peripheral_node(int n, eType* row_ptr, vType* col_idx, int start_node)
{
	int* visited = (int*)malloc(sizeof(int) * n); 
	memset(visited, 0, sizeof(int) * n);
	
	int* que = (int*)malloc(sizeof(int) * n); 

	int max_level = 0;
	int trial = 0; //number of BFSs
	int marker = 1; //not to reset visited
	while (trial < 10) { //final vertex is obtained
		marker++;
		
		que[0] = start_node;  
		visited[start_node] = marker;
		int qe = 1, qs = 0;
		int level_start = 0, level_end = qe, level_id = 1;
		
		while(qs < qe) {
			int v = que[qs++];
			for (int k = row_ptr[v]; k < row_ptr[v + 1]; ++k) {
				int w = col_idx[k];
				if (visited[w] != marker) {
					visited[w] = marker;
					que[qe++] = w;
				}
			}

			if(qs == level_end && qs < qe) {
			  level_start = qs;
			  level_end = qe;
			  level_id++;
			}
		}
		if(level_id == max_level) trial++; else trial = 0;
		max_level = level_id;
		int next_start_node = que[level_start + (rand() % (level_end - level_start))];

		start_node = next_start_node;
	}

	free(que);
	free(visited);
	return start_node;
}
