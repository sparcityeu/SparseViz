#include "MillerPritikin.h"
#include "OrderingHelpers.h"
#include <queue>

void MillerPritikin::orderingFunction()
{
    eType* row_ptr = this->getMatrix().getPtr();
    vType* col_idx = this->getMatrix().getInd();
    unsigned n = this->getMatrix().getRowCount();
    
	std::vector<int> depth(n, -1);
	std::vector<std::vector<int>> levels;

	auto process_root = [&](int root) {
		std::queue<int> q;
		q.push(root);
		depth[root] = 0;
		std::vector<int> level;
		unsigned current_level = 0;
		while (!q.empty()) {
			int u = q.front();
			q.pop();
			int d = depth[u];
			if (d > current_level)
			{
			  levels.emplace_back(level);
			  level.clear();
			  ++current_level;
			}
			level.emplace_back(u);
			for (int nnz = row_ptr[u]; nnz < row_ptr[u + 1]; ++nnz)
			{
			  int v = col_idx[nnz];
			  if (depth[v] == -1)
			  {
				depth[v] = d + 1;
				q.push(v);
			  }
			}
		}
		if (!level.empty())
		{
		  levels.emplace_back(level);
		}
	};

	int start = find_pseudo_peripheral_node(n, row_ptr, col_idx, 0);
	process_root(start);
	for (int i = 0; i < n; ++i) {
		if (depth[i] == -1) {
			process_root(i);
		}
	}

	std::vector<int> even_nodes;
	std::vector<int> odd_nodes;
	for (int lvl = 0; lvl < levels.size(); ++lvl) {
		if ((lvl % 2) == 0) {
			even_nodes.insert(even_nodes.end(), levels[lvl].begin(), levels[lvl].end());
		} else {
			odd_nodes.insert(odd_nodes.end(), levels[lvl].begin(), levels[lvl].end());
		}
	}
	if (even_nodes.size() < odd_nodes.size()) {
		std::swap(even_nodes, odd_nodes);
	}
	even_nodes.insert(even_nodes.end(), odd_nodes.begin(), odd_nodes.end());

    rowIPermutation = new unsigned[n];
    colIPermutation = new unsigned[n];

	for (int i = 0; i < n; ++i) {
        rowIPermutation[even_nodes[i]] = i;
        colIPermutation[even_nodes[i]] = i;
	}
}
