#include "LevelBasedSweep.h"
#include "OrderingHelpers.h"
#include <queue>

void LevelBasedSweep::orderingFunction()
{
    eType* row_ptr = this->getMatrix().getPtr();
    vType* col_idx = this->getMatrix().getInd();
    unsigned n = this->getMatrix().getRowCount();

	rowIPermutation = new unsigned[n];
	colIPermutation = new unsigned[n];	

	std::vector<int> depth(n, -1);
	std::vector<std::vector<int>> levels;

	auto comparatorAsc = [&](int v1, int v2)
	{
	  int v1deg = row_ptr[v1 + 1] - row_ptr[v1];
	  int v2deg = row_ptr[v2 + 1] - row_ptr[v2];
	  if (v1deg < v2deg)
	  {
		return true;
	  }
	  return false;
	};

	auto process_root = [&](int root) {
		std::queue<int> q;
		q.push(root);
		depth[root] = 0;
		std::vector<int> level;
		int current_level = 0;
		while (!q.empty()) {
			int u = q.front();
			q.pop();
			int d = depth[u];
			if (d > current_level)
			{
			  //std::sort(level.begin(), level.end(), comparatorAsc); // not sure if this is a requirement?
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
		  //std::sort(level.begin(), level.end(), comparatorAsc); // not sure again
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

	std::vector<int> flag(n, 0);
	std::vector<char> labelled(n, 0);
	int sweep = 0;
	int labelled_count = 0;

	while (labelled_count < n) {
		sweep++;
		for (int l = 0; l < levels.size(); ++l) {
			for (int idx = 0; idx < levels[l].size(); ++idx) {
				int u = levels[l][idx];
				if (labelled[u] || flag[u] == sweep) {
					continue;
				}
				rowIPermutation[u] = labelled_count++;
				labelled[u] = 1;
				for (int nnz = row_ptr[u]; nnz < row_ptr[u + 1]; ++nnz)
				{
				  int v = col_idx[nnz];
				  if (!labelled[v])
				  {
					flag[v] = sweep;
				  }
				}
			}
		}
	}

	for (int i = 0; i < n; ++i) {
        colIPermutation[i] = rowIPermutation[i];
	}
}
