//                                                                                                                                                                                                          
// Created on 22 September 2023 (Modified for RCM++/BNF on 9 November 2025)                                                                                                                                                                
//                                                                                                                                                                                                          

#include "RCMPPOrdering.h"
#include <vector>
#include <limits> // Required for std::numeric_limits<double>::max()
#include "Parameters.h" // Assuming this is needed for m_RootSelection, etc.

using namespace std;

// Destructor:
RCMPPOrdering::~RCMPPOrdering() {
}




void RCMPPOrdering::sortNeighborsBuckets(unsigned* queue,unsigned   start_v,unsigned   qe,double*    sort_values)
{
    unsigned count = qe - start_v;

    // --- Fallback: insertion sort for tiny batches ---
    if (count <= SMALL_SORT_THRESHOLD) {
        for (unsigned i = start_v; i < qe; ++i) {
            unsigned j = i;
            while (j > start_v && sort_values[queue[j]] > sort_values[queue[j - 1]]) {
                unsigned tmp  = queue[j];
                queue[j] = queue[j - 1];
                queue[j - 1]  = tmp;
                --j;
            }
        }
        return;
    }

    // --- Find degree range in this batch ---
    double min_deg = sort_values[queue[start_v]];
    double max_deg = sort_values[queue[start_v]];
    for (unsigned i = start_v + 1; i < qe; ++i) {
        double d = sort_values[queue[i]];
        if (d < min_deg) min_deg = d;
        if (d > max_deg) max_deg = d;
    }

    // All neighbours have identical degree — nothing to sort.
    if (min_deg == max_deg) return;

    const double range = max_deg - min_deg;
    const double bucket_scale = static_cast<double>(NUM_SORT_BUCKETS - 1);

   // In social networks, the peripheral nodes have near-identical eccentricity. So using log-scale mapping would give more accurate results.
   const double log_range = std::log(range + 1.0);
   auto degToBucket = [&](double d) -> int {
        int b = static_cast<int>(std::log(max_deg - d + 1.0) / log_range * bucket_scale);
        if (b < 0) b = 0;                      
        if (b >= (int)NUM_SORT_BUCKETS) b = NUM_SORT_BUCKETS - 1;
        return b;
    };
    
    
    // --- Count pass: how many nodes land in each bucket? ---
    // Bucket 0 holds the HIGHEST degrees (descending output order).
    unsigned bucket_counts[NUM_SORT_BUCKETS] = {};
    for (unsigned i = start_v; i < qe; ++i) {
        ++bucket_counts[degToBucket(sort_values[queue[i]])];
    }
    
    // --- Prefix sums write-position per bucket (relative to temp[0]) ---
    unsigned write_pos[NUM_SORT_BUCKETS];
    write_pos[0] = 0;
    for (unsigned b = 1; b < NUM_SORT_BUCKETS; ++b)
        write_pos[b] = write_pos[b - 1] + bucket_counts[b - 1];

    // --- Scatter pass into temporary buffer ---
    vector<unsigned> temp(count);
    for (unsigned i = start_v; i < qe; ++i) {
        unsigned node = queue[i];
        temp[write_pos[degToBucket(sort_values[node])]++] = node;
    }

    // --- Copy back into queue ---
    for (unsigned i = 0; i < count; ++i)
        queue[start_v + i] = temp[i];
}



// --- Helper Functions for BNF/GL Logic ---

/**
 * @brief Performs Breadth-First Search (BFS) to construct the level structure.
 * @param start_node The vertex to start the BFS from.
 * @param queue Pointer to the array where the ordered nodes are stored.
 * @param level Pointer to the array storing the level (distance from start_node) of each node.
 * @param ccsize Output: The size of the connected component found.
 * @param final_level Output: The eccentricity (depth) of the start_node.
 */
void RCMPPOrdering::runBFS(unsigned start_node, unsigned* queue, int* level, unsigned& ccsize, int& final_level) {
  unsigned* ptrs = this->getMatrix().getPtr();
  unsigned* ids = this->getMatrix().getInd();
  
  queue[0] = start_node;
  level[start_node] = 0;
  unsigned qs = 0;
  unsigned qe = 1;

  int clevel = 0;
  while(qs < qe) {
    unsigned v = queue[qs++];
    clevel = level[v];
    for(unsigned ptr = ptrs[v]; ptr < ptrs[v+1]; ptr++) {
      unsigned nbr = ids[ptr];
      // Only visit unvisited nodes (level == -1)
      if(level[nbr] == -1) { 
        level[nbr] = clevel + 1;
        queue[qe++] = nbr;
      }
    }
  }
  ccsize = qe;
  final_level = (ccsize > 0) ? level[queue[qe-1]] : -1;
}


void RCMPPOrdering::resetLevels(unsigned* queue, int* level, unsigned ccsize) {
    for (unsigned i = 0; i < ccsize; i++)
        level[queue[i]] = -1;
}


/**
 * @brief Finds the node with the minimum degree on the deepest level L_epsilon(v) (George-Liu step).
 * @param queue The array containing nodes of the connected component ordered by BFS.
 * @param level The array of node levels.
 * @param ccsize The size of the connected component.
 * @param final_level The deepest level (eccentricity).
 * @return unsigned The node on the deepest level with the minimum degree.
 */
unsigned RCMPPOrdering::findMinDegreeNodeOnDeepestLevel(unsigned* queue, int* level, unsigned ccsize, int final_level) {
    unsigned* ptrs = this->getMatrix().getPtr();
    
    if (ccsize == 0 || final_level < 0) return 0;

    unsigned best_root = 0; 
    unsigned min_degree = (unsigned)-1; 

    // Iterate backwards through the queue to find all nodes at the deepest level (L_epsilon(v))
    for (unsigned i = ccsize - 1; i < ccsize; i--) { 
      unsigned v = queue[i];
      if (level[v] < final_level) {
          break; // We've moved past the deepest level
      }
      
      unsigned current_degree = ptrs[v+1] - ptrs[v];
      if(current_degree < min_degree) {
        min_degree = current_degree;
        best_root = v;
      }
      if (i == 0) break; 
    }   
    return best_root;
}

/**
 * @brief Calculates the width of the level structure (the size of the largest level).
 * @param queue The array containing nodes of the connected component ordered by BFS.
 * @param level The array of node levels.
 * @param ccsize The size of the connected component.
 * @param final_level The deepest level (eccentricity).
 * @return double The maximum size of any level in the structure.
 */
double RCMPPOrdering::calculateWidth(unsigned* queue, int* level, unsigned ccsize, int final_level) {
    if (ccsize <= 1) return 1.0;

    int num_levels = final_level + 1;
    std::vector<unsigned> level_sizes(num_levels, 0); 
    
    // Count nodes in each level
    for(unsigned i = 0; i < ccsize; i++) {
        unsigned node = queue[i];
        int l = level[node];
        if (l >= 0 && l < num_levels) {
            level_sizes[l]++;
        }
    }

    // Find the maximum size (the width)
    double max_size = 0.0;
    for(unsigned size : level_sizes) {
        if((double)size > max_size) {
            max_size = (double)size;
        }
    }
    return max_size;
}

// --- Original RCM function (Cuthill-McKee part) ---

void RCMPPOrdering::rcm (   unsigned& root, unsigned* queue,
                          double* sort_values,
                          bool* is_permuted) {
  unsigned* ptrs = this->getMatrix().getPtr();
  unsigned* ids = this->getMatrix().getInd();
  
  queue[0] = root;
  unsigned qs = 0;
  unsigned qe = 1;

  // Define the threshold for sorting
  const double SORT_THRESHOLD = 10.0;

  while(qs < qe) {
    unsigned v = queue[qs++];
    unsigned start_v = qe;
    unsigned high_deg_marker = qe;

    for(unsigned ptr = ptrs[v]; ptr < ptrs[v+1]; ptr++) {
      unsigned nbr = ids[ptr];

      if(!is_permuted[nbr]) {
        is_permuted[nbr] = true;

        if (sort_values[nbr] >= SORT_THRESHOLD) {
          // 1. Move a lower-degree placeholder to the end
          unsigned temp = queue[high_deg_marker];
          queue[qe++] = temp;
          // 2. Insert high-degree neighbor into the sortable segment
          queue[high_deg_marker++] = nbr;
        } else {
          // Insert low-degree neighbor at the end (will not be sorted)
          queue[qe++] = nbr;
        }
      } 
    }

    // Sort ONLY the neighbors that exceeded the degree threshold
    if(high_deg_marker > start_v) {
        sortNeighborsBuckets(queue, start_v, high_deg_marker, sort_values);
    }
  }
}

// --- Main Ordering Function (RCM++) ---
void RCMPPOrdering::orderingFunction() {
  unsigned n = this->getMatrix().getRowCount();
  unsigned* ptrs = this->getMatrix().getPtr();
  unsigned* ids = this->getMatrix().getInd();

  rowIPermutation = new unsigned[n];
  colIPermutation = new unsigned[n];

  unsigned* queue = new unsigned[n];
  int* level = new int[n];
  double* sort_values = new double[n];
  bool* is_permuted = new bool [n];
  
  // Initialize state
  for (unsigned i = 0; i < n; i++) {
    level[i] = -1;
    is_permuted[i] = false;
    // Default sorting criteria: Degree
    if(m_SortingCriteria == "DEG") {
      sort_values[i] = (double)(ptrs[i + 1] - ptrs[i]);
    } else {
      sort_values[i] = 0.0; 
    }
  }

  unsigned total_permuted = 0;

  for (unsigned i = 0; i < n; i++) { 
    if (!is_permuted[i]) { 
      unsigned current_root = i;
      int max_eccentricity = -1;
      unsigned ccsize = 0;
      unsigned* current_queue_start = queue + total_permuted;
      
      double recordWidth = std::numeric_limits<double>::max();
      unsigned recordedNode = i;

      // Handle isolated nodes or self-loops
      if(ptrs[i+1] == ptrs[i] || ((ptrs[i+1] == ptrs[i] + 1) && ids[ptrs[i]] == i)) {
        ccsize = 1;
        current_queue_start[0] = i;
        is_permuted[i] = true;
      } else {
        // --- BNF / George-Liu Root Selection ---
        unsigned next_root = current_root;
	int iterations = 0;
	const int MAX_BNF_ITERATIONS = 5;
        while(iterations < MAX_BNF_ITERATIONS) {
	  iterations++;
          current_root = next_root;
          int current_eccentricity;
          
          runBFS(current_root, current_queue_start, level, ccsize, current_eccentricity); 
          double current_width = calculateWidth(current_queue_start, level, ccsize, current_eccentricity);
          for(unsigned j = 0; j < ccsize; j++) {
            is_permuted[current_queue_start[j]] = true; 
          }
          
	  
	  if(current_eccentricity > max_eccentricity) {
            max_eccentricity = current_eccentricity;
            recordWidth = current_width;
            recordedNode = current_root;
          } else if (current_eccentricity == max_eccentricity) {
            if (current_width < recordWidth) {
              recordWidth = current_width;
              recordedNode = current_root;
            }
          }

          next_root = findMinDegreeNodeOnDeepestLevel(current_queue_start, level, ccsize, current_eccentricity);
          resetLevels(current_queue_start, level, ccsize);		 
          
          if (!m_RootSelection) { recordedNode = i; break; }
          if (next_root == current_root) break; 
          if (current_eccentricity < max_eccentricity) break; 
        
	  if (next_root == current_root) break; 
    	  if (current_eccentricity <= max_eccentricity) break;
	
	} 

        // --- Small World Optimization ---
        bool small_world = (ccsize > 1) && (max_eccentricity < static_cast<int>(std::log2(static_cast<double>(ccsize))));
        if (small_world && m_RootSelection) {
          int dummy_sw;
          runBFS(i, current_queue_start, level, ccsize, dummy_sw);
          resetLevels(current_queue_start, level, ccsize);
 
          unsigned min_deg_node = current_queue_start[0];
          unsigned min_deg = ptrs[min_deg_node + 1] - ptrs[min_deg_node];
          for (unsigned j = 1; j < ccsize; ++j) {
            unsigned v = current_queue_start[j];
            unsigned dv = ptrs[v + 1] - ptrs[v];
            if (dv < min_deg) { min_deg = dv; min_deg_node = v; }
          }
          recordedNode = min_deg_node;
        }
	
        // --- Final RCM Step for this Component ---
        // Clean the is_permuted flags for this specific component before RCM traversal
        for(unsigned j = 0; j < ccsize; j++) {
            is_permuted[current_queue_start[j]] = false;
        }

        current_root = recordedNode; 
        is_permuted[current_root] = true; // Mark root for RCM
        rcm(current_root, current_queue_start, sort_values, is_permuted); 

        // --- Reversal & Mapping (The Fix) ---
        // Reverse ONLY this component to minimize bandwidth
        for (unsigned j = 0; j < ccsize / 2; j++) {
            std::swap(current_queue_start[j], current_queue_start[ccsize - 1 - j]);
        }
      }
     
      // Assign final permutation indices for this component
      for (unsigned j = 0; j < ccsize; j++) {
          unsigned node = current_queue_start[j];
          rowIPermutation[node] = total_permuted + j;
          colIPermutation[node] = total_permuted + j;
      }

      total_permuted += ccsize;    
    }
  }

  delete [] queue;
  delete [] level;
  delete [] sort_values;
  delete [] is_permuted;
}
