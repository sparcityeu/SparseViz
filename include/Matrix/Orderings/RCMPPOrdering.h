//
// Created by Kamer Kaya on 22.12.2023. (Modified by Mehmet Berkay Catak on 09.11.2025)
//

#ifndef SPARSEVIZ_RCMPPORDERING_H
#define SPARSEVIZ_RCMPPORDERING_H

#include "MatrixOrdering.h"
#include "SparseMatrix.h"
#include <vector>
#include <algorithm>
#include "config.h"
#include "helpers.h"

class RCMPPOrdering: public MatrixOrdering
{
public:
    RCMPPOrdering(SparseMatrix& matrix, std::string orderingName, std::string orderingParameters) :
    MatrixOrdering(matrix, orderingName, false, false), m_SortingCriteria("DEG"), m_RootSelection(true) {

        if(orderingParameters != "") {
            std::vector<std::string> parameters = split(orderingParameters, '/');

            if (parameters.size() > 0) {
                m_SortingCriteria = parameters[0];
            } 
            if (parameters.size() > 1) {
                if(parameters[1] == "FALSE") {
                    m_RootSelection = false;
                }
            }
        }
    }

	virtual ~RCMPPOrdering();
private:
    std::string m_SortingCriteria; 
    bool m_RootSelection; 

    static constexpr unsigned NUM_SORT_BUCKETS     = 16;
    static constexpr unsigned SMALL_SORT_THRESHOLD = 6;

    virtual void orderingFunction() override;
    
    // RCM++ (BNF) Helper Functions
    void runBFS (    unsigned start_node, unsigned* queue,
                       int* level, unsigned& ccsize, int& final_level);  
    unsigned findMinDegreeNodeOnDeepestLevel (    unsigned* queue, int* level, unsigned ccsize, int final_level); 
    double calculateWidth (    unsigned* queue, int* level, unsigned ccsize, int final_level); 
    void resetLevels(unsigned int* queue, int* level, unsigned int ccsize);

    void sortNeighborsBuckets(unsigned* queue, unsigned start_v, unsigned qe, double* sort_values);
    
    // Original RCM function
    void rcm (  unsigned& root, unsigned* queue,    
                double* sort_values,
                bool* is_permuted);
    
};

#endif //SPARSEVIZ_RCMPPORDERING_H
