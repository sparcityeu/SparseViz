//
// Created by grkm on 11/12/25.
//

#ifndef SPARSEVIZ_SLOANORDERING_H
#define SPARSEVIZ_SLOANORDERING_H

#include "MatrixOrdering.h"


class SloanOrdering  : public MatrixOrdering {
public:
    SloanOrdering(SparseMatrix& m,
                   std::string orderingName,
                   std::string /*unused*/)
        : MatrixOrdering(m,
                         std::move(orderingName),
                         /*rectangularSupport=*/false,
                         /*patternUnsymmetricSupport=*/false,
                         /*price=*/6.0) {}

protected:
    void orderingFunction() override;

    struct LevelStructure {
        std::vector<int> level;
        std::vector<std::vector<unsigned>> byLevel;
        unsigned depth = 0;
        unsigned width = 0;
    };

    // Node bookkeeping (everything per-node in one place)
    enum class Status : uint8_t {
        Inactive   = 0, // not adjacent to any active/postactive
        Preactive  = 1, // adjacent to an active node
        Active     = 2, // adjacent to a postactive node (but not postactive)
        Postactive = 3  // already labeled
    };

    struct NodeInfo {
        Status   status        = Status::Inactive;

        unsigned degree        = 0;   // m_i in the paper
        unsigned currentDegree = 0;   // n_i (computed from status + labeled neighborhood)
        int      distToEnd     = -1;  // d(e,i) from level structure rooted at e
        int      priority      = 0;   // P_i (integer scoring)

        bool inQueue = false;         // convenience flag for the unordered “priority queue”
    };

    // Parameters (should be user input later on)
    int W1 = 2;
    int W2 = 1;

    static void buildLevelStructure(
        unsigned root,
        const vType* ptr,
        const vType* ind,
        unsigned n,
        LevelStructure& out,
        unsigned widthMin = std::numeric_limits<unsigned>::max() // for “short-circuiting”
    );


    static void initializeNodeInfo(
        std::vector<NodeInfo>& info,
        const LevelStructure& Le,                 // already built from endNode (FULL, no abort)
        unsigned n,
        const std::vector<unsigned>& degree,
        const int &W1,
        const int &W2
    );

    static void Numbering(
        unsigned s,
        const vType* ptr,
        const vType* ind,
        unsigned n,
        int W1,
        std::vector<NodeInfo>& info,
        std::vector<unsigned>& eligible,
        std::vector<unsigned>& labels
    );
private:


    static void findPseudoPeripheralEndpoints(
        unsigned s,
        const vType* ptr,
        const vType* ind,
        unsigned n,
        const std::vector<unsigned>& degree,
        unsigned& sOut,
        unsigned& eOut
    );
/*
    static void recomputePriority(
    unsigned v,
    unsigned n,
    std::vector<NodeInfo>& info,
    int W1,
    int W2
);

    static void refreshNode(
        unsigned v,
        const vType* ptr,
        const vType* ind,
        unsigned n,
        std::vector<NodeInfo>& info,
        int W1,
        int W2
    );

    static void refreshClosedNeighborhood(
        unsigned center,
        const vType* ptr,
        const vType* ind,
        unsigned n,
        std::vector<NodeInfo>& info,
        int W1,
        int W2
    );

    static unsigned computeCurrentDegree(
    unsigned v,
    const vType* ptr,
    const vType* ind,
    const std::vector<NodeInfo>& info
);
*/
};


#endif //SPARSEVIZ_SLOANORDERING_H
