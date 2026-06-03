//
// Created by orderinggroup on 12/23/25.
//

#ifndef SPARSEVIZ_SLOAN_MGPSORDERING_H
#define SPARSEVIZ_SLOAN_MGPSORDERING_H


#include "SloanOrdering.h"

class MGPSSloanOrdering final : public SloanOrdering {
public:
    MGPSSloanOrdering(SparseMatrix& m,
                      std::string orderingName,
                      std::string /*unused*/)
        : SloanOrdering(m, std::move(orderingName), "") {}

protected:
    void orderingFunction() override;

private:
    static void findPseudoPeripheralEndpointsMGPS(
        unsigned s,
        const vType* ptr,
        const vType* ind,
        unsigned n,
        const std::vector<unsigned>& degree,
        unsigned& sOut,
        unsigned& eOut
    );

    static void buildLevelStructure(
    unsigned root,
    const vType* ptr,
    const vType* ind,
    unsigned  n,
    LevelStructure& out,
    unsigned widthMin
    );

    static bool isNeighborOfAnyPicked(
    unsigned v,
    const vType* ptr,
    const vType* ind,
    const std::vector<unsigned>& picked
    );
};





#endif //SPARSEVIZ_SLOAN_MGPSORDERING_H