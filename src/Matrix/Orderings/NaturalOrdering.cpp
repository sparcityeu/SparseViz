//
// Created on 25.12.2023.
//

#include "NaturalOrdering.h"


void NaturalOrdering::orderingFunction()
{
    if (rowIPermutation || colIPermutation)
    {
        return;
    }

    rowIPermutation = new vType[this->getMatrix().getRowCount()];
    for (int i = 0; i != this->getMatrix().getRowCount(); ++i)
    {
        rowIPermutation[i] = i;
    }

    colIPermutation = new vType[this->getMatrix().getColCount()];
    for (int i = 0; i != this->getMatrix().getColCount(); ++i)
    {
        colIPermutation[i] = i;
    }
}
