#ifndef SPARSEVIZ_NONZERO_H
#define SPARSEVIZ_NONZERO_H

#include "config.h"


struct Nonzero
{
    Nonzero() {}
    Nonzero(vType r, vType c, valType v): row(r), column(c), value(v) {}
    vType row;
    vType column;
    valType value;
};


#endif //SPARSEVIZ_NONZERO_H
