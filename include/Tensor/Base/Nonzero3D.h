//
// Created on 1/14/24.
//

#ifndef SPARSEVIZ_NONZERO3D_H
#define SPARSEVIZ_NONZERO3D_H

#include "config.h"

struct Nonzero3D
{
    Nonzero3D() {}
    Nonzero3D(vType x, vType y, vType z, valType value): x(x), y(y), z(z), value(value) {}
    bool operator==(const Nonzero3D& other) const { //checks coord equality
        if (x == other.x && y == other.y && z == other.z) {
            return true;
        }
        return false;
    }
    vType x;
    vType y;
    vType z;
    valType value;
};


#endif //SPARSEVIZ_NONZERO3D_H
