//
// Created by deniz on 3/30/24.
//

#include "sort.h"
#include "vector"


void countSort(std::vector< std::pair<unsigned, unsigned> >& inputArray, unsigned* permutation)
{
    unsigned N = inputArray.size();

    // Finding the maximum element of array inputArray[]
    unsigned M = 0;
    for (unsigned i = 0; i < N; i++) M = std::max(M, inputArray[i].second);

    // Prepare count array
    std::vector<unsigned> countArray(M + 1, 0);
    for (unsigned i = 0; i < N; i++) countArray[inputArray[i].second]++;
    for (unsigned i = 1; i <= M; i++) countArray[i] += countArray[i - 1];

    // Place the elements to the permutation
    std::vector<int> outputArray(N);
    for (unsigned i = N - 1; i < N; i--)
    {
        permutation[inputArray[i].first] = countArray[inputArray[i].second] - 1;
        countArray[inputArray[i].second]--;
    }
}


void sortNonzeros(vType order, vType* dims, eType nnz, vType* storage, valType* values)
{
    for (vType i = 0; i < order; ++i)
    {
        std::vector<std::vector<std::pair<std::vector<vType>, valType>>> buckets(dims[i]);
        for (eType j = 0; j < nnz; ++j)
        {
            std::vector<vType> coords(order);
            eType nnzStart = order * j;
            for (vType k = 0; k < order; ++k)
            {
                coords[k] = storage[nnzStart + k];
            }
            buckets[storage[nnzStart + i]].emplace_back(coords, values[j]);
        }

        eType idx = 0;
        for (vType j = 0; j < dims[i]; ++j)
        {
            for (auto& pair : buckets[j])
            {
                for (vType k = 0; k < order; ++k)
                {
                    storage[idx * order + k] = pair.first[k];
                }
                values[idx] = pair.second;
                ++idx;
            }
        }
    }
}


void Morton::sort(vType*& storage, valType*& vals, vType& order, eType& nnz, u_int8_t sbBits)
{
    vType left = 0;
    vType right = nnz - 1;
    switch (order)
    {
        case 3:
#pragma omp parallel num_threads(32)
        {
#pragma omp single nowait
            {
                quickMortonSort3D(storage, vals, left, right, sbBits);
            }
        }
            break;
        case 4:
#pragma omp parallel num_threads(32)
        {
#pragma omp single nowait
            {
                quickMortonSort4D(storage, vals, left, right, sbBits);
            }
        }
            break;
        default:
            break;
    }
}

vType Morton::quickSortPartition(vType *storage, valType *vals, vType left, vType right, Comparator comparator, Swapper swapper)
{
    vType middle = left + (right - left) / 2;
    vType* pivot = &storage[middle];

    // Pivot to the end
    swapper(&storage[middle], &vals[middle], &storage[right], &vals[right]);

    vType i = left - 1;

    for (vType j = left; j < right; ++j)
    {
        vType* current = &storage[j];
        if (comparator(current, pivot) < 0)
        {
            ++i;
            swapper(&storage[i], &vals[i], current, &vals[j]);
        }
    }

    swapper(&storage[i + 1], &vals[i + 1], &storage[right], &vals[right]);

    return (i + 1); // Position of pivot
}

void Morton::quickMortonSort3D(vType* storage, valType* vals, vType left, vType right, u_int8_t sbBits)
{
    if (left >= right)
    {
        return;
    }
    vType pivot = quickSortPartition(storage, vals, left, right, &quickMortonSort3DComparator, &swapNonzeros3D);
#pragma omp task firstprivate(left,pivot) shared(storage, vals)
    {
        quickMortonSort3D(storage, vals, left, pivot, sbBits);
    }
    quickMortonSort3D(storage, vals, pivot + 1, right, sbBits);
#pragma omp taskwait
}

void Morton::quickMortonSort4D(vType* storage, valType* vals, vType left, vType right, u_int8_t sbBits)
{
    if(left >= right)
    {
        return;
    }
    vType pivot = quickSortPartition(storage, vals, left, right, &quickMortonSort4DComparator, &swapNonzeros4D);
#pragma omp task firstprivate(left,pivot) shared(storage, vals)
    {
        quickMortonSort4D(storage, vals, left, pivot, sbBits);
    }
    quickMortonSort4D(storage, vals, pivot + 1, right, sbBits);
#pragma omp taskwait
}

int
Morton::quickMortonSort3DComparator(vType* coordinate1, vType* coordinate2)
{
    MortonIndex mKey1 = 0, mKey2 = 0;

    uint32_t x1 = coordinate1[0];
    uint32_t y1 = coordinate1[1];
    uint32_t z1 = coordinate1[2];
    uint32_t x2 = coordinate2[0];
    uint32_t y2 = coordinate2[1];
    uint32_t z2 = coordinate2[2];

    mKey1 =    morton256_z[(z1 >> 24) & 0xFF ] |
               morton256_y[(y1 >> 24) & 0xFF ] |
               morton256_x[(x1 >> 24) & 0xFF ];
    mKey1 =    mKey1 << 72 |
               morton256_z[(z1 >> 16) & 0xFF ] |
               morton256_y[(y1 >> 16) & 0xFF ] |
               morton256_x[(x1 >> 16) & 0xFF ];
    mKey1 =    mKey1 << 48 |
               morton256_z[(z1 >> 8) & 0xFF ] |
               morton256_y[(y1 >> 8) & 0xFF ] |
               morton256_x[(x1 >> 8) & 0xFF ];
    mKey1 =    mKey1 << 24 |
               morton256_z[(z1) & 0xFF ] |
               morton256_y[(y1) & 0xFF ] |
               morton256_x[(x1) & 0xFF ];

    mKey2 =    morton256_z[(z2 >> 24) & 0xFF ] |
               morton256_y[(y2 >> 24) & 0xFF ] |
               morton256_x[(x2 >> 24) & 0xFF ];
    mKey2 =    mKey2 << 72 |
               morton256_z[(z2 >> 16) & 0xFF ] |
               morton256_y[(y2 >> 16) & 0xFF ] |
               morton256_x[(x2 >> 16) & 0xFF ];
    mKey2 =    mKey2 << 48 |
               morton256_z[(z2 >> 8) & 0xFF ] |
               morton256_y[(y2 >> 8) & 0xFF ] |
               morton256_x[(x2 >> 8) & 0xFF ];
    mKey2 =    mKey2 << 24 |
               morton256_z[(z2) & 0xFF ] |
               morton256_y[(y2) & 0xFF ] |
               morton256_x[(x2) & 0xFF ];

    if(mKey1 < mKey2)
    {
        return -1;
    }
    else if(mKey1 > mKey2)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int
Morton::quickMortonSort4DComparator(vType* coordinate1, vType* coordinate2)
{
    MortonIndex mKey1, mKey2;

    uint32_t x1 = coordinate1[0];
    uint32_t y1 = coordinate1[1];
    uint32_t z1 = coordinate1[2];
    uint32_t w1 = coordinate1[3];
    uint32_t x2 = coordinate2[0];
    uint32_t y2 = coordinate2[1];
    uint32_t z2 = coordinate2[2];
    uint32_t w2 = coordinate2[3];

    static const uint64_t MASKS_64[]={0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, 0x0000FFFF0000FFFF};
    static const uint64_t SHIFTS_64[]= {1, 2, 4, 8, 16};
    static MortonIndex MASKS_128[] =
            {
                    (MortonIndex)0x5555555555555555 << 64 | 0x5555555555555555,
                    (MortonIndex)0x3333333333333333 << 64 | 0x3333333333333333,
                    (MortonIndex)0x0F0F0F0F0F0F0F0F << 64 | 0x0F0F0F0F0F0F0F0F,
                    (MortonIndex)0x00FF00FF00FF00FF << 64 | 0x00FF00FF00FF00FF,
                    (MortonIndex)0x0000FFFF0000FFFF << 64 | 0x0000FFFF0000FFFF,
                    (MortonIndex)0x00000000FFFFFFFF << 64 | 0x00000000FFFFFFFF
            };
    static const uint64_t SHIFTS_128[]= {1, 2, 4, 8, 16, 32};

    uint64_t tmp_64;
    MortonIndex x, y, z, w;

    tmp_64 = x1;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];

    x = tmp_64;
    x = (x | (x << SHIFTS_128[5])) & MASKS_128[5];
    x = (x | (x << SHIFTS_128[4])) & MASKS_128[4];
    x = (x | (x << SHIFTS_128[3])) & MASKS_128[3];
    x = (x | (x << SHIFTS_128[2])) & MASKS_128[2];
    x = (x | (x << SHIFTS_128[1])) & MASKS_128[1];
    x = (x | (x << SHIFTS_128[0])) & MASKS_128[0];

    tmp_64 = y1;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];

    y = tmp_64;
    y = (y | (y << SHIFTS_128[5])) & MASKS_128[5];
    y = (y | (y << SHIFTS_128[4])) & MASKS_128[4];
    y = (y | (y << SHIFTS_128[3])) & MASKS_128[3];
    y = (y | (y << SHIFTS_128[2])) & MASKS_128[2];
    y = (y | (y << SHIFTS_128[1])) & MASKS_128[1];
    y = (y | (y << SHIFTS_128[0])) & MASKS_128[0];

    tmp_64 = z1;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];

    z = tmp_64;
    z = (z | (z << SHIFTS_128[5])) & MASKS_128[5];
    z = (z | (z << SHIFTS_128[4])) & MASKS_128[4];
    z = (z | (z << SHIFTS_128[3])) & MASKS_128[3];
    z = (z | (z << SHIFTS_128[2])) & MASKS_128[2];
    z = (z | (z << SHIFTS_128[1])) & MASKS_128[1];
    z = (z | (z << SHIFTS_128[0])) & MASKS_128[0];

    tmp_64 = w1;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];

    w = tmp_64;
    w = (w | (w << SHIFTS_128[5])) & MASKS_128[5];
    w = (w | (w << SHIFTS_128[4])) & MASKS_128[4];
    w = (w | (w << SHIFTS_128[3])) & MASKS_128[3];
    w = (w | (w << SHIFTS_128[2])) & MASKS_128[2];
    w = (w | (w << SHIFTS_128[1])) & MASKS_128[1];
    w = (w | (w << SHIFTS_128[0])) & MASKS_128[0];

    mKey1 = w | (z << 1) | (y << 2) | (x << 3);

    tmp_64 = x2;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];

    x = tmp_64;
    x = (x | (x << SHIFTS_128[5])) & MASKS_128[5];
    x = (x | (x << SHIFTS_128[4])) & MASKS_128[4];
    x = (x | (x << SHIFTS_128[3])) & MASKS_128[3];
    x = (x | (x << SHIFTS_128[2])) & MASKS_128[2];
    x = (x | (x << SHIFTS_128[1])) & MASKS_128[1];
    x = (x | (x << SHIFTS_128[0])) & MASKS_128[0];

    tmp_64 = y2;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];

    y = tmp_64;
    y = (y | (y << SHIFTS_128[5])) & MASKS_128[5];
    y = (y | (y << SHIFTS_128[4])) & MASKS_128[4];
    y = (y | (y << SHIFTS_128[3])) & MASKS_128[3];
    y = (y | (y << SHIFTS_128[2])) & MASKS_128[2];
    y = (y | (y << SHIFTS_128[1])) & MASKS_128[1];
    y = (y | (y << SHIFTS_128[0])) & MASKS_128[0];

    tmp_64 = z2;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];

    z = tmp_64;
    z = (z | (z << SHIFTS_128[5])) & MASKS_128[5];
    z = (z | (z << SHIFTS_128[4])) & MASKS_128[4];
    z = (z | (z << SHIFTS_128[3])) & MASKS_128[3];
    z = (z | (z << SHIFTS_128[2])) & MASKS_128[2];
    z = (z | (z << SHIFTS_128[1])) & MASKS_128[1];
    z = (z | (z << SHIFTS_128[0])) & MASKS_128[0];

    tmp_64 = w2;
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[4])) & MASKS_64[4];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[3])) & MASKS_64[3];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[2])) & MASKS_64[2];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[1])) & MASKS_64[1];
    tmp_64 = (tmp_64 | (tmp_64 << SHIFTS_64[0])) & MASKS_64[0];

    w = tmp_64;
    w = (w | (w << SHIFTS_128[5])) & MASKS_128[5];
    w = (w | (w << SHIFTS_128[4])) & MASKS_128[4];
    w = (w | (w << SHIFTS_128[3])) & MASKS_128[3];
    w = (w | (w << SHIFTS_128[2])) & MASKS_128[2];
    w = (w | (w << SHIFTS_128[1])) & MASKS_128[1];
    w = (w | (w << SHIFTS_128[0])) & MASKS_128[0];

    mKey2 = w | (z << 1) | (y << 2) | (x << 3);

    if(mKey1 < mKey2)
    {
        return -1;
    }
    else if(mKey1 > mKey2)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void Morton::swapNonzeros3D(vType *coordinate1, valType *val1, vType *coordinate2, valType *val2)
{
    // Coordinate Change
    vType temp[3] = {coordinate2[0], coordinate2[1], coordinate2[2]};

    coordinate2[0] = coordinate1[0];
    coordinate2[1] = coordinate1[1];
    coordinate2[2] = coordinate1[2];

    coordinate1[0] = temp[0];
    coordinate1[1] = temp[1];
    coordinate1[2] = temp[2];

    // Value Change
    valType tempVal = *val2;

    *val2 = *val1;
    *val1 = tempVal;
}

void Morton::swapNonzeros4D(vType *coordinate1, valType *val1, vType *coordinate2, valType *val2)
{
    // Coordinate Change
    vType temp[4] = {coordinate2[0], coordinate2[1], coordinate2[2], coordinate2[3]};

    coordinate2[0] = coordinate1[0];
    coordinate2[1] = coordinate1[1];
    coordinate2[2] = coordinate1[2];
    coordinate2[3] = coordinate1[3];

    coordinate1[0] = temp[0];
    coordinate1[1] = temp[1];
    coordinate1[2] = temp[2];
    coordinate1[3] = temp[3];

    // Value Change
    valType tempVal = *val2;

    *val2 = *val1;
    *val1 = tempVal;
}
