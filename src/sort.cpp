//
// Created by deniz on 3/30/24.
//

#include "sort.h"
#include "vector"


void countSort(std::vector< std::pair<vType, vType> >& inputArray, vType* permutation)
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
    typedef std::pair<std::vector<vType>, valType> Element;
    typedef std::vector<Element> Bucket;
    typedef std::vector<Bucket> Buckets;

    for (vType i = order; i > 0; --i)
    {
        Buckets buckets(dims[i - 1]);
        for (eType j = 0; j < nnz; ++j)
        {
            std::vector<vType> coords(order);
            eType nnzStart = order * j;
            for (vType k = 0; k < order; ++k)
            {
                coords[k] = storage[nnzStart + k];
            }
            buckets[coords[i - 1]].emplace_back(coords, values[j]);
        }

        eType idx = 0;
        for (vType j = 0; j < dims[i - 1]; ++j)
        {
            for (auto& el : buckets[j])
            {
                for (vType k = 0; k < order; ++k)
                {
                    storage[idx * order + k] = el.first[k];
                }
                values[idx] = el.second;
                ++idx;
            }
        }
    }
}
