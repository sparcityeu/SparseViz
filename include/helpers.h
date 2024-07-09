#ifndef SPARSEVIZ_HELPERS_H
#define SPARSEVIZ_HELPERS_H

#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random> // for std::default_random_engine
#include <chrono> // for seeding the random number generator
#include <cassert>
#include <cctype>
#include "config.h"
#include <unordered_map>
#include <omp.h>


vType calculateBin(vType coordinate, vType dimSize, vType numBins);
std::string escapeSingleQuote(const std::string &input);
std::string calculateColorscale(bool is_exponential, bool is_red);
void trim(std::string &s);
std::vector<std::string> split(const std::string &s, char delimiter);
std::string merge(const std::vector<std::string>& splittedVersion);
std::string lowerString(const std::string& s);

struct entry {
	int row;
	int col;
	entry(int row, int col) : row(row), col(col) {}
};
typedef std::vector<entry> Entries;

typedef std::pair<vType, vType> vpair;
struct pair_hash
{
   // initialization
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return (pair.first >= pair.second) ? (pair.first * pair.first + pair.first + pair.second) : (pair.second * pair.second + pair.first);
    }
};


#endif //SPARSEVIZ_HELPERS_H
