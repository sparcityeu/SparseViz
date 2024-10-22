#ifndef SPARSEVIZ_GPUKERNELRESULT_H
#define SPARSEVIZ_GPUKERNELRESULT_H

#include <string>
#include <vector>


struct GPUKernelResult
{
    GPUKernelResult(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, const std::vector<int>& sharedMemorySizes, std::vector<double>& durations)
    :   kernelName(kernelName), gridSizes(gridSizes), blockSizes(blockSizes), sharedMemorySizes(sharedMemorySizes), durations(durations) {}

    std::string kernelName;
    std::vector<int> gridSizes;
    std::vector<int> blockSizes;
    std::vector<int> sharedMemorySizes;
    std::vector<double> durations;
};


#endif //SPARSEVIZ_GPUKERNELRESULT_H
