#ifndef SPARSEVIZ_KERNELRESULT_H
#define SPARSEVIZ_KERNELRESULT_H

#include <string>
#include <vector>


struct KernelResult
{
    KernelResult(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, const std::vector<double>& durations)
    :   kernelName(kernelName), threadCounts(threadCounts), schedulingPolicy(schedulingPolicy), chunkSize(chunkSize), durations(durations) {}

    std::string kernelName;
    std::vector<int> threadCounts;
    std::string schedulingPolicy;
    int chunkSize;
    std::vector<double> durations;
};


#endif //SPARSEVIZ_KERNELRESULT_H
