//
// Created by deniz on 3/18/24.
//

#ifndef SPARSEVIZ_SPARSEVIZPERFORMANCE_H
#define SPARSEVIZ_SPARSEVIZPERFORMANCE_H

#include <unordered_map>
#include <string>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <iostream>

// PERF RELATED
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <unistd.h>


enum BenchmarkSettings
{
    L1DataCacheLoad,
    L1DataCacheMiss,
    LLCacheLoad,
    LLCacheMiss,
//    L1InstructionCacheLoad,
//    L1InstructionCacheMiss,
    TLBDataCacheLoad,
    TLBDataCacheMiss,
    TLBInstructionCacheLoad,
    TLBInstructionCacheMiss,
    BranchInstructions,
    BranchMiss,
    Instructions,
    CPUClock,
    CPUChange,
    PageFault,
    ContextSwitch,
    END
};

std::vector<BenchmarkSettings> getSettings();

class SparseVizPerformance
{
public:
    typedef std::unordered_map<std::string, std::pair<int, double>> Results; // Key: BenchmarkSettings, Value: {fd, statistic}

public:
    SparseVizPerformance(BenchmarkSettings* settings, size_t size);
    SparseVizPerformance(const SparseVizPerformance& other) = delete;
    SparseVizPerformance& operator=(const SparseVizPerformance& other) = delete;
    SparseVizPerformance(SparseVizPerformance&& other) = delete;
    SparseVizPerformance& operator=(SparseVizPerformance&& other) = delete;
    ~SparseVizPerformance();

    void activatePerf();
    void deactivatePerf();
    void calculateResults();
    const Results& getResults() const;

private:
    static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);
    static double calculateRatio(double& dividend, double& divider);

private:
    Results m_Results; // Key: BenchmarkSettings, Value: {fd, statistic}
};


#endif //SPARSEVIZ_SPARSEVIZPERFORMANCE_H
