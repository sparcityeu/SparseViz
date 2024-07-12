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


enum CPUBenchmarkSettings
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

std::vector<CPUBenchmarkSettings> getSettings();

class SparseVizPerformance
{
public:
    typedef std::unordered_map<std::string, std::pair<int, double>> OperationResults; // Key: BenchmarkSettings, Value: {fd, statistic}
    typedef std::unordered_map<pthread_t, OperationResults> Results;

public:
    SparseVizPerformance();
    SparseVizPerformance(const SparseVizPerformance& other) = delete;
    SparseVizPerformance& operator=(const SparseVizPerformance& other) = delete;
    SparseVizPerformance(SparseVizPerformance&& other) = delete;
    SparseVizPerformance& operator=(SparseVizPerformance&& other) = delete;
    ~SparseVizPerformance();

    void activatePerf(CPUBenchmarkSettings* settings, size_t size);
    void continuePerf();
    void pausePerf();
    OperationResults deactivatePerf();
    void calculateResults();

private:
    static long perfEventOpen(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);
    inline double calculateRatio(double& dividend, double& divider);

private:
    Results m_Results;
    pthread_mutex_t m_ResultsLock;
};


#endif //SPARSEVIZ_SPARSEVIZPERFORMANCE_H
