//
// Created by deniz on 3/18/24.
//

#include "SparseVizPerformance.h"


std::vector<CPUBenchmarkSettings> getSettings()
{
    std::vector<CPUBenchmarkSettings> settings;

    for (size_t i = 0; i < static_cast<int>(CPUBenchmarkSettings::END); ++i)
    {
        settings.push_back(static_cast<CPUBenchmarkSettings>(i));
    }

    return settings;
}

SparseVizPerformance::SparseVizPerformance()
{
    pthread_mutex_init(&m_ResultsLock, NULL);
}

SparseVizPerformance::~SparseVizPerformance()
{
    pthread_mutex_destroy(&m_ResultsLock);
}

void SparseVizPerformance::activatePerf(CPUBenchmarkSettings* settings, size_t size)
{
    pthread_t myID = pthread_self();
    OperationResults operationResults;

    for (size_t i = 0; i < size; ++i)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.size = sizeof(struct perf_event_attr);
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        std::string config;

        switch (settings[i])
        {
            case Instructions:
                pe.type = PERF_TYPE_HARDWARE;
                config = "Instructions";
                pe.config = PERF_COUNT_HW_INSTRUCTIONS;
                break;
            case BranchInstructions:
                pe.type = PERF_TYPE_HARDWARE;
                config = "Branch Instructions";
                pe.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
                break;
            case BranchMiss:
                pe.type = PERF_TYPE_HARDWARE;
                config = "Branch Misses";
                pe.config = PERF_COUNT_HW_BRANCH_MISSES;
                break;
            case CPUClock:
                pe.type = PERF_TYPE_SOFTWARE;
                config = "CPU Clock";
                pe.config = PERF_COUNT_SW_CPU_CLOCK; // TASK_CLOCK would be better - will be tested later
                break;
            case CPUChange:
                pe.type = PERF_TYPE_SOFTWARE;
                config = "CPU Changes";
                pe.config = PERF_COUNT_SW_CPU_MIGRATIONS;
                break;
            case PageFault:
                pe.type = PERF_TYPE_SOFTWARE;
                config = "Page Faults";
                pe.config = PERF_COUNT_SW_PAGE_FAULTS_MAJ;
                break;
            case ContextSwitch:
                pe.type = PERF_TYPE_SOFTWARE;
                config = "Context Switches";
                pe.config = PERF_COUNT_SW_CONTEXT_SWITCHES;
                break;
            case L1DataCacheLoad:
                pe.type = PERF_TYPE_HW_CACHE;
                config = "L1 Data Cache Loads";
                pe.config = (PERF_COUNT_HW_CACHE_L1D << 0| PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                break;
            case L1DataCacheMiss:
                pe.type = PERF_TYPE_HW_CACHE;
                config = "L1 Data Cache Misses";
                pe.config = (PERF_COUNT_HW_CACHE_L1D << 0| PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                break;
//            case L1InstructionCacheLoad:
//                pe.type = PERF_TYPE_HW_CACHE;
//                config = "L1 Instruction Cache Loads";
//                pe.config = (PERF_COUNT_HW_CACHE_L1I << 0 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
//                break;
//            case L1InstructionCacheMiss:
//                pe.type = PERF_TYPE_HW_CACHE;
//                config = "L1 Instruction Cache Misses";
//                pe.config = (PERF_COUNT_HW_CACHE_L1I << 0 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
//                break;
            case LLCacheLoad:
                pe.type = PERF_TYPE_HW_CACHE;
                config = "LL Cache Loads";
                pe.config = (PERF_COUNT_HW_CACHE_LL << 0 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                break;
            case LLCacheMiss:
                pe.type = PERF_TYPE_HW_CACHE;
                config = "LL Cache Misses";
                pe.config = (PERF_COUNT_HW_CACHE_LL << 0 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                break;
            case TLBDataCacheLoad:
                pe.type = PERF_TYPE_HW_CACHE;
                config = "TLB Data Loads";
                pe.config = (PERF_COUNT_HW_CACHE_DTLB << 0 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                break;
            case TLBDataCacheMiss:
                pe.type = PERF_TYPE_HW_CACHE;
                config = "TLB Data Misses";
                pe.config = (PERF_COUNT_HW_CACHE_DTLB << 0 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                break;
            case TLBInstructionCacheLoad:
                pe.type = PERF_TYPE_HW_CACHE;
                config = "TLB Instruction Loads";
                pe.config = (PERF_COUNT_HW_CACHE_ITLB << 0 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                break;
            case TLBInstructionCacheMiss:
                pe.type = PERF_TYPE_HW_CACHE;
                config = "TLB Instruction Misses";
                pe.config = (PERF_COUNT_HW_CACHE_ITLB << 0 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                break;
            default:
                continue;
        }

        int fd = SparseVizPerformance::perfEventOpen(&pe, 0, -1, -1, 0);
        if (fd == -1)
        {
            throw std::runtime_error("Failed to open perf event: " + config);
        }
        operationResults[config] = {fd, 0};
    }

    for (auto& result: operationResults)
    {
        ioctl(result.second.first, PERF_EVENT_IOC_RESET, 0);
        ioctl(result.second.first, PERF_EVENT_IOC_ENABLE, 0);
    }

    pthread_mutex_lock(&m_ResultsLock);
    m_Results[myID] = operationResults;
    pthread_mutex_unlock(&m_ResultsLock);
}

void SparseVizPerformance::continuePerf()
{
    pthread_t myID = pthread_self();
    auto& operationData = m_Results[myID];

    for (auto& result: operationData)
    {
        ioctl(result.second.first, PERF_EVENT_IOC_ENABLE, 0);
    }
}

void SparseVizPerformance::pausePerf()
{
    pthread_t myID = pthread_self();
    auto& operationData = m_Results[myID];

    for (auto& result: operationData)
    {
        ioctl(result.second.first, PERF_EVENT_IOC_DISABLE, 0);
    }
}

SparseVizPerformance::OperationResults SparseVizPerformance::deactivatePerf()
{
    pthread_t myID = pthread_self();

    for (auto& result: m_Results[myID])
    {
        ioctl(result.second.first, PERF_EVENT_IOC_DISABLE, 0);
    }

    this->calculateResults();
    OperationResults ret = m_Results[myID];

    for (auto& result: m_Results[myID])
    {
        close(result.second.first);
    }

    pthread_mutex_lock(&m_ResultsLock);
    m_Results.erase(myID);
    pthread_mutex_unlock(&m_ResultsLock);

    return ret;
}

void SparseVizPerformance::calculateResults()
{
    pthread_t myID = pthread_self();

    auto& operationResults = m_Results[myID];

    for (auto& result: operationResults)
    {
        unsigned long long count;
        read(result.second.first, &count, sizeof(unsigned long long));
        result.second.second = count;
    }

    unsigned long long CPUSeconds = operationResults["CPU Clock"].second;
    CPUSeconds /= 1e9;
    operationResults["CPU Time Passed"] = {-1, CPUSeconds};
//    m_Results.erase("CPU Clock");

    // Calculating Ratios
    double l1DataMissRatio = this->calculateRatio(operationResults["L1 Data Cache Misses"].second, operationResults["L1 Data Cache Loads"].second);
    operationResults["L1 Data Cache Miss Ratio"] = {-1, l1DataMissRatio};
    operationResults.erase("L1 Data Cache Misses");
    operationResults.erase("L1 Data Cache Loads");

//    double l1InstructionMissRatio = this->calculateRatio(m_Results["L1 Instruction Cache Misses"].second, m_Results["L1 Instruction Cache Loads"].second);
//    m_Results["L1 Instruction Cache Miss Ratio"] = {-1, l1InstructionMissRatio};
//    m_Results.erase("L1 Instruction Cache Misses");
//    m_Results.erase("L1 Instruction Cache Loads");

    double llCacheMissRatio = this->calculateRatio(operationResults["LL Cache Misses"].second, operationResults["LL Cache Loads"].second);
    operationResults["LL Cache Miss Ratio"] = {-1, llCacheMissRatio};
    operationResults.erase("LL Cache Misses");
    operationResults.erase("LL Cache Loads");

    double tlbDataMissRatio = this->calculateRatio(operationResults["TLB Data Misses"].second, operationResults["TLB Data Loads"].second);
    operationResults["TLB Data Miss Ratio"] = {-1, tlbDataMissRatio};
    operationResults.erase("TLB Data Misses");
    operationResults.erase("TLB Data Loads");

    double tlbInstructionMissRatio = this->calculateRatio(operationResults["TLB Instruction Misses"].second, operationResults["TLB Instruction Loads"].second);
    operationResults["TLB Instruction Miss Ratio"] = {-1, tlbInstructionMissRatio};
    operationResults.erase("TLB Instruction Misses");
    operationResults.erase("TLB Instruction Loads");

    double branchMissRatio = this->calculateRatio(operationResults["Branch Misses"].second, operationResults["Branch Instructions"].second);
    operationResults["Branch Miss Ratio"] = {-1, branchMissRatio};
    operationResults.erase("Branch Misses");
    operationResults.erase("Branch Instructions");
}

inline double SparseVizPerformance::calculateRatio(double& dividend, double& divider)
{
    if (divider == 0)
    {
        return 0;
    }

    return ((dividend / divider) * 100);
}

long SparseVizPerformance::perfEventOpen(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}
