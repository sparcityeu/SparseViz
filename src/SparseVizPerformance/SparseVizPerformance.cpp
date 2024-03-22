//
// Created by deniz on 3/18/24.
//

#include "SparseVizPerformance.h"


std::vector<BenchmarkSettings> getSettings()
{
    std::vector<BenchmarkSettings> settings;

    for (size_t i = 0; i < static_cast<int>(BenchmarkSettings::END); ++i)
    {
        settings.push_back(static_cast<BenchmarkSettings>(i));
    }

    return settings;
}

SparseVizPerformance::SparseVizPerformance(BenchmarkSettings* settings, size_t size)
{
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

        int fd = perf_event_open(&pe, 0, -1, -1, 0);
        if (fd == -1)
        {
            throw std::runtime_error("Failed to open perf event: " + config);
        }
        m_Results[config] = {fd, 0};
    }
}

SparseVizPerformance::~SparseVizPerformance()
{
    for (auto& result: m_Results)
    {
        close(result.second.first);
    }
}

void SparseVizPerformance::activatePerf()
{
    for (auto& result: m_Results)
    {
        ioctl(result.second.first, PERF_EVENT_IOC_RESET, 0);
        ioctl(result.second.first, PERF_EVENT_IOC_ENABLE, 0);
    }
}

void SparseVizPerformance::deactivatePerf()
{
    for (auto& result: m_Results)
    {
        ioctl(result.second.first, PERF_EVENT_IOC_DISABLE, 0);
    }
}

void SparseVizPerformance::calculateResults()
{
    for (auto& result: m_Results)
    {
        unsigned long long count;
        read(result.second.first, &count, sizeof(unsigned long long));
        result.second.second = count;
    }


    unsigned long long CPUSeconds = m_Results["CPU Clock"].second;
    CPUSeconds /= 1e9;
    m_Results["CPU Time Passed"] = {-1, CPUSeconds};
//    m_Results.erase("CPU Clock");


    // Calculating Ratios
    double l1DataMissRatio = calculateRatio(m_Results["L1 Data Cache Misses"].second, m_Results["L1 Data Cache Loads"].second);
    m_Results["L1 Data Cache Miss Ratio"] = {-1, l1DataMissRatio};
    m_Results.erase("L1 Data Cache Misses");
    m_Results.erase("L1 Data Cache Loads");

//    double l1InstructionMissRatio = calculateRatio(m_Results["L1 Instruction Cache Misses"].second, m_Results["L1 Instruction Cache Loads"].second);
//    m_Results["L1 Instruction Cache Miss Ratio"] = {-1, l1InstructionMissRatio};
//    m_Results.erase("L1 Instruction Cache Misses");
//    m_Results.erase("L1 Instruction Cache Loads");

    double llCacheMissRatio = calculateRatio(m_Results["LL Cache Misses"].second, m_Results["LL Cache Loads"].second);
    m_Results["LL Cache Miss Ratio"] = {-1, llCacheMissRatio};
    m_Results.erase("LL Cache Misses");
    m_Results.erase("LL Cache Loads");

    double tlbDataMissRatio = calculateRatio(m_Results["TLB Data Misses"].second, m_Results["TLB Data Loads"].second);
    m_Results["TLB Data Miss Ratio"] = {-1, tlbDataMissRatio};
    m_Results.erase("TLB Data Misses");
    m_Results.erase("TLB Data Loads");

    double tlbInstructionMissRatio = calculateRatio(m_Results["TLB Instruction Misses"].second, m_Results["TLB Instruction Loads"].second);
    m_Results["TLB Instruction Miss Ratio"] = {-1, tlbInstructionMissRatio};
    m_Results.erase("TLB Instruction Misses");
    m_Results.erase("TLB Instruction Loads");

    double branchMissRatio = calculateRatio(m_Results["Branch Misses"].second, m_Results["Branch Instructions"].second);
    m_Results["Branch Miss Ratio"] = {-1, branchMissRatio};
    m_Results.erase("Branch Misses");
    m_Results.erase("Branch Instructions");

}

double SparseVizPerformance::calculateRatio(double& dividend, double& divider)
{
    if (divider == 0)
    {
        return 0;
    }

    return ((dividend / divider) * 100);
}

const SparseVizPerformance::Results& SparseVizPerformance::getResults() const
{
    return m_Results;
}

long SparseVizPerformance::perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}
