//
// Created on 1/18/24.
//

#include "MatrixKernelFunction.h"
#include "omp.h"


void MatrixKernelFunction::postprocess(const SparseMatrix& A, int iter)
{
    logger->makeSilentLog("Iteration " + std::to_string(iter) + " is completed on " + A.getName());
}

KernelResult MatrixKernelFunction::operator()(const SparseMatrix& A)
{
    if (KERNEL_PERFORMANCE_LOG)
    {
        sparseVizPerformance->pausePerf();
    }

    this->determineOMPSchedule();

    std::vector<double> durations;
    if (this->init(A))
    {
        for (int i = 0; i != threadCounts.size(); ++i)
        {
            double total = 0;
            omp_set_num_threads(threadCounts[i]);
            for (int r = 0; r != nRun; ++r)
            {
                if (KERNEL_PERFORMANCE_LOG && r == nIgnore)
                {
                    sparseVizPerformance->continuePerf();
                }

                this->preprocess(A);
                double start = omp_get_wtime();
                this->functionBody(A, r);
                double end = omp_get_wtime();

                if(r >= nIgnore)
                {
                    total += end - start;
                }
#ifdef LOG_KERNEL_DETAIL
                this->postprocess(A, r);
#endif
            }

            double duration = total / (nRun - nIgnore);
            durations.push_back(duration);
        }
    }
    omp_set_num_threads(omp_get_max_threads());
    return {kernelName, threadCounts, schedulingPolicy, chunkSize, durations};
}

void MatrixKernelFunction::determineOMPSchedule()
{
    if (schedulingPolicy == "static")
    {
        omp_set_schedule(omp_sched_static, chunkSize);
    }
    else if (schedulingPolicy == "auto")
    {
        omp_set_schedule(omp_sched_auto, chunkSize);
    }
    else if (schedulingPolicy == "dynamic")
    {
        omp_set_schedule(omp_sched_static, chunkSize);
    }
    else if (schedulingPolicy == "guided")
    {
        omp_set_schedule(omp_sched_static, chunkSize);
    }
}
