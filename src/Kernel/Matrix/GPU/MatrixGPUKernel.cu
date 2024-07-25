#include "MatrixGPUKernel.h"
#include "omp.h"


void MatrixGPUKernel::postprocess(const SparseMatrix& A, int iter)
{
    logger->makeSilentLog("Iteration " + std::to_string(iter) + " is completed on " + A.getName());
}

GPUKernelResult MatrixGPUKernel::operator()(const SparseMatrix& A)
{
    std::vector<double> durations;
    if (this->init(A))
    {
        for (int i = 0; i < gridSizes.size(); ++i)
        {
            double total = 0;
            for (int r = 0; r < nRun; ++r)
            {
                this->preprocess(A);
                double start = omp_get_wtime();
                this->hostFunction(A, r, gridSizes[i], blockSizes[i], sharedMemorySizes[i]);
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
    return {kernelName, gridSizes, blockSizes, sharedMemorySizes, durations};
}
