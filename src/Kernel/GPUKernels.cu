#include "GPUKernels.cuh"


__global__ void cudaHelloWorldKernel(char *output)
{
    const char hw[] = "Hello World From Cuda!";
    for (int i = 0; hw[i] != '\0'; ++i)
    {
        output[i] = hw[i];
    }
}


__global__ void spmvRowBased(const vType* ptrs, const vType* inds, const valType* vals, const vType* m, const valType* input, valType* output)
{
    vType i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < *m)
    {
        valType value = 0;
        for (vType c = ptrs[i]; c != ptrs[i + 1]; ++c)
        {
            vType adj = inds[c];
            valType weight = vals[c];

            value += weight * input[adj];
        }
        output[i] = value;
    }
}

__global__ void cooTensorAoS(const vType* storage, eType nnzcount, int order, const float* input_vals, float* output) {
    int no_threads = blockDim.x * gridDim.x;
    eType tid = threadIdx.x + blockDim.x * blockIdx.x;
    float sum = 0;
    for(eType i = tid; i < nnzcount; i += no_threads) {
        float temp = 1;
        for(int m = 0; m < order; m++) {
            temp *= input_vals[storage[i * order + m]];
        }
        sum += temp;
    }
    atomicAdd(output, sum);
}

__global__ void cooTensorSoA(const vType* storage, eType nnzcount, int order, const float* input_vals, float* output) {
    int no_threads = blockDim.x * gridDim.x;
    eType tid = threadIdx.x + blockDim.x * blockIdx.x;
    float sum = 0;
    for(eType i = tid; i < nnzcount; i += no_threads) {
        float temp = 1;
        for(int m = 0; m < order; m++) {
            temp *= input_vals[storage[m * nnzcount + i]];
        }
        sum += temp;
    }
    atomicAdd(output, sum);
}