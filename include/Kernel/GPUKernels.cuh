#include "cuda_runtime.h"
#include "config.h"


__global__ void cudaHelloWorldKernel(char* output);
__global__ void spmvRowBased(const vType* ptrs, const vType* inds, const valType* vals, const vType* m, const valType* input, valType* output);
__global__ void cooTensorAoS(const vType* storage, eType nnzcount, int order, const float* input_vals, float* output);
__global__ void cooTensorSoA(const vType* storage, eType nnzcount, int order, const float* input_vals, float* output);


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
