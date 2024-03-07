# Tutorial 4.2: How to Add GPU Matrix Kernel

In this tutorial, we will have a look at how GPU matrix kernels are
added to the SparseViz library. All user-defined GPU matrix kernel
classes should derive from the abstract MatrixGPUKernel class that
facilitates the integration of kernels into SparseViz. As indicated
previously, GPU matrix kernel parallelization is made possible with Cuda
in the SparseViz library. That's why, along the way, we are going to
need .cu and .cuh files in which to implement our parallel device
kernels that are going to be executed on GPU. First stop, we are going
to create our class declaration under the directory
SparseViz/include/Kernel/Matrix/GPU.

*AddingGPUMatrixKernel.h*

```cpp
#include "MatrixGPUKernel.h"

class AddingGPUMatrixKernel: public MatrixGPUKernel

{

public:

    AddingGPUMatrixKernel(const std::string& kernelName, const std::vector<int>& gridSizes, const std::vector<int>& blockSizes, int nRun, int nIgnore)
    :   MatrixGPUKernel(kernelName, gridSizes, blockSizes, nRun, nIgnore) {}
    
    virtual ~AddingGPUMatrixKernel() override;
    
    virtual bool init(const SparseMatrix& A);
    
    virtual void preprocess(const SparseMatrix& A) {}
    
    virtual void hostFunction(const SparseMatrix& A, int iterNumber, int gridSize, int blockSize);

};
```

It is almost the same declaration as Matrix CPU kernel except that the
functionBody name has been changed with hostFunction and it takes
additional 2 parameters indicating the gridSize and the blockSize of the
current iteration.

Now it is time for the implementation file, which we will create under
the directory SparseViz/src/Kernel/Matrix/GPU.

*AddingGPUMatrixKernel.cu*

```cpp
#include "AddingGPUMatrixKernel.h"

#include "GPUKernels.cuh"

#include "cuda_runtime.h"

bool AddingGPUMatrixKernel::init(const SparseMatrix &A)

{

    if (// some checks may be done)
    
    {
    
        return false;
    
    }
    
    // Your initialization code goes here
    
    return true;

}

AddingGPUMatrixKernel::~AddingGPUMatrixKernel()

{

    // Your destructor code goes here

}

void AddingGPUMatrixKernel::hostFunction(const SparseMatrix &A, int iterNumber, int gridSize, int blockSize)

{

    // Your host function code goes here, including Cuda memcpy and malloc statements.
    
    GPUMatrixKernel<<<gridSize, blockSize>>>(// Your gpu kernel parameters goes here);
    
    // Your host function code continues from there, including Cuda memcpy and free statements.

}
```

Some parts require details in the above source code. First, it is
important to realize that the file does not have a normal cpp source
code extension but has a .cu extension that requires compilation to be
done by the nvcc compiler. As a second remark, there is an additional
include from which GPUMatrixKernel declaration is received. Users may
include their device kernel declarations into
SparseViz/include/Kernel/GPUKernels.cuh file. From there, the
appropriate device kernel can be called as shown by the hostFunction
which we have completed its implementation of. Device kernel declaration
is done under GPUKernels.cuh cuda header file looks like the following:

*GPUKernels.cuh*

```cpp
--- OTHER DEVICE KERNEL DECLARATIONS (CROPPED) ---

__global__ void GPUMatrixKernel(// Your device kernel parameters goes here);
```

Similarly, its implementation could be done in the
SparseViz/src/Kernel/GPUKernels.cu file that looks like the following:

*GPUKernels.cu*

```cpp
__global__ void GPUMatrixKernel(// Your device kernel parameters goes here)

{

    // Your device kernel implementation goes here

}
```

Having made all necessary declarations and implementations both for our
host and device functions, we can proceed to add them into
CMakeLists.txt. Because .cu files are compiled only when Cuda and
Cuda-capable GPU are installed, we are going to add the files into
CMakeLists.txt conditionally, like so:

*CMakeLists.txt*

```cpp
include(CheckLanguage)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)

    enable_language(CUDA)
    
    add_definitions(-DCUDA_ENABLED)
    
    --- OTHER HEADER FILE APPEND STATEMENTS (CROPPED) ---
    
    list(APPEND HEADER_FILES include/Kernel/Matrix/GPU/AddingGPUMatrixKernel.h)
    
    --- OTHER SOURCE FILE APPEND STATEMENTS (CROPPED) ---
    
    list(APPEND SOURCE_FILES src/Kernel/Matrix/GPU/AddingGPUMatrixKernel.cu)

endif()
```

As a final step, we will introduce our matrix GPU kernel definition into
the SparseVizEngine, enabling the library to recognize and utilize this
new kernel. The only part to pay attention to is that every code that we
are going to include should be within a conditional macro named
CUDA_ENABLED.

*SparseVizEngine.h*

```cpp
#ifdef CUDA_ENABLED

--- OTHER GPU KERNEL INCLUDES (CROPPED) ---

#include "AddingGPUMatrixKernel.h"

#endif
```

*SparseVizEngine.cpp*

```cpp
#ifdef CUDA_ENABLED

MatrixGPUKernel *SparseVizEngine::matrixGPUKernelFactory(const std::string &kernelName, const std::vector<int> &gridSizes, const std::vector<int> &blockSizes, int nRun, int nIgnore)

{

    --- OTHER GPU KERNEL CLASSES (CROPPED) ---
    
    else if (kernelName == "AddingGPUMatrixKernel")
    
    {
    
        return new AddingGPUMatrixKernel(kernelName, gridSizes, blockSizes, nRun, nIgnore);
    
    }
    
    return nullptr;

}

#endif
```

Finally, we can run or GPU matrix kernel from the config file by
indicating it under the section named \*GPU_MATRIX_KERNELS\*, as
follows:

*config*

```cpp
*GPU_MATRIX_KERNELS*

AddingGPUMatrixKernel | 4 | 256 | 10 | 2
```

This concludes tutorial 4.2, adding GPU matrix kernel into the SparseViz
library.