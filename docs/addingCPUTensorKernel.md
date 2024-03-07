# Tutorial 5.1: How to Add CPU Tensor Kernel

SparseViz allows 2 types of tensor kernels to be added to the library
externally, CPU and GPU kernels. All CPU kernels can be written in
parallel using OpenMP. All GPU kernels, on the other hand, can be
written in parallel using Cuda. In this tutorial, we will analyze CPU
tensor kernels, especially how they can be added to the SparseViz
library.

All tensor kernels should be derived from the abstract class named
TensorKernelFunction in the SparseViz library. This abstract class
provides various functionalities to ease the implementation of derived
classes aiming to provide kernels for sparse tensors.
TensorKernelFunction dictates all its child classes to override 3 public
pure virtual functions to complete their implementation. These methods
are:

```cpp
virtual bool init(const SparseTensor& A);
```

This method could be thought of as a constructor for the child class. It
is guaranteed that this method will be called before the execution of
every tensor kernel. The only difference it has with a traditional
constructor is that it is a boolean-returning function. In case there
are some circumstances in which user-defined tensor kernels are not
desired to be executed on some type of sparse matrices, it can easily be
programmed within that init method with a return value of "false" that
would eliminate the run of subsequent methods of the class, ultimately
leading to the termination of the execution of the kernel.

```cpp
virtual void preprocess(const SparseTensor& A);
```

Tensor kernels could be executed from within the config file with a
specification of the number of iterations to be given as an argument.
This iteration number could be from 1 to many, depending on the user's
preference. This iteration and the for loop that enables it is
controlled by the base class TensorKernelFunction. The preprocess method
comes into play at the very first method to be called upon at the start
of the iteration. In the implementation of this method, we give
opportunity to users to preprocess any data or data structures, be it
their initialization or their population, to be used during the current
iteration of the for loop. It is guaranteed that this preprocess will be
called as the first method in each iteration and the duration it took to
complete this method will never be reflected in the overall duration
kernel has taken to be executed. Because users cannot control the flow
of the tensor kernel execution and thus cannot pass any additional
parameters to the functionBody method - actual kernel function -, the
only way it can use the variables initialized in it is by declaring them
as member variables of the class to be used them in the actual kernel
function.

```cpp
virtual void functionBody(const SparseTensor& A, int iterNumber);
```

As a last step user-defined child classes should complete their
implementation by providing the last pure virtual function a definition
that will carry out the actual logic of the tensor kernel. Within that
method, it is expected the users write the logic of the tensor kernel
that is executed on every ordered tensor indicated in the config file -
except ones that failed to proceed to this function because the init
method of the kernel returned false for them -. This function accepts an
additional integer parameter named iterNumber, which indicates the
number of consecutive times the functionBody has been called.

One important remark we need to make for our tensor kernel structure in
SparseViz is that during the execution of a kernel, all other threads
working for SparseViz process are temporarily stopped to be able to
provide all computational resources to the actual kernel execution and
to make experiments as realistic as possible.

Having learned each method we need to override in user-defined kernels
we can complete the declaration and the implementation of our kernel
named AddingCPUTensorKernel under the directories
SparseViz/include/Kernel/Tensor/CPU and SparseViz/src/Kernel/Tensor/CPU,
respectively.

*AddingCPUTensorKernel.h*

```cpp
#include "TensorKernelFunction.h"

class AddingCPUTensorKernel: public TensorKernelFunction

{

public:

    AddingCPUTensorKernel(const std::string& kernelName, const std::vector<int>& threadCounts, const std::string& schedulingPolicy, int chunkSize, int nRun, int nIgnore)
    :   TensorKernelFunction(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore) {}
    
    virtual ~AddingCPUTensorKernel() override;
    
    virtual bool init(const SparseTensor& A) override;
    
    virtual void preprocess(const SparseTensor& A) override;
    
    virtual void functionBody(const SparseTensor& A, int iterNumber);

};
```

One extension we made for the class declaration on top of the 3 methods
that TensorKernelFunction dictates its child class to override is the
virtual destructor. If anywhere during the init or in the preprocess
method you allocated memory on the heap or you made a process requiring
a cleanup (like opening a file or making a database connection), then in
the virtual destructor, it is also your responsibility to clean all them
up. Also, in the constructor that has an empty implementation, we did
not forget to initialize the parent class with the parameters provided
to the child class constructor. To remember what these parameters refer
to, you can do so by looking at tutorial 1, how to use the config file.

*AddingCPUTensorKernel.cpp*

```cpp
#include "AddingCPUTensorKernel.h"

bool AddingCPUTensorKernel::init(const SparseTensor &A)

{

    if (// some checks may be done)
    
    {
    
        return false;
    
    }
    
    // Your initialization code goes here
    
    return true;

}

AddingCPUTensorKernel::~AddingCPUTensorKernel()

{

    // Your destructor code goes here.

}

void AddingCPUTensorKernel::preprocess(const SparseTensor& A)

{

    // Your preprocess code goes here.

}

void AddingCPUTensorKernel::functionBody(const SparseTensor& A, int iterNumber)

{

    // Your function body code goes here.

}
```

The remaining steps to integrate the tensor kernel into SparseViz
closely mirror those for adding tensor ordering. Initially, we must
incorporate our class declaration and implementation into the
CMakeLists.txt. Following this, we will introduce our tensor kernel
definition into the SparseVizEngine, enabling the library to recognize
and utilize this new kernel.

*CMakeLists.txt*

```cpp
set(HEADER_FILES

    --- OTHER HEADER FILES (CROPPED) ---
    
    include/Kernel/Tensor/CPU/AddingCPUTensorKernel.h

)

set(SOURCE_FILES

    --- OTHER SOURCE FILES (CROPPED) ---
    
    src/Kernel/Tensor/CPU/AddingCPUTensorKernel.cpp

)
```

*SparseVizEngine.h*

```cpp
#include "AddingCPUTensorKernel.h"
```

*SparseVizEngine.cpp*

```cpp
TensorKernelFunction *SparseVizEngine::tensorKernelFactory(const std::string &kernelName, const std::vector<int>& threadCounts, const std::string &schedulingPolicy, int chunkSize, int nRun, int nIgnore)

{

    --- OTHER KERNEL CLASSES (CROPPED) ---
    
    else if (kernelName == "AddingCPUTensorKernel")
    
    {
    
        return new AddingCPUTensorKernel(kernelName, threadCounts, schedulingPolicy, chunkSize, nRun, nIgnore);
    
    }
    
    return nullptr;

}
```

Having made all of these steps, we can safely run our library with an
absolute path pointing to our config file in which we indicated our new
tensor kernel named AddingCPUTensorKernel under the section
\*TENSOR_KERNELS\* to make it get executed on each ordered matrices,
like so:

*config*

```cpp
*TENSOR_KERNELS*

AddingCPUTensorKernel | WeHaveAddedACPUTensorKernel | 1/2/4/6/8/16 | dynamic | 256 | 10 | 2
```

This concludes tutorial 5.1, the way to add CPU tensor kernels into the
library. In tutorial 5.2, its GPU counterpart will be explained.