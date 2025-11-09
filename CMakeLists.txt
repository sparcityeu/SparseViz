cmake_minimum_required(VERSION 3.22)
project(SparseViz)

# C++ Standard
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Cuda
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    enable_language(CUDA)
    add_definitions(-DCUDA_ENABLED)
endif()

# MPI
find_package(MPI QUIET)
if(MPI_FOUND)
    add_definitions(-DMPI_ENABLED)
    include_directories(${MPI_INCLUDE_PATH})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
endif()

# CURL
find_package(CURL REQUIRED)

# Directories
include_directories(lib)
include_directories(include)
include_directories(include/ConfigurationFiles)
include_directories(include/Matrix)
include_directories(include/Matrix/Base)
include_directories(include/Matrix/Orderings)
include_directories(include/Matrix/Visualization)
include_directories(include/Tensor)
include_directories(include/Tensor/Base)
include_directories(include/Tensor/Orderings)
include_directories(include/Tensor/Visualization)
include_directories(include/Kernel)
include_directories(include/Kernel/Matrix)
include_directories(include/Kernel/Matrix/CPU)
include_directories(include/Kernel/Matrix/GPU)
include_directories(include/Kernel/Tensor)
include_directories(include/Kernel/Tensor/CPU)
include_directories(include/Kernel/Tensor/GPU)
include_directories(include/SparseVizLog)
include_directories(include/SparseVizPerformance)
include_directories(include/IO)
include_directories(src)
include_directories(src/Matrix)
include_directories(src/Matrix/Base)
include_directories(src/Matrix/Orderings)
include_directories(src/Matrix/Visualization)
include_directories(src/Tensor)
include_directories(src/Tensor/Base)
include_directories(src/Tensor/Orderings)
include_directories(src/Tensor/Visualization)
include_directories(src/Kernel)
include_directories(src/Kernel/Matrix)
include_directories(src/Kernel/Matrix/CPU)
include_directories(src/Kernel/Matrix/GPU)
include_directories(src/Kernel/Tensor)
include_directories(src/Kernel/Tensor/CPU)
include_directories(src/Kernel/Tensor/GPU)
include_directories(src/SparseVizLog)
include_directories(src/SparseVizPerformance)
include_directories(src/IO)
include_directories(MatrixVisualizationFiles)
include_directories(TensorVisualizationFiles)

# Common Files
set(COMMON_HEADER_FILES
        include/ConfigurationFiles/config.h
        include/ConfigurationFiles/SparseVizEngine.h
        include/Matrix/Orderings/TrivialOrdering.h
        include/Matrix/Orderings/MatrixOrdering.h
        include/Matrix/Base/SparseMatrix.h
        include/helpers.h
        include/Matrix/Visualization/MatrixVisualizer.h
        include/json.hpp
        include/Parameters.h
        include/Matrix/Orderings/NaturalOrdering.h
        include/Matrix/Orderings/RCMOrdering.h
        include/Matrix/Orderings/GrayOrdering.h
        include/Matrix/Orderings/BOBAOrdering.h
        include/Matrix/Orderings/SlashBurnOrdering.h
        include/Matrix/Orderings/PatohOrdering.h
        include/Matrix/Orderings/DynaDegOrdering.h
        include/Matrix/Orderings/MinDegOrdering.h
        include/Matrix/Orderings/OrderingHelpers.h
        include/Matrix/Orderings/MillerPritikin.h
        include/Matrix/Orderings/LevelBasedSweep.h
        include/Matrix/Orderings/FileOrdering.h
        include/Matrix/Orderings/amd.h
        include/Matrix/Orderings/AMDOrdering.h
        include/Tensor/Base/SparseTensor.h
        include/IO/MatrixMarketIOLibrary.h
        include/Tensor/Orderings/TensorOrdering.h
        include/Tensor/Orderings/TensorNaturalOrdering.h
        include/Tensor/Orderings/COOKPartiteOrdering.h
        include/Tensor/Visualization/TensorVisualizer.h
        include/Kernel/Matrix/CPU/MatrixKernelFunction.h
        include/Kernel/Tensor/CPU/TensorKernelFunction.h
        include/Kernel/Tensor/CPU/COOTensorKernel.h
        include/Kernel/KernelResult.h
        include/Kernel/Matrix/CPU/SpmvRowBased.h
        include/Kernel/Matrix/CPU/SpmmRowBased.h
        include/Kernel/Matrix/CPU/Bfs.h
        include/Kernel/Matrix/CPU/SequentialBfs.h
        include/SparseVizLog/SparseVizLogger.h
        include/IO/SparseVizIO.h
        include/SparseVizPerformance/SparseVizPerformance.h
        include/Tensor/Base/SparseTensorCOO.h
        include/sort.h
)

set(COMMON_SOURCE_FILES
        src/main.cpp
        include/ConfigurationFiles/config.cpp
        include/ConfigurationFiles/SparseVizEngine.cpp
        src/IO/MatrixMarketIOLibrary.cpp
        src/Matrix/Orderings/TrivialOrdering.cpp
        src/Matrix/Orderings/MatrixOrdering.cpp
        src/Matrix/Base/SparseMatrix.cpp
        src/helpers.cpp
        src/Matrix/Visualization/MatrixVisualizer.cpp
        src/Parameters.cpp
        src/Matrix/Orderings/NaturalOrdering.cpp
        src/Matrix/Orderings/RCMOrdering.cpp
        src/Matrix/Orderings/GrayOrdering.cpp
        src/Matrix/Orderings/BOBAOrdering.cpp
        src/Matrix/Orderings/SlashBurnOrdering.cpp
        src/Matrix/Orderings/PatohOrdering.cpp
        src/Matrix/Orderings/DynaDegOrdering.cpp
        src/Matrix/Orderings/MinDegOrdering.cpp
        src/Matrix/Orderings/AMDOrdering.cpp
        src/Matrix/Orderings/OrderingHelpers.cpp
        src/Matrix/Orderings/MillerPritikin.cpp
        src/Matrix/Orderings/LevelBasedSweep.cpp
        src/Matrix/Orderings/FileOrdering.cpp
        src/Tensor/Base/SparseTensor.cpp
        src/Tensor/Orderings/TensorOrdering.cpp
        src/Tensor/Orderings/TensorNaturalOrdering.cpp
        src/Tensor/Orderings/COOKPartiteOrdering.cpp
        src/Tensor/Visualization/TensorVisualizer.cpp
        src/Kernel/Matrix/CPU/MatrixKernelFunction.cpp
        src/Kernel/Tensor/CPU/TensorKernelFunction.cpp
        src/Kernel/Tensor/CPU/COOTensorKernel.cpp
        src/Kernel/Matrix/CPU/SpmvRowBased.cpp
        src/Kernel/Matrix/CPU/SpmmRowBased.cpp
        src/Kernel/Matrix/CPU/Bfs.cpp
        src/Kernel/Matrix/CPU/SequentialBfs.cpp
        src/SparseVizLog/SparseVizLogger.cpp
        src/IO/SparseVizIO.cpp
        src/SparseVizPerformance/SparseVizPerformance.cpp
        src/Tensor/Base/SparseTensorCOO.cpp
        src/sort.cpp
)

# Rabbit Files
set(RABBIT_HEADER_FILES
        include/Matrix/Orderings/rabbit_order.hpp
        include/Matrix/Orderings/RabbitOrdering.h
)

set(RABBIT_SOURCE_FILES
        src/Matrix/Orderings/RabbitOrdering.cpp
)

# GPU Files
set(GPU_HEADER_FILES
        include/GPUHelpers.h
        include/Kernel/GPUKernelResult.h
        include/Kernel/Matrix/GPU/MatrixGPUKernel.h
        include/Kernel/Tensor/GPU/TensorGPUKernel.h
        include/Kernel/Tensor/GPU/COOTensorGPUKernelSoA.h
        include/Kernel/Tensor/GPU/COOTensorGPUKernelAoS.h
)

set(GPU_SOURCE_FILES
        src/Kernel/Matrix/GPU/MatrixGPUKernel.cu
        src/Kernel/Tensor/GPU/TensorGPUKernel.cu
        src/Kernel/Tensor/GPU/COOTensorGPUKernelAoS.cu
        src/Kernel/Tensor/GPU/COOTensorGPUKernelSoA.cu
)

# MPI Files
set(MPI_HEADER_FILES

)

set(MPI_SOURCE_FILES

)

# All Files
set(FILES
        ${COMMON_HEADER_FILES}
        ${COMMON_SOURCE_FILES}
)

# Add Rabbit Files to All Files if Rabbit License is found
file(READ "/home/delbek/sparseviz/deniz_matrix.config" configContent)
string(REGEX MATCH "RABBIT_LICENCE = ([^\n]*)" _ ${configContent})
set(rabbitLicence ${CMAKE_MATCH_1})
string(STRIP ${rabbitLicence} rabbitLicence)

if (NOT rabbitLicence STREQUAL "None")
    add_definitions(-DRABBIT_AVAILABLE)
    list(APPEND FILES ${RABBIT_HEADER_FILES})
    list(APPEND FILES ${RABBIT_SOURCE_FILES})
endif()

# Add GPU Files to All Files if CUDA is found
if(CUDAToolkit_FOUND)
    list(APPEND FILES ${GPU_HEADER_FILES})
    list(APPEND FILES ${GPU_SOURCE_FILES})
endif()

# Add MPI Files to All Files if MPI is found
if(MPI_FOUND)
    list(APPEND FILES ${MPI_HEADER_FILES})
    list(APPEND FILES ${MPI_SOURCE_FILES})
endif()

# Create executable
add_executable(SparseViz
        ${FILES}
)

# Force CMake to use the C++ linker driver
set_target_properties(SparseViz PROPERTIES
    LINKER_LANGUAGE CXX
)

# Link libraries
target_link_libraries(SparseViz PRIVATE
    CURL::libcurl
    stdc++
    /home/delbek/sparseviz/lib/libpatoh.a
    /home/delbek/sparseviz/lib/libamd.a
    numa
    OpenMP::OpenMP_CXX
    $<$<BOOL:${MPI_FOUND}>:MPI::MPI_CXX>
    $<$<BOOL:${CUDAToolkit_FOUND}>:cuda>
    $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::cudart>
    $<$<BOOL:${CUDAToolkit_FOUND}>:${CUDA_LIBRARIES}>
)

# Options
target_compile_options(SparseViz PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -g -std=c++17>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info --std=c++17>
)
