cmake_minimum_required(VERSION 3.22)
project(SparseViz)

set(CMAKE_CXX_STANDARD 20)

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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
include_directories(include/Testing)
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
include_directories(src/Testing)
include_directories(src/SparseVizLog)
include_directories(src/SparseVizPerformance)
include_directories(src/IO)
include_directories(MatrixVisualizationFiles)
include_directories(TensorVisualizationFiles)
include_directories(TestFiles)

set(HEADER_FILES
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
        include/Matrix/Orderings/amd.h
        include/Matrix/Orderings/AMDOrdering.h
        include/Tensor/Base/SparseTensor.h
        include/Matrix/Base/MatrixMarketIOLibrary.h
        include/Matrix/Base/Nonzero.h
        include/Tensor/Base/Nonzero3D.h
        include/Tensor/Orderings/TensorOrdering.h
        include/Tensor/Orderings/TensorNaturalOrdering.h
        include/Tensor/Orderings/TensorKPartiteOrdering.h
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
        include/IO/pigo.hpp
        include/Kernel/Matrix/CPU/Dijkstra.h
        include/Kernel/Matrix/CPU/Bellman_Ford.h
        include/Kernel/Matrix/CPU/Floyd_Warshall.h
        include/Testing/SparseVizTest.h
        include/SparseVizPerformance/SparseVizPerformance.h
)

set(SOURCE_FILES
        src/main.cpp
        include/ConfigurationFiles/config.cpp
        include/ConfigurationFiles/SparseVizEngine.cpp
        src/Matrix/Base/MatrixMarketIOLibrary.cpp
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
        src/Tensor/Base/SparseTensor.cpp
        src/Tensor/Orderings/TensorOrdering.cpp
        src/Tensor/Orderings/TensorNaturalOrdering.cpp
        src/Tensor/Orderings/TensorKPartiteOrdering.cpp
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
        src/Kernel/Matrix/CPU/Dijkstra.cpp
        src/Kernel/Matrix/CPU/Bellman_Ford.cpp
        src/Kernel/Matrix/CPU/Floyd_Warshall.cpp
        src/Testing/SparseVizTest.cpp
        src/SparseVizPerformance/SparseVizPerformance.cpp
)

file(READ " -- YOUR CONFIG FILE TO GET PROJECT INFO -- " configContent)
string(REGEX MATCH "RABBIT_LICENCE = ([^\n]*)" _ ${configContent})
set(rabbitLicence ${CMAKE_MATCH_1})
string(STRIP ${rabbitLicence} rabbitLicence)

if (NOT rabbitLicence STREQUAL "None")
    add_definitions(-DRABBIT_AVAILABLE)
    list(APPEND HEADER_FILES include/Matrix/Orderings/rabbit_order.hpp)
    list(APPEND HEADER_FILES include/Matrix/Orderings/RabbitOrdering.h)
    list(APPEND SOURCE_FILES src/Matrix/Orderings/RabbitOrdering.cpp)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DCUDA_ENABLED)
    list(APPEND HEADER_FILES include/Kernel/Matrix/GPU/MatrixGPUKernel.h)
    list(APPEND HEADER_FILES include/Kernel/Tensor/GPU/TensorGPUKernel.h)
    list(APPEND HEADER_FILES include/Kernel/Tensor/GPU/COOTensorGPUKernelSoA.h)
    list(APPEND HEADER_FILES include/Kernel/Tensor/GPU/COOTensorGPUKernelAoS.h)
    list(APPEND HEADER_FILES include/Kernel/GPUKernelResult.h)
    list(APPEND HEADER_FILES include/Kernel/GPUKernels.cuh)
    list(APPEND HEADER_FILES include/Kernel/Matrix/GPU/CudaHelloWorld.h)
    list(APPEND HEADER_FILES include/Kernel/Matrix/GPU/GPUSpmvRowBased.h)
    list(APPEND SOURCE_FILES src/Kernel/Matrix/GPU/MatrixGPUKernel.cu)
    list(APPEND SOURCE_FILES src/Kernel/Tensor/GPU/TensorGPUKernel.cu)
    list(APPEND SOURCE_FILES src/Kernel/Tensor/GPU/COOTensorGPUKernelAoS.cu)
    list(APPEND SOURCE_FILES src/Kernel/Tensor/GPU/COOTensorGPUKernelSoA.cu)
    list(APPEND SOURCE_FILES src/Kernel/GPUKernels.cu)
    list(APPEND SOURCE_FILES src/Kernel/Matrix/GPU/CudaHelloWorld.cu)
    list(APPEND SOURCE_FILES src/Kernel/Matrix/GPU/GPUSpmvRowBased.cu)
endif()

add_executable(SparseViz
        ${SOURCE_FILES}
        README.md
)

target_link_libraries(SparseViz ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
target_link_libraries(SparseViz --YOUR_PATOH_LIBRARY_FOLDER--/libpatoh.a)
target_link_libraries(SparseViz --YOUR_AMD_BUILD_FOLDER--/libamd.a)
target_link_libraries(SparseViz libnuma.a)
