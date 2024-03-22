//
// Created on 1/15/24.
//

#include "TensorOrdering.h"
#include <cmath>
#include "helpers.h"
#include <string.h>
#include "SparseVizIO.h"
#include "SparseVizPerformance.h"


TensorOrdering::TensorOrdering(SparseTensor &tensor,  const std::vector<vType>& active_modes, std::string orderingName, unsigned int orderingFunctionPrice)
:   tensor(tensor),
    orderedDimensions(nullptr),
    active_modes(active_modes),
    orderingName(orderingName)
{
    this->calculateOrderingPrice(orderingFunctionPrice);
#ifdef CONFIG_LOG
    std::cout << "Ordering price of the " << orderingName << " on " << this->getTensor().getName() << " is determined to be " << orderingPrice << std::endl;
#endif
}

SparseTensor &TensorOrdering::getTensor()
{
    return tensor;
}

vType **TensorOrdering::getOrderedDimensions() const
{
    if (!orderedDimensions)
    {
       // throw std::runtime_error("The generate ordering method should have been called for one time before this method returns the ordered dimensions!");
       return nullptr;
    }
    else
    {
        return orderedDimensions;
    }
}

TensorOrdering::~TensorOrdering()
{
    if (orderedDimensions)
    {
        for (int i = 0; i != tensor.getOrder(); ++i)
        {
            delete [] orderedDimensions[i];
        }
        delete [] orderedDimensions;
    }
}

void TensorOrdering::pushKernelResult(const KernelResult& kernelResult)
{
    kernelResults.push_back(kernelResult);
}

void TensorOrdering::pushGPUKernelResult(const GPUKernelResult &kernelResult)
{
    gpuKernelResults.push_back(kernelResult);
}

void TensorOrdering::generateOrdering(bool writeOrdering)
{
    if (!USE_EXISTING_ORDERINGS || !this->readOrdering())
    {

        std::vector<BenchmarkSettings> settings = getSettings();
        SparseVizPerformance* perf = new SparseVizPerformance(settings.data(), settings.size());

        double start_time = omp_get_wtime();

        perf->activatePerf();
        this->orderingFunction();
        perf->deactivatePerf();

        this->checkPermutationCorrectness();

        double end_time = omp_get_wtime();

        logger.logTensorOrdering(this, end_time - start_time, perf);

        delete perf;

        if(writeOrdering && EXPORT_ORDERINGS)
        {
            this->writeOrdering();
        }
    }
}

void TensorOrdering::checkPermutationCorrectness() {
    const vType* dims = tensor.getDims();
    std::vector<bool>* markers = new std::vector<bool>[tensor.getOrder()];
    //std::cout << "CPC" << tensor.getName() << " " << this->getOrderingName() << " " << active_modes[0] << " " << active_modes[1] << " " << active_modes[2] << std::endl;
    for(const vType &m : active_modes) {
        markers[m].resize(dims[m], false);
   
        for (int j = 0; j != dims[m]; ++j) {
            if ((orderedDimensions[m][j] < 0) || orderedDimensions[m][j] >= dims[m]) {
                throw std::logic_error("The " + std::to_string(m) + " mode order for " + orderingName + " is not generated correctly: " + std::to_string(j) + " has an ordering value " + std::to_string(orderedDimensions[m][j]));
            }
            markers[m][orderedDimensions[m][j]] = true;
        }
  
        for (int j = 0; j != dims[m]; ++j) {
            if (markers[m][j] == false) {
                throw std::logic_error("The " + std::to_string(m) + " mode order for " + orderingName + " is not generated correctly: " + std::to_string(j) + " is not covered ");
            }
        }
    }

    delete[] markers;
}

bool TensorOrdering::readOrdering()
{
    std::string binaryFileName = TENSOR_ORDERING_FILES_DIR + orderingName + "_" + this->getTensor().getName();
    for(const vType& m : active_modes) {
        binaryFileName += "_" + std::to_string(m);
    }
    binaryFileName += ".bin";

    double start_time = omp_get_wtime();
    bool ret = SparseVizIO::readTensorOrderingFromBinaryFile(binaryFileName, tensor.getOrder(), this->getTensor().getDims(), orderedDimensions);
    double end_time = omp_get_wtime();
    if (ret)
    {
        logger.logReadingTensorOrdering(this, end_time - start_time);
    }

    return ret;
}

void TensorOrdering::writeOrdering()
{
    std::string binaryFileName = TENSOR_ORDERING_FILES_DIR + orderingName + "_" + this->getTensor().getName(); 
    for(const vType& m : active_modes) {
        binaryFileName += "_" + std::to_string(m);
    }
    binaryFileName += ".bin";

    double start_time = omp_get_wtime();
    SparseVizIO::writeTensorOrderingToBinaryFile(binaryFileName, tensor.getOrder(), this->getTensor().getDims(), orderedDimensions);
    double end_time = omp_get_wtime();

    logger.logWritingTensorOrdering(this, end_time - start_time);
}

void TensorOrdering::calculateOrderingPrice(unsigned int orderingFunctionPrice)
{
    const double nnzBaseFactor = 0.05;
    const double dimensionBaseFactor = 0.001;
    const vType* dims = this->getTensor().getDims();

    double nnzPrice = log(this->getTensor().getNNZCount() + 1) * nnzBaseFactor;
    double dimensionPrice = 0;
    for (int i = 0; i != tensor.getOrder(); ++i)
    {
        dimensionPrice += log(dims[i] + 1) * dimensionBaseFactor;
    }

    orderingPrice = orderingFunctionPrice + nnzPrice + dimensionPrice;
}
