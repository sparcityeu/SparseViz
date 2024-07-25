#include "config.h"
#include "helpers.h"
#include "SparseVizEngine.h"


SparseVizLogger* logger = new SparseVizLogger;
SparseVizPerformance* sparseVizPerformance = new SparseVizPerformance;
bool TIMING_LOG = true;
std::string PROJECT_DIR = "-1";
double MAX_TIME_BEFORE_ABORTING_ORDERING = 1000;
bool EXPORT_ORDERED_SPARSE_STRUCTURES = true;
bool USE_EXISTING_ORDERED_SPARSE_STRUCTURES = true;
bool EXPORT_ORDERINGS = true;
bool USE_EXISTING_ORDERINGS = true;
std::string MATRIX_FILES_DIR = "-1";
std::string MATRIX_ORDERING_FILES_DIR = "-1";
std::string MATRIX_VISUALIZATION_FILES_DIR = "-1";
std::string TENSOR_FILES_DIR = "-1";
std::string TENSOR_ORDERING_FILES_DIR = "-1";
std::string TENSOR_VISUALIZATION_FILES_DIR = "-1";
std::string LOGO_PATH;
std::string FAVICON_PATH;
std::string ZOO_TYPE = "-1";
std::string CHART_TYPE = "-1";
unsigned int MAX_DIM = 64;
std::string TEST_DIRECTORY = "/home/users/kaya/SparseViz/TestFiles/";
std::string TEST_CONFIG = TEST_DIRECTORY + "template_test_file";
TensorType TENSOR_STORAGE_TYPE;
BlockType BLOCK_SIZE;
bool ORDERING_PERFORMANCE_LOG = false;
bool KERNEL_PERFORMANCE_LOG = false;


ConfigFileReader::ConfigFileReader(const std::string& configFile)
:   m_Engine(nullptr)
{
    m_File.open(configFile);

    if (!m_File.is_open())
    {
        throw std::runtime_error("The configuration file (" + configFile + ") could not be opened.");
    }
}

ConfigFileReader::~ConfigFileReader()
{
    m_File.close();
#ifndef TEST
    logger->createCSVFile(PROJECT_DIR + m_LogFilePath);
#endif
    delete m_Engine;
    delete sparseVizPerformance;
    delete logger;
}

void ConfigFileReader::readConfigFile()
{
    std::string line;
    std::string section;

    while (getline(m_File, line))
    {
        // Skip empty lines and comments
        if (line.empty() || line.substr(0, 2) == "//") continue;

        std::vector<std::string> lineCommentsDeleted = split(line, ' ');
        for (auto i = lineCommentsDeleted.begin(); i != lineCommentsDeleted.end(); ++i)
        {
            if (i->substr(0, 2) == "//")
            {
                lineCommentsDeleted.erase(i, lineCommentsDeleted.end());
                break;
            }
        }
        line = merge(lineCommentsDeleted);

        if (line == "*SETTINGS*")
        {
            section = "SETTINGS";
        }
        else if (line == "*MATRICES*")
        {
            section = "MATRICES";
        }
        else if (line == "*MATRIX_ORDERINGS*")
        {
            section = "MATRIX_ORDERINGS";
        }
        else if (line == "*TENSORS*")
        {
            section = "TENSORS";
        }
        else if (line == "*TENSOR_ORDERINGS*")
        {
            section = "TENSOR_ORDERINGS";
        }
        else if (line == "*MATRIX_KERNELS*")
        {
            section = "MATRIX_KERNELS";
        }
        else if (line == "*GPU_MATRIX_KERNELS*")
        {
            section = "GPU_MATRIX_KERNELS";
        }
        else if (line == "*GPU_TENSOR_KERNELS*")
        {
            section = "GPU_TENSOR_KERNELS";
        }
        else if (line == "*TENSOR_KERNELS*")
        {
            section = "TENSOR_KERNELS";
        }
        else
        {
            if (section == "SETTINGS")
            {
                this->readSetting(line);
            }
            else if (section == "MATRICES")
            {
                if (checkZooTypeMatch(section))
                {
                    m_Matrices.push_back(MATRIX_FILES_DIR + line);
                }
            }
            else if (section == "MATRIX_ORDERINGS")
            {
                if (checkZooTypeMatch(section))
                {
                    m_MatrixOrderings.push_back(line);
                }
            }
            else if (section == "TENSORS")
            {
                if (checkZooTypeMatch(section))
                {
                    m_Tensors.push_back(line);
                }
            }
            else if (section == "TENSOR_ORDERINGS")
            {
                static bool fullTensorRead = false;
                if (checkZooTypeMatch(section) && !fullTensorRead)
                {
                    m_TensorOrderings.push_back(line);
                    if (ZOO_TYPE == "FULLTENSOR")
                    {
                      //  fullTensorRead = true;
                    }
                }
            }
            else if (section == "MATRIX_KERNELS")
            {
                if (checkZooTypeMatch(section))
                {
                    m_MatrixKernels.push_back(line);
                }
            }
            else if (section == "GPU_MATRIX_KERNELS")
            {
                if (checkZooTypeMatch(section))
                {
                    m_GPUMatrixKernels.push_back(line);
                }
            }
            else if (section == "GPU_TENSOR_KERNELS")
            {
                if (checkZooTypeMatch(section))
                {
                    m_GPUTensorKernels.push_back(line);
                }
            }
            else if (section == "TENSOR_KERNELS")
            {
                if (checkZooTypeMatch(section))
                {
                    m_TensorKernels.push_back(line);
                }
            }
        }
    }
    this->validateConfigStatus();
}

SparseVizEngine *ConfigFileReader::instantiateEngine()
{
    this->readConfigFile();
    m_Engine = new SparseVizEngine();

    for (const auto& matrix: m_Matrices)
    {       
        m_Engine->addMatrix(matrix);
    }
    m_Engine->constructMatrices();

    for (const auto& matrixOrd: m_MatrixOrderings)
    {
        this->readMatrixOrdering(matrixOrd);
    }

    for (const auto& tensor: m_Tensors)
    {
        this->readTensor(tensor);
    }
    m_Engine->constructTensors();

    for (const auto& tensorOrd: m_TensorOrderings)
    { 
        this->readTensorOrdering(tensorOrd);
    }

    for (const auto& matrixKer: m_MatrixKernels)
    { 
        this->readMatrixKernel(matrixKer);
    }

    for (const auto& gpuMatrixKer: m_GPUMatrixKernels)
    { 
#ifdef CUDA_ENABLED
        this->readGPUMatrixKernel(gpuMatrixKer);
#else
        std::cout << "WARNING: CUDA SUPPORT COULD NOT BE FOUND! YOUR GPU MATRIX KERNELS WILL NOT BE EXECUTED ON ORDERED MATRICES." << std::endl;
        break;
#endif
    }

    for (const auto& gpuTensorKer: m_GPUTensorKernels)
    {
#ifdef CUDA_ENABLED
        this->readGPUTensorKernel(gpuTensorKer);
#else
        std::cout << "WARNING: CUDA SUPPORT COULD NOT BE FOUND! YOUR GPU TENSOR KERNELS WILL NOT BE EXECUTED ON ORDERED TENSORS." << std::endl;
        break;
#endif
    }

    for (const auto& tensorKer: m_TensorKernels)
    {
        this->readTensorKernel(tensorKer);
    }

    m_Engine->generateMatrixOrderings();
    m_Engine->generateTensorOrderings();
    return m_Engine;
}

void ConfigFileReader::readSetting(const std::string& line)
{
    std::vector<std::string> lineSplitted = split(line, '=');

    if (lineSplitted[0] == "LOG_FILE")
    {
        m_LogFilePath = lineSplitted[1];
    }
    else if (lineSplitted[0] == "TIMING_LOG")
    {
        TIMING_LOG = (lineSplitted[1] == "TRUE");
    }
    else if (lineSplitted[0] == "PROJECT_DIR")
    {
        PROJECT_DIR = lineSplitted[1];
    }
    else if (lineSplitted[0] == "MAX_TIME_BEFORE_ABORTING_ORDERING")
    {
        try
        {
            MAX_TIME_BEFORE_ABORTING_ORDERING = std::stoi(lineSplitted[1]);
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for MAX_TIME_BEFORE_ABORTING_ORDERING");
        }
    }
    else if (lineSplitted[0] == "MATRIX_FILES_DIR")
    {
        MATRIX_FILES_DIR = lineSplitted[1];
    }
    else if (lineSplitted[0] == "CHART_TYPE")
    {
        CHART_TYPE = lineSplitted[1];
    }
    else if (lineSplitted[0] == "TENSOR_FILES_DIR")
    {
        TENSOR_FILES_DIR = lineSplitted[1];
    }
    else if (lineSplitted[0] == "TENSOR_ORDERING_FILES_DIR")
    {
        TENSOR_ORDERING_FILES_DIR = lineSplitted[1];
    }
    else if (lineSplitted[0] == "TENSOR_VISUALIZATION_FILES_DIR")
    {
        TENSOR_VISUALIZATION_FILES_DIR = lineSplitted[1];
    }
    else if (lineSplitted[0] == "EXPORT_ORDERED_SPARSE_STRUCTURES")
    {
        EXPORT_ORDERED_SPARSE_STRUCTURES = (lineSplitted[1] == "TRUE" || lineSplitted[1] == "true");
    }
    else if (lineSplitted[0] == "USE_EXISTING_ORDERED_SPARSE_STRUCTURES")
    {
        USE_EXISTING_ORDERED_SPARSE_STRUCTURES = (lineSplitted[1] == "TRUE" || lineSplitted[1] == "true");
    }
    else if (lineSplitted[0] == "EXPORT_ORDERINGS")
    {
        EXPORT_ORDERINGS = (lineSplitted[1] == "TRUE" || lineSplitted[1] == "true");
    }
    else if (lineSplitted[0] == "USE_EXISTING_ORDERINGS")
    {
        USE_EXISTING_ORDERINGS = (lineSplitted[1] == "TRUE" || lineSplitted[1] == "true");
    }
    else if (lineSplitted[0] == "MATRIX_ORDERING_FILES_DIR")
    {
        MATRIX_ORDERING_FILES_DIR = lineSplitted[1];
    }
    else if (lineSplitted[0] == "MATRIX_VISUALIZATION_FILES_DIR")
    {
        MATRIX_VISUALIZATION_FILES_DIR = lineSplitted[1];
    }
    else if (lineSplitted[0] == "LOGO_PATH")
    {
        LOGO_PATH = PROJECT_DIR + lineSplitted[1];
    }
    else if (lineSplitted[0] == "FAVICON_PATH")
    {
        FAVICON_PATH = lineSplitted[1];
    }
    else if (lineSplitted[0] == "ZOO_TYPE")
    {
        ZOO_TYPE = lineSplitted[1];
    }
    else if (lineSplitted[0] == "MAX_DIM")
    {
        try
        {
            MAX_DIM = std::stoi(lineSplitted[1]);
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for MAX_DIM");
        }
    }
    else if (lineSplitted[0] == "TENSOR_STORAGE_TYPE")
    {
        if (lineSplitted[1] == "COO")
        {
            TENSOR_STORAGE_TYPE = COO;
        }
        else if (lineSplitted[1] == "CSF")
        {
            TENSOR_STORAGE_TYPE = CSF;
        }
        else if (lineSplitted[1] == "HiCOO")
        {
            TENSOR_STORAGE_TYPE = HiCOO;
        }
        else
        {
            throw std::runtime_error("TENSOR_STORAGE_TYPE must be one of the following types: COO, CSF, HiCOO");
        }
    }
    else if (lineSplitted[0] == "BLOCK_SIZE")
    {
        try
        {
            BLOCK_SIZE = std::stoi(lineSplitted[1]);
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for BLOCK_SIZE");
        }
    }
    else if (lineSplitted[0] == "ORDERING_PERFORMANCE_LOG")
    {
        ORDERING_PERFORMANCE_LOG = (lineSplitted[1] == "ENABLED" || lineSplitted[1] == "enabled");
    }
    else if (lineSplitted[0] == "KERNEL_PERFORMANCE_LOG")
    {
        KERNEL_PERFORMANCE_LOG = (lineSplitted[1] == "ENABLED" || lineSplitted[1] == "enabled");
    }
}

void ConfigFileReader::readTensor(const std::string &line)
{
    std::vector<std::string> lineSplitted = split(line, '|');
    std::string tensorName = split(lineSplitted[0], '.')[0];
    std::vector<std::string> dimensionsStringSplitted;

    std::set<std::string> uniqueDimensions;
    std::vector<vType> dimensions;
    dimensions.reserve(3);

    if (lineSplitted.size() == 1)
    {
        dimensions.emplace_back(0);
        dimensions.emplace_back(1);
        dimensions.emplace_back(2);
        m_Engine->addTensor(TENSOR_FILES_DIR + lineSplitted[0], dimensions);
        return;
    }

    dimensionsStringSplitted = split(lineSplitted[1], '/');

    for (const auto& i : dimensionsStringSplitted)
    {
        uniqueDimensions.insert(i);
    }

    if (uniqueDimensions.size() != 3)
    {
        std::cout << "SKIPPING CONSTRUCTION: " << tensorName << " dims: ";
        std::cout << uniqueDimensions.size() << " - ";
        for(auto& dim : uniqueDimensions) {
            std::cout << dim << " ";
        } 
        std::cout << std::endl;
        std::cout << "Tensors visualizer works only with 3D tensors, you must select 3 different modes for your tensor: " << tensorName << std::endl;
        return;
    }

    try
    {
        for (const auto& dimension: uniqueDimensions)
        {
            dimensions.emplace_back(std::stoul(dimension));
        }
    }
    catch (const std::invalid_argument& e)
    {
        std::cout << "SKIPPING CONSTRUCTION: " << tensorName << std::endl;
        std::cout << "You gave invalid argument to the mode selection of the tensor: " << tensorName << std::endl;
        return;
    }

    m_Engine->addTensor(TENSOR_FILES_DIR + lineSplitted[0], dimensions);
}

void ConfigFileReader::readMatrixOrdering(const std::string &line)
{
    std::string orderingClassName = "";
    std::string orderingName = "NoName";
    std::string orderingParameters = "";

    std::vector<std::string> lineSplitted = split(line, '|');

    if(lineSplitted.size() >= 1)
    {
        orderingClassName = lineSplitted[0];
    }
    if(lineSplitted.size() >= 2)
    {
        orderingName = lineSplitted[1];
    }
    if(lineSplitted.size() >= 3)
    {
        orderingParameters = lineSplitted[2];
    }

    m_Engine->addMatrixOrdering(orderingClassName, orderingName, orderingParameters);
}

void ConfigFileReader::readTensorOrdering(const std::string &line)
{
    std::string orderingClassName = "";
    std::string orderingName = "NoName";
    std::string orderingParameters = "";

    std::vector<std::string> lineSplitted = split(line, '|');

    if(lineSplitted.size() >= 1)
    {
        orderingClassName = lineSplitted[0];
    }
    if(lineSplitted.size() >= 2)
    {
        orderingName = lineSplitted[1];
    }
    if(lineSplitted.size() >= 3)
    {
        orderingParameters = lineSplitted[2];
    }

    m_Engine->addTensorOrdering(orderingClassName, orderingName, orderingParameters, ZOO_TYPE);
}

bool ConfigFileReader::checkZooTypeMatch(const std::string &section)
{
    if ((ZOO_TYPE == "MAT" || ZOO_TYPE == "MATORD") && (section == "MATRICES" || section == "MATRIX_ORDERINGS" || section == "MATRIX_KERNELS" || section == "GPU_MATRIX_KERNELS"))
    {
        return true;
    }
    if ((ZOO_TYPE == "TENS" || ZOO_TYPE == "TENSORD" || ZOO_TYPE == "FULLTENSOR") && (section == "TENSORS" || section == "TENSOR_ORDERINGS" || section == "TENSOR_KERNELS" || section == "GPU_TENSOR_KERNELS"))
    {
        return true;
    }
    return false;
}

void ConfigFileReader::readMatrixKernel(const std::string &line)
{
    std::vector<std::string> lineSplitted = split(line, '|');

    int chunkSize, nRun, nIgnore;
    try
    {
        chunkSize = std::stoi(lineSplitted[3]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for CHUNK_SIZE of " + lineSplitted[0]);
    }

    try
    {
        nRun = std::stoi(lineSplitted[4]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for N_RUN of " + lineSplitted[0]);
    }
    try
    {
        nIgnore = std::stoi(lineSplitted[5]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for N_IGNORE of " + lineSplitted[0]);
    }

    std::vector<std::string> threadCountsString = split(lineSplitted[1], '/');
    std::vector<int> threadCounts;
    threadCounts.reserve(threadCountsString.size());
    for (int i = 0; i != threadCountsString.size(); ++i)
    {
        try
        {
            int threadCount = std::stoi(threadCountsString[i]);
            threadCounts.push_back(threadCount);
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for the thread counts of " + lineSplitted[0]);
        }
    }

    m_Engine->addMatrixKernel(lineSplitted[0], threadCounts, lineSplitted[2], chunkSize, nRun, nIgnore);
}

void ConfigFileReader::readTensorKernel(const std::string &line)
{
    std::vector<std::string> lineSplitted = split(line, '|');
    std::string kernelParameters = "";

    int chunkSize, nRun, nIgnore;
    try
    {
        chunkSize = std::stoi(lineSplitted[3]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for CHUNK_SIZE of " + lineSplitted[0]);
    }

    try
    {
        nRun = std::stoi(lineSplitted[4]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for N_RUN of " + lineSplitted[0]);
    }
    try
    {
        nIgnore = std::stoi(lineSplitted[5]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for N_IGNORE of " + lineSplitted[0]);
    }

    std::vector<std::string> threadCountsString = split(lineSplitted[1], '/');
    std::vector<int> threadCounts;
    threadCounts.reserve(threadCountsString.size());
    for (int i = 0; i != threadCountsString.size(); ++i)
    {
        try
        {
            int threadCount = std::stoi(threadCountsString[i]);
            threadCounts.push_back(threadCount);
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for the thread counts of " + lineSplitted[0]);
        }
    }

    if (lineSplitted.size() >= 7)
    {
        kernelParameters = lineSplitted[6];
    }

    m_Engine->addTensorKernel(lineSplitted[0], threadCounts, lineSplitted[2], chunkSize, nRun, nIgnore, kernelParameters);
}

#ifdef CUDA_ENABLED
void ConfigFileReader::readGPUMatrixKernel(const std::string &line)
{
    std::vector<std::string> lineSplitted = split(line, '|');

    std::string kernelClassName = lineSplitted[0];
    std::string kernelName = lineSplitted[1];
    std::vector<std::string> gridSizesString = split(lineSplitted[2], '/');
    std::vector<std::string> blockSizesString = split(lineSplitted[3], '/');
    std::vector<std::string> sharedMemorySizesString = split(lineSplitted[4], '/');

    if (gridSizesString.size() != blockSizesString.size() && blockSizesString.size() != sharedMemorySizesString.size())
    {
        throw std::runtime_error("Length of the gridSizes, blockSizes, and sharedMemorySizes does not match for " + kernelName);
    }

    std::vector<int> gridSizes(gridSizesString.size());
    std::vector<int> blockSizes(gridSizesString.size());
    std::vector<int> sharedMemorySizes(gridSizesString.size());
    for (int i = 0; i != gridSizesString.size(); ++i)
    {
        try
        {
            int gridSize = std::stoi(gridSizesString[i]);
            gridSizes[i] = gridSize;
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for the gridSizes of " + kernelName);
        }

        try
        {
            int blockSize = std::stoi(blockSizesString[i]);
            blockSizes[i] = blockSize;
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for the blockSizes of " + kernelName);
        }

        try
        {
            int sharedMemorySize = std::stoi(sharedMemorySizesString[i]);
            sharedMemorySizes[i] = sharedMemorySize;
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for the sharedMemorySizes of " + kernelName);
        }
    }

    std::string kernelClassParameters = lineSplitted[5];

    int nRun, nIgnore;
    try
    {
        nRun = std::stoi(lineSplitted[6]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for the nRun of " + kernelName);
    }

    try
    {
        nIgnore = std::stoi(lineSplitted[7]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for the nIgnore of " + kernelName);
    }

    m_Engine->addGPUMatrixKernel(kernelClassName, kernelName, gridSizes, blockSizes, sharedMemorySizes, kernelClassParameters, nRun, nIgnore);
}
#endif

#ifdef CUDA_ENABLED
void ConfigFileReader::readGPUTensorKernel(const std::string &line)
{
    std::vector<std::string> lineSplitted = split(line, '|');

    std::string kernelClassName = lineSplitted[0];
    std::string kernelName = lineSplitted[1];
    std::vector<std::string> gridSizesString = split(lineSplitted[2], '/');
    std::vector<std::string> blockSizesString = split(lineSplitted[3], '/');
    std::vector<std::string> sharedMemorySizesString = split(lineSplitted[4], '/');

    if (gridSizesString.size() != blockSizesString.size() && blockSizesString.size() != sharedMemorySizesString.size())
    {
        throw std::runtime_error("Length of the gridSizes, blockSizes, and sharedMemorySizes does not match for " + kernelName);
    }

    std::vector<int> gridSizes(gridSizesString.size());
    std::vector<int> blockSizes(gridSizesString.size());
    std::vector<int> sharedMemorySizes(gridSizesString.size());
    for (int i = 0; i != gridSizesString.size(); ++i)
    {
        try
        {
            int gridSize = std::stoi(gridSizesString[i]);
            gridSizes[i] = gridSize;
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for the gridSizes of " + kernelName);
        }

        try
        {
            int blockSize = std::stoi(blockSizesString[i]);
            blockSizes[i] = blockSize;
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for the blockSizes of " + kernelName);
        }

        try
        {
            int sharedMemorySize = std::stoi(sharedMemorySizesString[i]);
            sharedMemorySizes[i] = sharedMemorySize;
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid format for the sharedMemorySizes of " + kernelName);
        }
    }

    std::string kernelClassParameters = lineSplitted[5];

    int nRun, nIgnore;
    try
    {
        nRun = std::stoi(lineSplitted[6]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for the nRun of " + kernelName);
    }

    try
    {
        nIgnore = std::stoi(lineSplitted[7]);
    }
    catch (const std::invalid_argument &e)
    {
        throw std::runtime_error("Invalid format for the nIgnore of " + kernelName);
    }

    m_Engine->addGPUTensorKernel(kernelClassName, kernelName, gridSizes, blockSizes, sharedMemorySizes, kernelClassParameters, nRun, nIgnore);
}
#endif

void ConfigFileReader::validateConfigStatus()
{
    if (PROJECT_DIR == "-1")
    {
        throw std::runtime_error("YOU MUST SET PROJECT_DIR SETTING EXPLICITLY IN THE CONFIG FILE!");
    }

    if (ZOO_TYPE == "-1")
    {
        throw std::runtime_error("YOU MUST SET ZOO_TYPE SETTING EXPLICITLY IN THE CONFIG FILE! -> AVAILABLE ZOO TYPES: MAT, MATORD, TENS, TENSORD, FULLTENSOR");
    }
    if (ZOO_TYPE != "MAT" && ZOO_TYPE != "MATORD" && ZOO_TYPE != "TENS" && ZOO_TYPE != "TENSORD" && ZOO_TYPE != "FULLTENSOR")
    {
        throw std::runtime_error("YOU HAVE INDICATED INCORRECT ZOO_TYPE IN THE CONFIG FILE! -> AVAILABLE ZOO TYPES: MAT, MATORD, TENS, TENSORD, FULLTENSOR");
    }

    if (ZOO_TYPE == "MAT" || ZOO_TYPE == "MATORD")
    {
        if (MATRIX_FILES_DIR == "-1" || MATRIX_ORDERING_FILES_DIR == "-1" || MATRIX_VISUALIZATION_FILES_DIR == "-1")
        {
            throw std::runtime_error("MAT AND MATORD ZOO TYPES REQUIRE YOU TO INDICATE FOLLOWING SETTINGS EXPLICITLY: MATRIX_FILES_DIR, MATRIX_ORDERING_FILES_DIR, MATRIX_VISUALIZATION_FILES_DIR");
        }
    }
    else if (ZOO_TYPE == "TENS" || ZOO_TYPE == "TENSORD" || ZOO_TYPE == "FULLTENSOR")
    {
        if (TENSOR_FILES_DIR == "-1" || TENSOR_ORDERING_FILES_DIR == "-1" || TENSOR_VISUALIZATION_FILES_DIR == "-1")
        {
            throw std::runtime_error("TENS, TENSORD, AND FULLTENSOR ZOO TYPES REQUIRE YOU TO INDICATE FOLLOWING SETTINGS EXPLICITLY: TENSOR_FILES_DIR, TENSOR_ORDERING_FILES_DIR, TENSOR_VISUALIZATION_FILES_DIR");
        }
    }

    if (CHART_TYPE == "-1")
    {
        throw std::runtime_error("YOU MUST SET CHART_TYPE SETTING EXPLICITLY IN THE CONFIG FILE! -> AVAILABLE CHART TYPES: NNZ, ABS");
    }
    if (CHART_TYPE != "ABS" && CHART_TYPE != "NNZ")
    {
        throw std::runtime_error("YOU HAVE INDICATED INCORRECT CHART_TYPE IN THE CONFIG FILE! -> AVAILABLE CHART TYPES: NNZ, ABS");
    }
}
