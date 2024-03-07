#include "SparseVizTest.h"
#include "config.h"
#include "helpers.h"
#include "SparseVizEngine.h"


namespace Constants
{
    std::vector<std::string> zooTypeArray =
            {
                    "MAT",
                    "MATORD",
                    "TENS",
                    "TENSORD",
                    "FULLTENSOR"
            };
    std::vector<std::string> chartTypeArray =
            {
                    "NNZ",
                    "ABS"
            };
    std::vector<std::string> matrixKernelArray =
            {
                    "Dijkstra",
                    "Bellman_Ford",
                    "Floyd_Warshall",
                    "BFS",
                    "SPMVRowBased",
                    "SequentialBFS",
                    "SPMMRowBased"
            };
    std::vector<std::string> tensorKernelArray =
            {
                    "COOTensorKernel"
            };
    std::vector<std::string> matrixKernelGPUArray =
            {
                    "SPMVRowBased"
            };
    std::vector<std::string> tensorKernelGPUArray =
            {
                    "COOTensorGPUKernelAoS",
                    "COOTensorGPUKernelSoA"
            };
    std::vector<std::string> threadCountArray =
            {
                    std::to_string('1'),
                    std::to_string('2'),
                    std::to_string('4'),
                    std::to_string('8'),
                    "16"
            };
    std::vector<std::string> schedulingPolicyArray =
            {
                    "static",
                    "auto",
                    "dynamic",
                    "guided"
            };
    std::pair<int, int> chunkSizeWidthArray(1, 256);
    std::pair<int, int> nRunWidth(1, 10);
    std::pair<int, int> nIgnoreWidth(0, 10);
    std::vector<std::string> gridSizeArray =
            {
                    "128",
                    "256",
                    "512",
                    "1024"
            };
    std::vector<std::string> blockSizeArray =
            {
                    "8",
                    "16",
                    "32",
                    "64"
            };
    std::pair<int, int> kernelGenerationWidthArray(0, 4);
}

void ConfigFileGenerator::generateConfigFile(const std::string &path)
{
    std::ifstream readStream(TEST_CONFIG);

    std::string zooType;

    std::string file;
    std::string line;
    while (std::getline(readStream, line))
    {
        if (line.back() == '=')
        {
            if (
                    line == "EXPORT_ORDERED_SPARSE_STRUCTURES =" ||
                    line == "USE_EXISTING_ORDERED_SPARSE_STRUCTURES =" ||
                    line == "EXPORT_ORDERINGS =" ||
                    line == "USE_EXISTING_ORDERINGS =" )
            {
                line += ' ' + ConfigFileGenerator::getRandomBooleanValue();
            }
            else if (line == "ZOO_TYPE =")
            {
                zooType = ConfigFileGenerator::getRandomZooType();
                line += ' ' + zooType;
            }
            else if (line == "CHART_TYPE =")
            {
                line += ' ' + ConfigFileGenerator::getRandomChartType();
            }
            file += line;
        }
        else
        {
            if (line == "*MATRIX_KERNELS*" || line == "*GPU_MATRIX_KERNELS*" || line == "*TENSOR_KERNEL*" || line == "*GPU_TENSOR_KERNELS*")
            {
                file += line + '\n';
                int randomNumber = ConfigFileGenerator::getRandomNumber(Constants::kernelGenerationWidthArray.first, Constants::kernelGenerationWidthArray.second);
                if (line == "*MATRIX_KERNELS*" && (zooType == "MAT" || zooType == "MATORD"))
                {
                    for (int i = 0; i < randomNumber - 1; ++i)
                    {
                        file += ConfigFileGenerator::getRandomMatrixKernel() + '\n';
                    }
                    if (randomNumber > 0)
                    {
                        file += ConfigFileGenerator::getRandomMatrixKernel();
                    }
                }
                else if (line == "*GPU_MATRIX_KERNELS*" && (zooType == "MAT" || zooType == "MATORD"))
                {
                    for (int i = 0; i < randomNumber - 1; ++i)
                    {
                        file += ConfigFileGenerator::getRandomGPUMatrixKernel() + '\n';
                    }
                    if (randomNumber > 0)
                    {
                        file += ConfigFileGenerator::getRandomGPUMatrixKernel();
                    }
                }
                else if (line == "*TENSOR_KERNEL*" && (zooType == "TENS" || zooType == "TENSORD" || zooType == "FULLTENSORD"))
                {
                    for (int i = 0; i < randomNumber - 1; ++i)
                    {
                        file += ConfigFileGenerator::getRandomTensorKernel() + '\n';
                    }
                    if (randomNumber > 0)
                    {
                        file += ConfigFileGenerator::getRandomTensorKernel();
                    }
                }
                else if (line == "*GPU_TENSOR_KERNELS*" && (zooType == "TENS" || zooType == "TENSORD" || zooType == "FULLTENSORD"))
                {
                    for (int i = 0; i < randomNumber - 1; ++i)
                    {
                        file += ConfigFileGenerator::getRandomGPUTensorKernel() + '\n';
                    }
                    if (randomNumber > 0)
                    {
                        file += ConfigFileGenerator::getRandomGPUTensorKernel();
                    }
                }
            }
            else
            {
                file += line;
            }
        }

        file += '\n';
    }

    readStream.close();

    std::ofstream writeStream(path);

    writeStream << file;
    writeStream.close();
}

int ConfigFileGenerator::getRandomNumber(int start, int finish)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribution(start, finish);
    return distribution(gen);
}

std::string ConfigFileGenerator::getRandomArrayElement(const std::vector<std::string> &arr)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribution(0, arr.size() - 1);
    int randomNum = distribution(gen);

    return arr[randomNum];
}

std::string ConfigFileGenerator::getRandomArrayElements(const std::vector<std::string> &arr, int no)
{
    std::string ret;

    if (no == -1)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distribution(1, arr.size());
        int upperBound = distribution(gen);
        for (int i = 0; i != upperBound; ++i)
        {
            ret += '/' + ConfigFileGenerator::getRandomArrayElement(arr);
        }
    }
    else
    {
        for (int i = 0; i != no; ++i)
        {
            ret += '/' + ConfigFileGenerator::getRandomArrayElement(arr);
        }
    }

    ret = ret.substr(1);

    return ret;
}

std::string ConfigFileGenerator::getRandomZooType()
{
    return getRandomArrayElement(Constants::zooTypeArray);
}

std::string ConfigFileGenerator::getRandomChartType()
{
    return getRandomArrayElement(Constants::chartTypeArray);
}

std::string ConfigFileGenerator::getRandomBooleanValue()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribution(0, 1);
    int randomNum = distribution(gen);

    if (randomNum == 0)
    {
        return "TRUE";
    }
    else
    {
        return "FALSE";
    }
}

std::string ConfigFileGenerator::getRandomMatrixKernel()
{
    if (!Constants::matrixKernelArray.empty())
    {
        std::string line = getRandomArrayElement(Constants::matrixKernelArray) + " | ";
        line += getRandomArrayElements(Constants::threadCountArray) + " | ";
        line += getRandomArrayElement(Constants::schedulingPolicyArray) + " | ";
        line += std::to_string(getRandomNumber(Constants::chunkSizeWidthArray.first, Constants::chunkSizeWidthArray.second)) + " | ";

        int run = getRandomNumber(Constants::nRunWidth.first, Constants::nRunWidth.second);
        line += std::to_string(run) + " | ";
        line += std::to_string(getRandomNumber(Constants::nIgnoreWidth.first, run ? run < Constants::nIgnoreWidth.second : Constants::nIgnoreWidth.second));

        return line;
    }

    return std::string();
}

std::string ConfigFileGenerator::getRandomGPUMatrixKernel()
{
    if (!Constants::matrixKernelGPUArray.empty())
    {
        std::string line = getRandomArrayElement(Constants::matrixKernelGPUArray) + " | ";
        std::string gridSizeString = getRandomArrayElements(Constants::gridSizeArray);
        line += gridSizeString + " | ";
        line += getRandomArrayElements(Constants::blockSizeArray, split(gridSizeString, '/').size()) + " | ";

        int run = getRandomNumber(Constants::nRunWidth.first, Constants::nRunWidth.second);
        line += std::to_string(run) + " | ";
        line += std::to_string(getRandomNumber(Constants::nIgnoreWidth.first, run ? run < Constants::nIgnoreWidth.second : Constants::nIgnoreWidth.second));

        return line;
    }

    return std::string();
}

std::string ConfigFileGenerator::getRandomTensorKernel()
{
    if (!Constants::tensorKernelArray.empty())
    {
        std::string line = getRandomArrayElement(Constants::tensorKernelArray) + " | ";
        line += getRandomArrayElements(Constants::threadCountArray) + " | ";
        line += getRandomArrayElement(Constants::schedulingPolicyArray) + " | ";
        line += std::to_string(getRandomNumber(Constants::chunkSizeWidthArray.first, Constants::chunkSizeWidthArray.second)) + " | ";

        int run = getRandomNumber(Constants::nRunWidth.first, Constants::nRunWidth.second);
        line += std::to_string(run) + " | ";
        line += std::to_string(getRandomNumber(Constants::nIgnoreWidth.first, run ? run < Constants::nIgnoreWidth.second : Constants::nIgnoreWidth.second));

        return line;
    }
    return std::string();
}

std::string ConfigFileGenerator::getRandomGPUTensorKernel()
{
    if (!Constants::tensorKernelGPUArray.empty())
    {
        std::string line = getRandomArrayElement(Constants::tensorKernelGPUArray) + " | ";
        std::string gridSizeString = getRandomArrayElements(Constants::gridSizeArray);
        line += gridSizeString + " | ";
        line += getRandomArrayElements(Constants::blockSizeArray, split(gridSizeString, '/').size()) + " | ";

        int run = getRandomNumber(Constants::nRunWidth.first, Constants::nRunWidth.second);
        line += std::to_string(run) + " | ";
        line += std::to_string(getRandomNumber(Constants::nIgnoreWidth.first, run ? run < Constants::nIgnoreWidth.second : Constants::nIgnoreWidth.second));

        return line;
    }
    return std::string();
}


SparseVizTest* SparseVizTest::m_InstancePointer = nullptr;


void SparseVizTest::runTests(unsigned int totalTestNo)
{
    for (int i = 0; i != totalTestNo; ++i)
    {
        m_CurrentTestNo = i;
        this->initializeTestDirectory();
        this->allocateTestLogFile();
        std::string testConfig = this->generateConfigFile();

        std::cout << "TEST " << i << " STARTED" << std::endl;
        this->logTest("TEST " + std::to_string(i) + " STARTED");;
        try
        {
            ConfigFileReader reader(testConfig);
            SparseVizEngine* engine = reader.instantiateEngine();
            engine->runEngine();
        }
        catch (const std::exception& e)
        {
            std::string prefix = "AN EXCEPTION HAS BEEN THROWN: ";
            this->logTest(prefix + e.what());
            this->deallocateTestLogFile();
        }
        std::cout << "TEST " << i << " ENDED" << std::endl;
        this->logTest("TEST " + std::to_string(i) + " ENDED");

        this->deallocateTestLogFile();
    }
}

void SparseVizTest::initializeTestDirectory()
{
    m_CurrentDirectory = TEST_DIRECTORY + "TEST" + std::to_string(m_CurrentTestNo) + '/';
    mkdir(m_CurrentDirectory.c_str(), 0777);
}

void SparseVizTest::allocateTestLogFile()
{
    std::string filePath = m_CurrentDirectory + "log.txt";
    m_CurrentLogFile = new std::ofstream(filePath);
}

void SparseVizTest::deallocateTestLogFile()
{
    m_CurrentLogFile->close();
    delete m_CurrentLogFile;
}

std::string SparseVizTest::generateConfigFile()
{
    std::string filePath = m_CurrentDirectory + "config.txt";
    ConfigFileGenerator::generateConfigFile(filePath);

    return filePath;
}

SparseVizTest *SparseVizTest::getSparseVizTester()
{
    if (!m_InstancePointer)
    {
        m_InstancePointer = new SparseVizTest();
    }
    return m_InstancePointer;
}

void SparseVizTest::logTest(const std::string &log)
{
#pragma omp critical
    {
        *m_CurrentLogFile << log << '\n';
    }
}
