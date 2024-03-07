
#ifndef SPARSEVIZ_SPARSEVIZTEST_H
#define SPARSEVIZ_SPARSEVIZTEST_H

#include <string>
#include <vector>
#include <ostream>
#include <istream>
#include <random>
#include <sstream>


class ConfigFileGenerator
{
public:
    static void generateConfigFile(const std::string& path);

private:
    static std::string getRandomZooType();
    static std::string getRandomChartType();
    static std::string getRandomBooleanValue();
    static std::string getRandomMatrixKernel();
    static std::string getRandomGPUMatrixKernel();
    static std::string getRandomTensorKernel();
    static std::string getRandomGPUTensorKernel();

    // helpers
    static std::string getRandomArrayElement(const std::vector<std::string>& arr);
    static std::string getRandomArrayElements(const std::vector<std::string>& arr, int no = -1);
    static int getRandomNumber(int start, int finish);
};


class SparseVizTest
{
public:
    static SparseVizTest* getSparseVizTester();

    SparseVizTest(const SparseVizTest& other) = delete;
    SparseVizTest(SparseVizTest&& other) = delete;

    void runTests(unsigned totalTestNo);

    void logTest(const std::string& log);
    std::string getCurrentDirectory() const {return m_CurrentDirectory;}

private:
    explicit SparseVizTest(): m_CurrentLogFile(nullptr), m_CurrentTestNo(0) {}
    ~SparseVizTest() {}

    void initializeTestDirectory();

    void allocateTestLogFile();
    void deallocateTestLogFile();

    std::string generateConfigFile();

private:
    static SparseVizTest* m_InstancePointer;
    std::string m_CurrentDirectory;
    std::ofstream* m_CurrentLogFile;
    unsigned m_CurrentTestNo;
};


#endif //SPARSEVIZ_SPARSEVIZTEST_H
