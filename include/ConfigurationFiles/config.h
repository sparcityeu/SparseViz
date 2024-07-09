#ifndef SPARSEVIZ_CONFIG_H
#define SPARSEVIZ_CONFIG_H

#include "string"
#include <fstream>
#include <ostream>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include "omp.h"
#include "SparseVizLogger.h"
#include <set>
#include <cstdint>


typedef unsigned int vType; //!< Vertex data type to be used throughout the program.
typedef unsigned int eType; //!< Edge data type to be used throughout the program.
typedef double valType; //!< Value/Weight data type to be used throughout the program.
typedef unsigned int BlockType; //!< Block index data type to be used in HiCOO tensor representation.
typedef u_int8_t OffsetType; //!< Offset index data type to be used in HiCOO tensor representation.
typedef unsigned __int128 MortonIndex; //!< Morton index data type to be used during morton sorting.

class SparseMatrix;
class SparseTensor;
class SparseVizEngine;

extern SparseVizLogger logger;    //!< Logger that is used for logging purposes especially when the operation needs to be logged onto the .csv file.
extern bool TIMING_LOG;    //!< Indicates whether or not timing should be logged into terminal | Default = true.
extern std::string PROJECT_DIR;    //!< The path of the root project directory.
extern double MAX_TIME_BEFORE_ABORTING_ORDERING;    //!< Maximum time dedicated to orderings before they are forcefully aborted | Default = 1000.
extern bool EXPORT_ORDERED_SPARSE_STRUCTURES;    //!< Indicates whether or not sparse structures that have been ordered should be written to binary files | Default = true.
extern bool USE_EXISTING_ORDERED_SPARSE_STRUCTURES;    //!< Indicates whether or not sparse structures that had been ordered and that have been written to binary files are allowed to be read | Default = true.
extern bool EXPORT_ORDERINGS;    //!< Indicates whether or not orderings made should be written to binary files | Default = true.
extern bool USE_EXISTING_ORDERINGS; //!< Indicates whether or not ordering that have been made previously are allowed to be read | Default = true.
extern std::string MATRIX_FILES_DIR;    //!< The path to the directory under which .mtx and .mtx.bin files are located.
extern std::string MATRIX_ORDERING_FILES_DIR;    //!< The path to the directory under which .bin matrix ordering files are located.
extern std::string MATRIX_VISUALIZATION_FILES_DIR;    //!< The path to the directory to which .html matrix visualization files are going to be generated.
extern std::string TENSOR_FILES_DIR;    //!< The path to the directory under which .tns and .tns.bin files are located.
extern std::string TENSOR_ORDERING_FILES_DIR;    //!< The path to the directory under which .bin tensor ordering files are located.
extern std::string TENSOR_VISUALIZATION_FILES_DIR;    //!< The path to the directory to which .html tensor visualization files are going to be generated.
extern std::string LOGO_PATH;    //!< Logo path of the html.
extern std::string FAVICON_PATH;    //!< Favicon path of the html.
extern std::string ZOO_TYPE;    //!< Determines the active ZOO_TYPE.
extern std::string CHART_TYPE; //!< Determines the chart_type visualization files are generated with.
extern unsigned int MAX_DIM;    //!< Maximum dimension that can be seen in the visualization files | Default = 64.
extern std::string TEST_CONFIG;
extern std::string TEST_DIRECTORY;
// #define TEST

enum TensorType
{
    COO,
    CSF,
    HiCOO
};
extern TensorType TENSOR_STORAGE_TYPE;
extern BlockType BLOCK_SIZE;

/*!
 * @brief ConfigFileReader is a class that is responsible for reading, parsing, and processing the config file and initializing the SparseVizEngine based on it.
 */
class ConfigFileReader
{
public:
    /*!
     * @brief Initializes the ConfigFileReader.
     * @param configFile Path of the config file from which the reader is to be initialized.
     */
    ConfigFileReader(const std::string& configFile);

    /*!
     * @brief Copy Constructor is unavailable.
     */
    ConfigFileReader(const ConfigFileReader& other) = delete;

    /*!
     * @brief Move Constructor is unavailable.
     */
    ConfigFileReader(ConfigFileReader&& other) = delete;

    /*!
     * @brief Copy Operator is unavailable.
     */
    ConfigFileReader& operator=(const ConfigFileReader& other) = delete;
    /*!
     * @brief Move Operator is unavailable.
     */
    ConfigFileReader& operator=(ConfigFileReader&& other) = delete;

    /*!
     * @brief Destructor for cleaning up the memory space.
     */
    ~ConfigFileReader();

    /*!
     * @brief Instantiates SparseVizEngine object to conduct the operations requested in the config file.
     */
    SparseVizEngine* instantiateEngine();

private:
    /*!
     * @brief Controls whether there is any error made in the config file.
     * @throws std::runtime_error if it is concluded that there are mistakes made in the config file.
     * @details Error could be done in the config file in many ways, such as:
     * - Not indicating or making typos in mandatory settings of the config file, such as PROJECT_DIR or ZOO_TYPE.
     * - Indicating the ZOO_TYPE as MAT but not providing additional settings that are required to be indicated with MAT ZOO_TYPE, such as:
     *   MATRIX_FILES_DIR, MATRIX_ORDERINGS_FILES_DIR, MATRIX_VISUALIZATION_FILES_DIR.
     */
    void validateConfigStatus();

    /*!
     * @brief Reads config file.
     */
    void readConfigFile();

    /*!
     * @brief Reads single setting indicated in the config file under the section *SETTINGS*.
     * @param line The setting line to be read.
     */
    void readSetting(const std::string& line);

    /*!
     * @brief Reads single tensor indicated in the config file under the section *TENSORS*.
     * @param line The tensor line to be read.
     */
    void readTensor(const std::string& line);

    /*!
     * @brief Reads single matrix ordering indicated in the config file under the section *MATRIX_ORDERINGS*.
     * @param line The matrix ordering line to be read.
     */
    void readMatrixOrdering(const std::string& line);

    /*!
     * @brief Reads single tensor ordering indicated in the config file under the section *TENSOR_ORDERINGS*.
     * @param line The tensor ordering line to be read.
     */
    void readTensorOrdering(const std::string& line);

    /*!
     * @brief Reads single matrix kernel indicated in the config file under the section *MATRIX_KERNELS*.
     * @param line The matrix kernel line to be read.
     */
    void readMatrixKernel(const std::string& line);

    /*!
     * @brief Reads single tensor kernel indicated in the config file under the section *TENSOR_KERNELS*.
     * @param line The tensor kernel line to be read.
     */
    void readTensorKernel(const std::string& line);

#ifdef CUDA_ENABLED
    /*!
     * @brief Reads single matrix gpu kernel indicated in the config file under the section *GPU_MATRIX_KERNELS*.
     * @param line The matrix gpu kernel line to be read.
     * @warning It is compiled only if CUDA and CUDA compatible device are present in the user machine.
     */
    void readGPUMatrixKernel(const std::string& line);

    /*!
     * @brief Reads single tensor gpu kernel indicated in the config file under the section *GPU_TENSOR_KERNELS*.
     * @param line The tensor gpu kernel line to be read.
     * @warning It is compiled only if CUDA and CUDA compatible device are present in the user machine.
     */
    void readGPUTensorKernel(const std::string& line);
#endif

    /*!
     * @brief Helper function to enable the reader to skip the sections of no importance.
     * @param line Section to be checked if it requires skipping.
     * @return returns false if it requires skipping, true otherwise.
     * @details The way it is utilized is as following:
     * Assume that the ZOO_TYPE of the config file is MAT, then there is no need to read any section related to tensors including:
     * *TENSORS*, *TENSOR_ORDERINGS*, *TENSOR_KERNELS*, *GPU_TENSOR_KERNELS*.
     */
    bool checkZooTypeMatch(const std::string& section);

private:
    std::string m_LogFilePath;

    std::vector<std::string> m_Matrices;
    std::vector<std::string> m_MatrixOrderings;
    std::vector<std::string> m_MatrixKernels;
    std::vector<std::string> m_GPUMatrixKernels;

    std::vector<std::string> m_Tensors;
    std::vector<std::string> m_TensorOrderings;
    std::vector<std::string> m_TensorKernels;
    std::vector<std::string> m_GPUTensorKernels;

    std::ifstream m_File;
    SparseVizEngine* m_Engine;
};


#endif //SPARSEVIZ_CONFIG_H
