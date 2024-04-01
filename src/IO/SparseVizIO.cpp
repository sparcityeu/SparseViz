#include "SparseVizIO.h"
#include "config.h"
#include "helpers.h"
#include "SparseMatrix.h"
#include "omp.h"
#include "SparseTensor.h"
#include <sstream>
#include "MatrixMarketIOLibrary.h"
#include "SparseTensorCOO.h"
#include "SparseTensorCSF.h"
#include "SparseTensorHICOO.h"
#include "pigo.hpp"
#include "sort.h"


SparseMatrix* SparseVizIO::readMatrixFromMarketFile(const std::string &marketFileName)
{
    // Checking whether it has already been read and saved to a binary file previously
    std::string binaryFileName = marketFileName + ".bin";
    std::string matrixName = split(split(binaryFileName, '.').front(), '/').back();

    SparseMatrix* matrix = SparseVizIO::readMatrixFromBinaryFile(binaryFileName, matrixName);
    if (matrix != nullptr)
    {
        return matrix;
    }

    double start_time = omp_get_wtime();
    int row, column;

    MM_typecode matcode;
    FILE* f = fopen(marketFileName.c_str(), "r");

    if (f == nullptr)
    {
#pragma omp critical
        {
            throw std::runtime_error("Failed to open the file: " + marketFileName);
        }
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        fclose(f);
#pragma omp critical
        {
            throw std::runtime_error(marketFileName + ": Could not process the Matrix Market banner.");
        }
    }

    if (!(mm_is_valid(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)))
    {
        fclose(f);
#pragma omp critical
        {
            throw std::runtime_error(marketFileName + ": Unsupported type: [" + mm_typecode_to_str(matcode) + ']');
        }
    }

    bool isSymmetric = mm_is_symmetric(matcode);

    int nnz;
    if (mm_read_mtx_crd_size(f, &row, &column, &nnz) != 0)
    {
        fclose(f);
        throw std::runtime_error(marketFileName + ": Could not read coordinate size.");
    }
    vType* storage = new vType[nnz * 2];
    valType* values = new valType[nnz];

    int r, c;
    double val;
    char line[256];

    fgets(line, sizeof(line), f);
    int itemsRead = sscanf(line, "%d %d %lg", &r, &c, &val);
    storage[0] = --r; storage[1] = --c;
    if (itemsRead == 2)
    {
        memset(values, 1, sizeof(valType) * nnz);

        for (int i = 1; i != nnz; ++i)
        {
            fgets(line, sizeof(line), f);
            sscanf(line, "%d %d", &r, &c);

            --r; // converting 1-based indexing to 0-based indexing
            --c;
            eType nnzStart = i * 2;

            storage[nnzStart] = r;
            storage[nnzStart + 1] = c;

            if (isSymmetric && (r != c))
            {
                nnzStart = (i + 1) * 2;
                storage[nnzStart] = c;
                storage[nnzStart + 1] = r;
            }
        }
    }
    else
    {
        values[0] = val;

        for (int i = 1; i != nnz; ++i)
        {
            fgets(line, sizeof(line), f);
            sscanf(line, "%d %d %lg", &r, &c, &val);

            --r; // converting 1-based indexing to 0-based indexing
            --c;
            eType nnzStart = i * 2;

            storage[nnzStart] = r;
            storage[nnzStart + 1] = c;
            values[i] = val;

            if (isSymmetric && (r != c))
            {
                nnzStart = (i + 1) * 2;
                storage[nnzStart] = c;
                storage[nnzStart + 1] = r;
                values[i + 1] = val;
            }
        }
    }

    fclose(f);
    double end_time = omp_get_wtime();

    // Order By row, column ASC
    vType dims[2]; dims[0] = row; dims[1] = column;
    sortNonzeros(2, dims, nnz, storage, values);
    matrix = new SparseMatrix(matrixName, row, column, nnz, storage, values);
    delete[] storage;

    logger.logReadingMatrixMarket(matrix, end_time - start_time);

    // Writing the matrix into a binary file to accelerate the speed at which the matrix will be read next time
    SparseVizIO::writeMatrixToBinaryFile(binaryFileName, matrix);

    return matrix;
}

void SparseVizIO::writeMatrixToBinaryFile(const std::string &binaryFileName, SparseMatrix* matrix)
{
    FILE *fp = nullptr;
    if ((fp = fopen(binaryFileName.c_str(), "w")) == nullptr)
    {
        std::cout << binaryFileName << " cannot be created" << endl;
    }

    double start_time = omp_get_wtime();

    vType& rowCount = matrix->getRowCount();
    vType& colCount = matrix->getColCount();
    vType& nnzCount = matrix->getNNZCount();
    bool& isPatternSymmetric = matrix->isPatternSymmetric();
    bool& isSymmetric = matrix->isSymmetric();

    fwrite (&rowCount, sizeof(vType), 1, fp);
    fwrite (&colCount, sizeof(vType), 1, fp);
    fwrite (&nnzCount, sizeof(vType), 1, fp);
    fwrite (&isPatternSymmetric, sizeof(bool), 1, fp);
    fwrite (&isSymmetric, sizeof(bool), 1, fp);

    vType* ptr = matrix->getPtr();
    fwrite (ptr, sizeof(vType), (rowCount + 1), fp);

    vType* ind = matrix->getInd();
    fwrite (ind, sizeof(vType), nnzCount, fp);

    valType* val = matrix->getValues();
    fwrite (val, sizeof(valType), nnzCount, fp);

    fclose(fp);

    double end_time = omp_get_wtime();
    logger.logWritingMatrixBinary(matrix, end_time - start_time);
}

SparseMatrix* SparseVizIO::readMatrixFromBinaryFile(const std::string &binaryFileName, const std::string& name)
{
    /*
     * Reads matrix from a binary file, faster than MatrixConstructor::constructMatrixFromMarketFile but should only be called
     * when the market file has been read at least one time in the past by MatrixConstructor::constructMatrixFromMarketFile
     */

    FILE *fp = nullptr;
    if ((fp = fopen(binaryFileName.c_str(), "r")) == nullptr)
    {
        return nullptr;
    }

    double start_time = omp_get_wtime();

    vType row, column, nnzCount;
    bool isPatternSymmetric, isSymmetric;

    fread (&row, sizeof(vType), 1, fp);
    fread (&column, sizeof(vType), 1, fp);
    fread (&nnzCount, sizeof(vType), 1, fp);
    fread (&isPatternSymmetric, sizeof(bool), 1, fp);
    fread (&isSymmetric, sizeof(bool), 1, fp);

    SparseMatrix* matrix = new SparseMatrix(name, row, column, nnzCount, isSymmetric, isPatternSymmetric);

    vType* ptr = matrix->getPtr();
    fread (ptr, sizeof(vType), (row + 1), fp);

    vType* ind = matrix->getInd();
    fread (ind, sizeof(vType), nnzCount, fp);

    valType* val = matrix->getValues();
    fread (val, sizeof(valType), nnzCount, fp);
    fclose(fp);
    double end_time = omp_get_wtime();
    logger.logReadingMatrixBinary(matrix, end_time - start_time);
    return matrix;
}

SparseMatrix *
SparseVizIO::readOrderedMatrixFromBinaryFile(const std::string &orderingName, const std::string &matrixName)
{
    std::string name = orderingName + "_ordered_" + matrixName;
    std::string binaryFileName = name + ".mtx.bin";
    SparseMatrix* matrix = SparseVizIO::readMatrixFromBinaryFile(MATRIX_FILES_DIR + binaryFileName, name);
    return matrix;
}

void SparseVizIO::writeOrderedMatrixToBinaryFile(SparseMatrix *matrix)
{
    std::string binaryFileName = matrix->getName() + ".mtx.bin";
    SparseVizIO::writeMatrixToBinaryFile(MATRIX_FILES_DIR + binaryFileName, matrix);
}

bool SparseVizIO::readMatrixOrderingFromBinaryFile(const std::string& binaryFileName, vType **rowIPermutation, vType **colIPermutation, vType nRow, vType nCol)
{
    FILE *fp = nullptr;
    if ((fp = fopen(binaryFileName.c_str(), "r")) != nullptr)
    {
        *rowIPermutation = new vType[nRow];
        *colIPermutation = new vType[nCol];

        fread (*rowIPermutation, sizeof(vType), nRow, fp);
        fread (*colIPermutation, sizeof(vType), nRow, fp);

        fclose(fp);

        return true;
    }
    return false;
}

void SparseVizIO::writeMatrixOrderingToBinaryFile(const std::string &binaryFileName, vType *rowIPermutation,
                                                  vType *colIPermutation, vType nRow, vType nCol)
{
    FILE *fp = nullptr;
    if ((fp = fopen(binaryFileName.c_str(), "w")) == nullptr)
    {
        throw std::runtime_error(binaryFileName + ": Cannot open the output file!");
    }

    fwrite (rowIPermutation, sizeof(vType), nRow, fp);
    fwrite (colIPermutation, sizeof(vType), nCol, fp);

    fclose(fp);
}

SparseTensor* SparseVizIO::readTensorFromMarketFile(const std::string &marketFileName)
{
    std::string binaryFileName = split(marketFileName, '.').front() + ".tns.bin";
    std::string tensorName = split(split(binaryFileName, '.').front(), '/').back();

    SparseTensor* tensor = SparseVizIO::readTensorFromBinaryFile(binaryFileName, tensorName);
    if (tensor != nullptr)
    {
        return tensor;
    }

    double start_time = omp_get_wtime();

    // Currently construction of all tensor formats depend on the COO format, which itself depends on the pigo tensor.
    auto pigoTensor = new pigo::Tensor<vType, eType, vType*, valType, valType*, true>(marketFileName, pigo::FileType::EDGE_LIST);
    SparseTensorCOO* coo = new SparseTensorCOO(tensorName, pigoTensor);
    for(eType i = 0; i < coo->getNNZ(); i++)
    {
        for(int o = 0; o < tensor->getOrder(); o++)
        {
            coo->getStorage()[i * tensor->getOrder() + o]--;
        }
    }

    if (TENSOR_STORAGE_TYPE == COO)
    {
        tensor = coo;
    }
    else if (TENSOR_STORAGE_TYPE == CSF)
    {
        sortNonzeros(coo->getOrder(), coo->getDims(), coo->getNNZ(), coo->getStorage(), coo->getValues());
        tensor = SparseTensorCSF::constructCSFFromCOO("csf_" + tensorName, coo);
    }
    else if (TENSOR_STORAGE_TYPE == HiCOO)
    {
        tensor = SparseTensorHICOO::constructHICOOFromCOO("hicoo_" + tensorName, BLOCK_SIZE, coo, SB_BITS);
    }

    double end_time = omp_get_wtime();

    logger.logReadingTensorMarket(tensor, end_time - start_time);

    SparseVizIO::writeTensorToBinaryFile(binaryFileName, tensor);

    return tensor;
}

void SparseVizIO::writeTensorToBinaryFile(const std::string &binaryFileName, SparseTensor *tensor)
{
    double start_time = omp_get_wtime();

    try
    {
        tensor->save(binaryFileName);
    }
    catch (const pigo::Error&)
    {
        std::cout << binaryFileName << " cannot be created" << std::endl;
        return;
    }
    catch (const std::runtime_error&)
    {
        return;
    }

    double end_time = omp_get_wtime();
    logger.logWritingTensorBinary(tensor, end_time - start_time);
}

SparseTensor *SparseVizIO::readTensorFromBinaryFile(const std::string &binaryFileName, const std::string& name)
{
    try
    {
        double start_time = omp_get_wtime();

        // Currently construction of all tensor formats depend on the COO format, which itself depends on the pigo tensor.
        auto pigoTensor = new pigo::Tensor<vType, eType, vType*, valType, valType*, true>(binaryFileName, pigo::FileType::PIGO_TENSOR_BIN);
        SparseTensorCOO* coo = new SparseTensorCOO(name, pigoTensor);

        SparseTensor* tensor = nullptr;
        if (TENSOR_STORAGE_TYPE == COO)
        {
            tensor = coo;
        }
        else if (TENSOR_STORAGE_TYPE == CSF)
        {
            tensor = SparseTensorCSF::constructCSFFromCOO("csf_" + name, coo);
        }
        else if (TENSOR_STORAGE_TYPE == HiCOO)
        {
            tensor = SparseTensorHICOO::constructHICOOFromCOO("hicoo_" + name, BLOCK_SIZE, coo, SB_BITS);
        }

        double end_time = omp_get_wtime();

        logger.logReadingTensorBinary(tensor, end_time - start_time);

        return tensor;
    }
    catch (const pigo::Error& e)
    {
        return nullptr;
    }
}

SparseTensor* SparseVizIO::readOrderedTensorFromBinaryFile(const std::string &orderingName, const std::string &tensorName)
{
    std::string name = orderingName + "_ordered_" + tensorName;
    std::string binaryFileName = name + ".tns.bin";
    SparseTensor* tensor = SparseVizIO::readTensorFromBinaryFile(TENSOR_FILES_DIR + binaryFileName, name);
    return tensor;
}

void SparseVizIO::writeOrderedTensorToBinaryFile(SparseTensor *tensor)
{
    std::string binaryFileName = tensor->getName() + ".tns.bin";
    SparseVizIO::writeTensorToBinaryFile(TENSOR_FILES_DIR + binaryFileName, tensor);
}

bool SparseVizIO::readTensorOrderingFromBinaryFile(const std::string &binaryFileName, vType norder, vType* dims, vType**& orderedDimensions)
{
    try
    {
        pigo::File binaryFile(binaryFileName, pigo::OpenMode::READ);
        orderedDimensions = new vType*[norder];
        for (vType i = 0; i != norder; ++i)
        {   
            orderedDimensions[i] = new vType[dims[i]];
            size_t sizeToRead = sizeof(vType) * dims[i];
            binaryFile.parallel_read(reinterpret_cast<char*>(orderedDimensions[i]), sizeToRead);
        }
        return true;
    }
    catch (const pigo::Error& e)
    {
        return false;
    }
}

void SparseVizIO::writeTensorOrderingToBinaryFile(const std::string &binaryFileName, vType norder, vType* dims, vType** orderedDimensions)
{
    try
    {
        size_t totalSize = 0;
        for (vType i = 0; i != norder; ++i)
        {
            size_t sizeToWrite = sizeof(vType) * dims[i];
            totalSize += sizeToWrite;
        }
        pigo::File binaryFile(binaryFileName, pigo::OpenMode::WRITE, totalSize);
        for (vType i = 0; i != norder; ++i)
        {
            size_t sizeToWrite = sizeof(vType) * dims[i];
            binaryFile.parallel_write(reinterpret_cast<char*>(orderedDimensions[i]), sizeToWrite);
        }
    }
    catch (const pigo::Error& e)
    {
        std::cout << "Skipping writing tensor ordering to a binary file: " << e.what() << std::endl;
    }
}
