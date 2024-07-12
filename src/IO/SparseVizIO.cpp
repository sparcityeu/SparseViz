#include "SparseVizIO.h"
#include "config.h"
#include "helpers.h"
#include "SparseMatrix.h"
#include "omp.h"
#include "SparseTensor.h"
#include <sstream>
#include "MatrixMarketIOLibrary.h"
#include "SparseTensorCOO.h"
#include "sort.h"
#include "stdlib.h"
#include "unistd.h"
#include "sys/mman.h"
#include "sys/stat.h"
#include "fcntl.h"
#include "stdio.h"


SparseMatrix* SparseVizIO::readMatrixFromMarketFile(const std::string &marketFileName)
{
    // Checking whether it has already been read and saved to a binary file previously
    std::string binaryFileName = marketFileName + ".bin";
    std::vector<std::string> marketFileParsed = split(marketFileName, '.');
    if (marketFileParsed.size() <= 1) throw std::runtime_error("Your matrix file name should have a file extension of .mtx");
    std::string matrixName = split(marketFileParsed[marketFileParsed.size() - 2], '/').back();

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
    int itemsRead = sscanf(line, "%d %d %lf", &r, &c, &val);
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

        for (int i = 1; i < nnz; ++i)
        {
            fgets(line, sizeof(line), f);
            sscanf(line, "%d %d %lf", &r, &c, &val);

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

    logger->logReadingMatrixMarket(matrix, end_time - start_time);

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
        return;
    }

    double start_time = omp_get_wtime();

    vType& rowCount = matrix->getRowCount();
    vType& colCount = matrix->getColCount();
    vType& nnzCount = matrix->getNNZCount();
    bool& isPatternSymmetric = matrix->isPatternSymmetric();
    bool& isSymmetric = matrix->isSymmetric();

    fwrite(&rowCount, sizeof(vType), 1, fp);
    fwrite(&colCount, sizeof(vType), 1, fp);
    fwrite(&nnzCount, sizeof(vType), 1, fp);
    fwrite(&isPatternSymmetric, sizeof(bool), 1, fp);
    fwrite(&isSymmetric, sizeof(bool), 1, fp);

    vType* ptr = matrix->getPtr();
    fwrite(ptr, sizeof(vType), (rowCount + 1), fp);

    vType* ind = matrix->getInd();
    fwrite(ind, sizeof(vType), nnzCount, fp);

    valType* val = matrix->getValues();
    fwrite(val, sizeof(valType), nnzCount, fp);

    fclose(fp);

    double end_time = omp_get_wtime();
    logger->logWritingMatrixBinary(matrix, end_time - start_time);
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

    fread(&row, sizeof(vType), 1, fp);
    fread(&column, sizeof(vType), 1, fp);
    fread(&nnzCount, sizeof(vType), 1, fp);
    fread(&isPatternSymmetric, sizeof(bool), 1, fp);
    fread(&isSymmetric, sizeof(bool), 1, fp);

    SparseMatrix* matrix = new SparseMatrix(name, row, column, nnzCount, isSymmetric, isPatternSymmetric);

    vType* ptr = matrix->getPtr();
    fread(ptr, sizeof(vType), (row + 1), fp);

    vType* ind = matrix->getInd();
    fread(ind, sizeof(vType), nnzCount, fp);

    valType* val = matrix->getValues();
    fread(val, sizeof(valType), nnzCount, fp);
    fclose(fp);
    double end_time = omp_get_wtime();
    logger->logReadingMatrixBinary(matrix, end_time - start_time);
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

        fread(*rowIPermutation, sizeof(vType), nRow, fp);
        fread(*colIPermutation, sizeof(vType), nRow, fp);

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

    fwrite(rowIPermutation, sizeof(vType), nRow, fp);
    fwrite(colIPermutation, sizeof(vType), nCol, fp);

    fclose(fp);
}

SparseTensor* SparseVizIO::readTensorFromMarketFile(const std::string &marketFileName)
{
    std::string binaryFileName = marketFileName + ".bin";
    std::vector<std::string> marketFileParsed = split(marketFileName, '.');
    if (marketFileParsed.size() <= 1) throw std::runtime_error("Your tensor file name should have a file extension of .tns");
    std::string tensorName = split(marketFileParsed[marketFileParsed.size() - 2], '/').back();

    SparseTensor* tensor = SparseVizIO::readTensorFromBinaryFile(binaryFileName, tensorName);
    if (tensor != nullptr)
    {
        return tensor;
    }

    double start_time = omp_get_wtime();

    int fd = open(marketFileName.c_str(), O_RDONLY);
    struct stat sb;
    if (fd == -1)
    {
        throw std::runtime_error("File " + marketFileName + " could not be opened.");
    }
    if (fstat(fd, &sb) == -1)
    {
        throw std::runtime_error("Error occured trying to get the information related to " + marketFileName + ".");
    }
    char* addr = (char*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED)
    {
        throw std::runtime_error("Mapping of the file: " + marketFileName + " into memory has failed.");
    }

    char* start = addr;
    while ((*start) == '\n')
    {
        ++start;
    }
    char* end = addr + sb.st_size;
    while ((*(end - 1)) == '\n')
    {
        --end;
    }

    std::string firstLine = getLine(start);
    std::vector<std::string> firstLineSplitted = split(firstLine, ' ');

    vType order;
    order = firstLineSplitted.size() - 1;
    vType* dims = new vType[order];
    memset(dims, 0, sizeof(vType) * order);
    eType nnz = 0;

    std::vector<vType> storage;
    std::vector<valType > values;
    for (int i = 0; i < order; ++i)
    {
        vType coord = std::stoul(firstLineSplitted[i]) - 1;
        dims[i] = coord + 1;
        storage.push_back(coord);
    }
    values.push_back(std::stod(firstLineSplitted[order]));
    ++nnz;

    unsigned total = end - start;
    unsigned threadCount = omp_get_max_threads();
    unsigned long long chunkSize = std::ceil(static_cast<double>(total) / threadCount);

#pragma omp parallel num_threads(threadCount) default(none) shared(threadCount, order, dims, nnz, storage, values, start, end, chunkSize)
    {
        unsigned myID = omp_get_thread_num();
        char* myAddrStart;
        char* myAddrEnd;

        // start positioning
        if (myID == 0)
        {
            myAddrStart = start;
        }
        else
        {
            myAddrStart = start + myID * chunkSize;
            while (myAddrStart < end && *(myAddrStart - 1) != '\n')
            {
                ++myAddrStart;
            }
        }

        // end positioning
        if (myID == threadCount - 1)
        {
            myAddrEnd = end;
        }
        else
        {
            myAddrEnd = start + (myID + 1) * chunkSize;
            while (myAddrEnd < end && *myAddrEnd != '\n')
            {
                ++myAddrEnd;
            }
        }

        // ensuring that the subsequent threads do not capture the same 2 lines into their chunk
        if (myID < threadCount - 1)
        {
            char* nextThreadStart = start + (myID + 1) * chunkSize;
            while (nextThreadStart < end && *(nextThreadStart - 1) != '\n')
            {
                ++nextThreadStart;
            }
            if (myAddrEnd > nextThreadStart)
            {
                myAddrEnd = nextThreadStart - 1;
            }
        }

        // thread private tensor
        vType* t_dims = new vType[order];
        memset(t_dims, 0, sizeof(vType) * order);
        eType t_nnz = 0;

        std::vector<vType> t_storage;
        std::vector<valType > t_values;
        //

        std::string line;
        std::vector<std::string> lineSplitted;
        while (myAddrStart < myAddrEnd)
        {
            line = getLine(myAddrStart);
            lineSplitted = split(line, ' ');
            if (lineSplitted.size() != (order + 1)) continue;
            for (int i = 0; i < order; ++i)
            {
                vType coord = std::stoul(lineSplitted[i]) - 1;
                if ((coord + 1) > t_dims[i])
                {
                    t_dims[i] = coord + 1;
                }
                t_storage.push_back(coord);
            }
            t_values.push_back(std::stod(lineSplitted[order]));
            ++t_nnz;
        }

        #pragma omp critical
        {
            nnz += t_nnz;
            for (int i = 0; i < order; ++i)
            {
                if (t_dims[i] > dims[i])
                {
                    dims[i] = t_dims[i];
                }
            }

            for (int i = 0; i < t_nnz * order; ++i)
            {
                storage.push_back(t_storage[i]);
            }

            for (int i = 0; i < t_nnz; ++i)
            {
                values.push_back(t_values[i]);
            }
        };

        delete[] t_dims;
    };

    close(fd);
    munmap(addr, sb.st_size);

    vType* storageArr = new vType[nnz * order];
    memcpy(storageArr, storage.data(), sizeof(vType) * nnz * order);
    valType* valuesArr = new valType[nnz];
    memcpy(valuesArr, values.data(), sizeof(valType) * nnz);
    tensor = new SparseTensorCOO(tensorName, order, dims, nnz, valuesArr, storageArr);

    double end_time = omp_get_wtime();

    logger->logReadingTensorMarket(tensor, end_time - start_time);

    SparseVizIO::writeTensorToBinaryFile(binaryFileName, tensor);

    return tensor;
}

void SparseVizIO::writeTensorToBinaryFile(const std::string &binaryFileName, SparseTensor *tensor)
{
    FILE *fp = nullptr;
    if ((fp = fopen(binaryFileName.c_str(), "w")) == nullptr)
    {
        std::cout << binaryFileName << " cannot be created" << endl;
        return;
    }

    double start_time = omp_get_wtime();

    vType order = tensor->getOrder();
    vType* dims = tensor->getDims();
    eType nnz = tensor->getNNZ();
    valType* values = tensor->getValues();
    vType* storage = dynamic_cast<SparseTensorCOO*>(tensor)->getStorage();

    fwrite(&order, sizeof(vType), 1, fp);
    fwrite(dims, sizeof(vType), order, fp);
    fwrite(&nnz, sizeof(eType), 1, fp);
    fwrite(values, sizeof(valType), nnz, fp);
    fwrite(storage, sizeof(vType), nnz * order, fp);

    fclose(fp);

    double end_time = omp_get_wtime();

    logger->logWritingTensorBinary(tensor, end_time - start_time);
}

SparseTensor *SparseVizIO::readTensorFromBinaryFile(const std::string &binaryFileName, const std::string& name)
{
    FILE *fp = nullptr;
    if ((fp = fopen(binaryFileName.c_str(), "r")) == nullptr)
    {
        return nullptr;
    }

    double start_time = omp_get_wtime();

    vType order;
    vType* dims;
    eType nnz;

    fread(&order, sizeof(vType), 1, fp);
    dims = new vType[order];
    fread(dims, sizeof(vType), order, fp);
    fread(&nnz, sizeof(eType), 1, fp);
    SparseTensor* tensor = new SparseTensorCOO(name, order, dims, nnz);
    fread(tensor->getValues(), sizeof(valType), nnz, fp);
    fread(dynamic_cast<SparseTensorCOO*>(tensor)->getStorage(), sizeof(vType), nnz * order, fp);

    double end_time = omp_get_wtime();

    logger->logReadingTensorBinary(tensor, end_time - start_time);

    return tensor;
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
    FILE *fp = nullptr;
    if ((fp = fopen(binaryFileName.c_str(), "r")) != nullptr)
    {
        orderedDimensions = new vType*[norder];
        for (vType i = 0; i < norder; ++i)
        {
            orderedDimensions[i] = new vType[dims[i]];
            fread(orderedDimensions[i], sizeof(vType), dims[i], fp);
        }
        fclose(fp);
        return true;
    }
    return false;
}

void SparseVizIO::writeTensorOrderingToBinaryFile(const std::string &binaryFileName, vType norder, vType* dims, vType** orderedDimensions)
{
    FILE *fp = nullptr;
    if ((fp = fopen(binaryFileName.c_str(), "r")) != nullptr)
    {
        for (vType i = 0; i < norder; ++i)
        {
            fwrite(orderedDimensions[i], sizeof(vType), dims[i], fp);
        }
        fclose(fp);
    }
}
