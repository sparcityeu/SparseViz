#include "config.h"
#include "PatohOrdering.h"
#include "COOKPartiteOrdering.h"
#include "SparseVizEngine.h"
#include "SparseMatrix.h"
#include <stdexcept>
#include "omp.h"
#include "SparseVizTest.h"


int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        throw std::runtime_error("You must provide your config file path as an argument to this program!");
    }
    omp_set_nested(1);
    omp_set_dynamic(0);

    omp_init_lock(&(PatohOrdering::patoh_lock));
    omp_init_lock(&(COOKPartiteOrdering::kpartite_lock));
 
    omp_set_num_threads(omp_get_max_threads());

    std::string arg1 = argv[1];
    if (arg1 == "test")
    {
#ifndef TEST
        {
            throw std::runtime_error("You are trying to run the tester program without TEST macro defined in the source code. Either provide executable the -DTEST argument or add the TEST macro into the config.h file.");
        }
#endif
        unsigned testNo;
        try
        {
            testNo = std::stoul(argv[2]);
        }
        catch (const std::invalid_argument& e)
        {
            throw std::runtime_error("You have indicated an invalid argument for the number of tests to be conducted.");
        }
        SparseVizTest* tester =  SparseVizTest::getSparseVizTester();
        tester->runTests(testNo);
    }
    else
    {
        ConfigFileReader reader(argv[1]);
        SparseVizEngine* engine = reader.instantiateEngine();
        engine->runEngine();
    }

    omp_destroy_lock(&(PatohOrdering::patoh_lock));
    omp_destroy_lock(&(COOKPartiteOrdering::kpartite_lock));

    std::unordered_map<std::string, SparseMatrix*>::iterator kpartite_matrices_iter;
    for (kpartite_matrices_iter = (COOKPartiteOrdering::kpartite_matrices).begin(); kpartite_matrices_iter != (COOKPartiteOrdering::kpartite_matrices).end(); kpartite_matrices_iter++) {
        delete kpartite_matrices_iter->second;
    }

    std::unordered_map<std::string, omp_lock_t*>::iterator kpartite_locks_iter;
    for (kpartite_locks_iter = (COOKPartiteOrdering::kpartite_locks).begin(); kpartite_locks_iter != (COOKPartiteOrdering::kpartite_locks).end(); kpartite_locks_iter++) {
        omp_destroy_lock((kpartite_locks_iter->second));
        delete kpartite_locks_iter->second;
    }
    return 0;
}
