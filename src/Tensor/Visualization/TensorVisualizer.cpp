#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random> // for std::default_random_engine
#include <chrono> // for seeding the random number generator
#include <cassert>
#include <omp.h>
#include "TensorVisualizer.h"
#include "SparseVizLogger.h"
#include "SparseVizTest.h"
#include "SparseTensorCOO.h"

using namespace std;

std::string stat_to_html_table(const TensorOrdering &o, const TStatistic &stat)
{
    std::stringstream stream;
    std::string temp;

    std::ostringstream table;
    table << "<div class=\"container\">\n";
    table << "<table class=\"responsivep-table\">\n";
    table << "<caption style=\"background-color:#FF3030; color:white;\"	>";
    table << "Statistics for ";
    temp = o.getOrderingName();
    std::transform(temp.begin(), temp.end(), temp.begin(), ::toupper);
    table << temp;
    table << "</caption>\n";
    table << "<thead>\n";
    table << "<tr>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Bin Count</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Empty Bins</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Avg. NNZ</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Median NNZ</th>\n";
    table << "</tr>\n";
    table << "</thead>\n";
    table << "<tbody>\n";
    table << "<tr>\n";
    stream << std::fixed << std::setprecision(0) << stat.no_bins;
    table << "<th scope=\"row\" Bin Count=\"#D3D3D3\" color=\"white\" align=\"left\">" << stream.str() << "</th>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.no_empty_bins;
    table << "<td data-title=\"Empty Bins\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.mean_nnz;
    table << "<td data-title=\"<Avg. NNZ>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.median_nnz;
    table << "<td data-title=\"<Median NNZ>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    table << "</tr>\n";
    table << "<thead>\n";
    table << "<tr>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Fiber</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">X-Y</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">X-Z</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Y-Z</th>\n";
    table << "</tr>\n";
    table << "</thead>\n";
    table << "<tbody>\n";
    table << "<tr>\n";
    table << "<th scope=\"row\" Fiber=\"#D3D3D3\" color=\"white\" align=\"left\">" << ">1 nnz" << "</th>\n";
    stream << std::fixed << std::setprecision(0) << stat.fiberCounts[0];
    table << "<td data-title=\"<X-Y>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.fiberCounts[1];
    table << "<td data-title=\"<X-Z>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.fiberCounts[2];
    table << "<td data-title=\"<Y-Z>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    table << "</tr>\n";
    table << "<tr>\n";
    table << "<th scope=\"row\" Fiber=\"#D3D3D3\" color=\"white\" align=\"left\">" << "1 nnz" << "</th>\n";
    stream << std::fixed << std::setprecision(0) << stat.singleNNZfiberCounts[0];
    table << "<td data-title=\"<X-Y>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.singleNNZfiberCounts[1];
    table << "<td data-title=\"<X-Z>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.singleNNZfiberCounts[2];
    table << "<td data-title=\"<Y-Z>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    table << "</tr>\n";
    table << "<thead>\n";
    table << "<tr>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Stat</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Average</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Maximum</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Normalized</th>\n";
    table << "</tr>\n";
    table << "</thead>\n";
    table << "<tbody>\n";
    table << "<tr>\n";
    table << "<th scope=\"row\" bgcolor=\"#D3D3D3\" color=\"white\" align=\"left\">X-Span</th>\n";
    stream << std::fixed << std::setprecision(0) << stat.avgSpan[2];
    table << "<td data-title=\"Average\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.maxSpan[2];
    table << "<td data-title=\"<Maximum>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.normSpan[2];
    table << "<td data-title=\"<Normalized>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    table << "</tr>\n";
    table << "<tr>\n";
    table << "<th scope=\"row\" bgcolor=\"#D3D3D3\" color=\"white\" align=\"left\">Y-Span</th>\n";
    stream << std::fixed << std::setprecision(0) << stat.avgSpan[1];
    table << "<td data-title=\"Average\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.maxSpan[1];
    table << "<td data-title=\"<Maximum>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.normSpan[1];
    table << "<td data-title=\"<Normalized>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    table << "</tr>\n";
    table << "<tr>\n";
    table << "<th scope=\"row\" bgcolor=\"#D3D3D3\" color=\"white\" align=\"left\">Z-Span</th>\n";
    stream << std::fixed << std::setprecision(0) << stat.avgSpan[0];
    table << "<td data-title=\"Average\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.maxSpan[0];
    table << "<td data-title=\"<Maximum>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.normSpan[0];
    table << "<td data-title=\"<Normalized>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    table << "</tr>\n";
    table << "</table>\n";
    
    auto results = o.getKernelResults();
    if(results.size() > 0) 
    { 
        int maxTC = 0;
        for(auto& res : results) {
            maxTC = std::max((int)maxTC, (int)(res.threadCounts.size()));
        }

        table << "<table class=\"responsive-table\">\n";
        table << "<caption style=\"background-color:#FF3030; color:white;\"	> KERNEL EXECUTION TIMES  </caption>\n";
        for(auto& res : results) 
        {
            table << "<thead>\n";
            table << "<tr>\n";
            table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Name</th>\n";
            table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Scheduling</th>\n";
            table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Chunk</th>\n";
            for(int i = 0; i < res.durations.size(); i++) {
                if(i != res.durations.size() - 1) {
                    table << "<th align=\"right\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">" << res.threadCounts[i] << "</th>\n";
                } else {
                    table << "<th colspan=\"" << maxTC - i << "\" align=\"right\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">" << res.threadCounts[i] << "</th>\n";
                }
            }
            table << "</tr>\n";
            table << "</thead>\n";
            table << "<tbody>\n";
            table << "<tr>\n";
            table << "<th scope=\"row\" bgcolor=\"#D3D3D3\" color=\"white\" align=\"left\">" <<  res.kernelName <<"</th>\n";
            table << "<td data-title=\"Scheduling\" data-type=\"number\" align=\"left\">" << res.schedulingPolicy << "</td>\n";
            if(res.chunkSize == 0) {
                table << "<td data-title=\"Chunk\" data-type=\"text\" align=\"left\">" << "Default" << "</td>\n";
            } else {
                table << "<td data-title=\"Chunk\" data-type=\"text\" align=\"left\">" << res.chunkSize << "</td>\n";
            }
            for(int i = 0; i < res.durations.size(); i++) {
                stream.str(std::string());
                stream << std::fixed << std::setprecision(3) << res.durations[i];
                if(i != res.durations.size() - 1) {
                    table << "<td data-title=\"Exec Time\" data-type=\"text\" align=\"right\">" << stream.str() << "</td>\n";
                } else {
                    table << "<td colspan=\"" << maxTC - i << "\" data-title=\"Exec Time\" data-type=\"text\" align=\"right\">" << stream.str() << "</td>\n";
                }
                stream.str(std::string());
            }
            table << "</tr>\n";
            table << "</tbody>\n";
        }
        table << "</table>\n";
    }

    auto gpu_results = o.getGPUKernelResults();
    if(gpu_results.size() > 0) 
    { 
        table << "<table class=\"responsive-table\">\n";
        table << "<caption style=\"background-color:#FF3030; color:white;\"	> GPU KERNEL EXECUTION TIMES  </caption>\n";

        table << "<thead>\n";
        table << "<tr>\n";
        table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Name</th>\n";
        table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Grid Sz</th>\n";
        table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Block Sz</th>\n";
        table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Runime</th>\n";
        table << "</tr>\n";
        table << "</thead>\n";

        for(auto& res : gpu_results) 
        {        
            table << "<tbody>\n";
            table << "<tr>\n";
            for(int i = 0; i < res.durations.size(); i++) {
                table << "<th scope=\"row\" bgcolor=\"#D3D3D3\" color=\"white\" align=\"left\">" <<  res.kernelName <<"</th>\n";
                table << "<td data-title=\"Grid Sz\" data-type=\"number\" align=\"left\">" << res.gridSizes[i] << "</td>\n";
                table << "<td data-title=\"Block Sz\" data-type=\"number\" align=\"left\">" << res.blockSizes[i] << "</td>\n";
                table << "<td data-title=\"Execution Time\" data-type=\"text\" align=\"left\">" << res.durations[i] << "</td>\n";
            }
            table << "</tr>\n";
            table << "</tbody>\n";
        }
        table << "</table>\n";
    }

    table << "</div>\n";
    return table.str();
}

void visualizeTensorOrderings(TensorOrdering** orderings, int norder) {
    double start_time = omp_get_wtime();

    const SparseTensorCOO& tensor = dynamic_cast<const SparseTensorCOO&>(orderings[0]->getTensor());
    const std::vector<vType>& active_modes = orderings[0]->getActiveModes();

    std::string filename = tensor.getName();
    for(const vType& m : active_modes) {
        filename += "_" + std::to_string(m);
    }

    const int tensor_order = tensor.getOrder();
    const vType* full_dims = tensor.getDims();
    const vType* nonzeros = tensor.getStorage();
    const valType* values = tensor.getValues();
    const vType nnzCount = tensor.getNNZ();

    logger->makeSilentLog( "visualizeTensorOrderings is started for " + filename);

    const vType dims[3] = {full_dims[active_modes[0]], full_dims[active_modes[1]], full_dims[active_modes[2]]};

    vType scaleFactor = std::max(std::max(dims[0], dims[1]), dims[2]) / MAX_DIM;
    scaleFactor = scaleFactor == 0 ? 1: scaleFactor;
    vType *vis_dims = new vType[3];
    for(int i = 0; i < 3; i++) {
        vis_dims[i] = dims[i] / scaleFactor;
        if (vis_dims[i] < 8) vis_dims[i] = 8;
        if(dims[i] < vis_dims[i]) vis_dims[i] = dims[i];
    }
   
    TensorBin ****tensorLists = new TensorBin ***[norder];
    for (int n = 0; n < norder; ++n) {
        tensorLists[n] = new TensorBin **[vis_dims[0]];
        for (int i = 0; i < vis_dims[0]; ++i) {
            tensorLists[n][i] = new TensorBin *[vis_dims[1]];
            for (int j = 0; j < vis_dims[1]; ++j) {
                tensorLists[n][i][j] = new TensorBin[vis_dims[2]];
            }
        }
    }

    std::unordered_map<vpair, vType, pair_hash> fibers[3];
   
    std::vector<vType> fiberMins[3]; for(int i = 0; i < 3; i++) fiberMins[i].reserve(nnzCount/3);
    std::vector<vType> fiberMaxs[3]; for(int i = 0; i < 3; i++) fiberMaxs[i].reserve(nnzCount/3);
    std::vector<vType> fiberNNZs[3]; for(int i = 0; i < 3; i++) fiberNNZs[i].reserve(nnzCount/3);

    std::unordered_map<vpair, vType>::iterator iter; 
    //std::cout << "DIMS " <<  dims[0] << " " <<  dims[1] << " " << dims[2] << std::endl;
    //std::cout << "VDIMS " <<  vis_dims[0] << " " <<  vis_dims[1] << " " << vis_dims[2] << std::endl;

    // Iterate over the nonzeros
    vType ordered_nnz[3];
    vType binIDs[3];


    TStatistic stats[norder];
    double start_t = 0;
    for (vType i = 0; i < nnzCount; i++) {
        vType x = nonzeros[(i * tensor_order) + active_modes[0]]; 
        vType y = nonzeros[(i * tensor_order) + active_modes[1]]; 
        vType z = nonzeros[(i * tensor_order) + active_modes[2]]; 
     
        valType val = values[i];
        
        vType vec_locs[3];
        for(int d = 0; d < 3; d++) {
            vpair pair;
            if(d == 2) pair.first = y; else pair.first = x;
            if(d == 0) pair.second = y; else pair.second = z; 

            std::unordered_map<vpair, vType>::iterator iter = fibers[d].find(pair);
            if(iter == fibers[d].end()) {
                vec_locs[d] = fibers[d][pair] = fiberMins[d].size();

                for(int n = 0; n < norder; n++) {
                    fiberMins[d].push_back(dims[2 - d]); 
                    fiberMaxs[d].push_back(0);
                }

                fiberNNZs[d].push_back(1);
            } else {
                vec_locs[d] = iter->second;
                fiberNNZs[d][(iter->second)/norder]++;
            }
        }

        for (int n = 0; n < norder; ++n) {
            vType** order = orderings[n]->getOrderedDimensions();

            ordered_nnz[0] = order[active_modes[0]][x];
            ordered_nnz[1] = order[active_modes[1]][y];
            ordered_nnz[2] = order[active_modes[2]][z];
            if(ordered_nnz[0] >= dims[0] || ordered_nnz[1] >= dims[1] || ordered_nnz[2] >= dims[2]) {
                std::cerr << orderings[n]->getOrderingName() << ": Error - Ordered "    << ordered_nnz[0] << " " << dims[0] << " | "
                                                                                        << ordered_nnz[1] << " " << dims[1] << " | "
                                                                                        << ordered_nnz[2] << " " << dims[2] << std::endl;
                throw std::runtime_error("a");
                return;
            }
            for(int d = 0; d < 3; d++) {
                binIDs[d] = calculateBin(ordered_nnz[d], dims[d], vis_dims[d]);
                if (binIDs[d] < 0 || binIDs[d] >= vis_dims[d]) {
                    cerr << "Unexpected bin index value. bin: " << d << ", " <<  binIDs[d] << std::endl;
                    return;
                }
            }

            TensorBin& tbin = tensorLists[n][binIDs[0]][binIDs[1]][binIDs[2]];
            tbin.nonzeroCount++;
            tbin.totalValues += val;
            tbin.absTotalValues += abs(val);

            for(int d = 0; d < 3; d++) {
                if(fiberMins[d][vec_locs[d] + n] > ordered_nnz[2-d]) {
                    fiberMins[d][vec_locs[d] + n] = ordered_nnz[2-d];
                }
                if(fiberMaxs[d][vec_locs[d] + n] < ordered_nnz[2-d]) {
                    fiberMaxs[d][vec_locs[d] + n] = ordered_nnz[2-d];
                }
            }
        }
    }
    //std::cout << xsum << " " << ysum << " " << zsum << std::endl;
    //std::cout << "Fiber Counts " << fiberMins[0].size()/norder << " " << fiberMins[1].size()/norder << " " << fiberMins[2].size()/norder << std::endl;

    for(int d = 0; d < 3; d++) {
        for(vType v = 0; v < fiberMins[d].size(); v += norder) {
            for(int n = 0; n < norder; n++) {
                vType diff = fiberMaxs[d][v + n] - fiberMins[d][v + n] + 1;
                //std::cout << d << " " << v << " " << n << " " << diff << std::endl;
                if(diff > 1) { //checks if there is more than one nonzero in this fiber
                    stats[n].fiberCounts[d]++;
                    stats[n].avgSpan[d] += diff;
                    stats[n].maxSpan[d] = std::max((int)stats[n].maxSpan[d], (int)diff);
                    stats[n].normSpan[d] += ((double)diff) / fiberNNZs[d][v / norder];
                } else if(diff == 0) {
                    stats[n].singleNNZfiberCounts[d]++;
                }
            }
        }
    }    

    for(int d = 0; d < 3; d++) {
        for(int n = 0; n < norder; n++) {
            if(stats[n].fiberCounts[d] == 0) {
                stats[n].avgSpan[d] = stats[n].normSpan[d] = stats[n].maxSpan[d] = 0;
            } else {
                stats[n].avgSpan[d] /= stats[n].fiberCounts[d]; 
                stats[n].normSpan[d] /= stats[n].fiberCounts[d]; 
            }
        }
    }

    double end_time = omp_get_wtime();

    logger->makeSilentLog(orderings[0]->getTensor().getName() + " - nonzeros are processed in ", end_time - start_time);

    for (int i = 0; i != norder; ++i)
    {
        stats[i].tensorName = orderings[i]->getTensor().getName();
        stats[i].orderingName = orderings[i]->getOrderingName();
        logger->logTensorProcessing(TENSOR_VISUALIZATION_FILES_DIR + filename + ".html", stats[i], end_time - start_time);
    }

    //std::cout << "Fiber Counts " <<  fiberCounts[0] << " " <<  fiberCounts[1] << " " <<  fiberCounts[2] << std::endl;
    //std::cout << "Single NNZ Fiber Counts " <<  singleNNZfiberCounts[0] << " " <<  singleNNZfiberCounts[1] << " " <<  singleNNZfiberCounts[2] << std::endl;

    std::string filePath;

#ifdef TEST
    filePath = SparseVizTest::getSparseVizTester()->getCurrentDirectory() + filename + ".html";
#else
    filePath = TENSOR_VISUALIZATION_FILES_DIR + filename + ".html";
#endif

    std::ofstream html_file(filePath);
    html_file << R"(
   <!DOCTYPE html>
   <html>
   <head>
   <meta charset="UTF-8">
   <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
   <style>
   body {
      font-family: 'Orbitron', sans-serif;
      font-size:70%;
   }
   .header {
   display: flex;
      justify-content: space-between;
      align-items: center;
   width: 100%;
   }
   .filename {
      font-size: 18px;
   }
   .title {
      text-align: left;
   }
   .title-main {
   margin: 0;
   }
   .title-sub {
   margin: 0;
   }
   margin: 0;
   padding: 0;
      box-sizing: border-box;
   }
   .hoverlayer
   {
   z-index: 1000 !important;
   opacity: 1 !important;
   visibility: visible !important;
   }
   </style>
   <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
   <link rel="icon" href="favicon.ico" type="image/x-icon">
   </head>
   <body>
   <div class="header">
   <div class="title">
   <h1 class="title-main">SparseViz Tensor</h1>
   <h2 class="title-sub">Visualization</h2>
   </div>
   
   
    <div class="title">
   <h2>Tensor Name: )" + filename + R"(</h2><hr>)";
    html_file << "<h3 class=\"title-sub\">Dimensions: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "</h3>";
    html_file << "<h3 class=\"title-sub\">Nonzeros: " << nnzCount << "</h3>";

    html_file << "</div>\n"; // Close right header div
    html_file << "</div>\n"; // Close header div

    html_file << "<div id=\"aspect\">\n";
    html_file << "<button onClick=\"choose('data')\">Actual Tensor Sizes</button>\n";
    html_file << "<button onClick=\"choose('cube')\">Cube Tensors</button>\n";
    html_file << "<script>\n";
    html_file << "function choose(choice) {\n";
    for(int n = 0; n < norder; n++) {
        html_file << "choose_" << n << "(choice);\n";
    }
    html_file << "}\n";
    html_file << "</script>\n";
    html_file << "</div>\n";

    for (int n = 0; n < norder; n++) {
        vector<vector<int>> topDownSum(vis_dims[0], vector<int>(vis_dims[1], 0));
        vector<vector<int>> sideSum(vis_dims[1], vector<int>(vis_dims[2], 0));
        vector<vector<int>> tubeSum(vis_dims[0], vector<int>(vis_dims[2], 0));
        vector<vector<int>> topDownTotalValues(vis_dims[0], vector<int>(vis_dims[1], 0));
        vector<vector<int>> sideTotalValues(vis_dims[1], vector<int>(vis_dims[2], 0));
        vector<vector<int>> tubeTotalValues(vis_dims[0], vector<int>(vis_dims[2], 0));
        vector<vector<int>> topDownAbsTotalValues(vis_dims[0], vector<int>(vis_dims[1], 0));
        vector<vector<int>> sideAbsTotalValues(vis_dims[1], vector<int>(vis_dims[2], 0));
        vector<vector<int>> tubeAbsTotalValues(vis_dims[0], vector<int>(vis_dims[2], 0)); // for heatmaps
        nlohmann::json json_array;                                                        // for 3D Scatter plot
        nlohmann::json x_bins_json, y_nonzero_json;                                       // for barPlots
        nlohmann::json x_bins_abs_json, y_absTotalValues_json;                            // for barPlots
        vector<int> nonZeros;

        stats[n].no_bins =  vis_dims[2] * vis_dims[1] * vis_dims[0];
        stats[n].no_empty_bins = 0;
        for (int x = 0; x < vis_dims[0]; x++) {
            for (int y = 0; y < vis_dims[1]; y++) {
                for (int z = 0; z < vis_dims[2]; z++) {
                    nlohmann::json bin_json;
                    bin_json["binX"] = x;
                    bin_json["binY"] = y;
                    bin_json["binZ"] = z;
                    bin_json["nonzeroCount"] = tensorLists[n][x][y][z].nonzeroCount;
                    bin_json["totalValues"] = tensorLists[n][x][y][z].totalValues;
                    bin_json["absTotalValues"] = tensorLists[n][x][y][z].absTotalValues;
                    json_array.push_back(bin_json);
                    int nonZeroCount = tensorLists[n][x][y][z].nonzeroCount;

                    if (nonZeroCount == 0) {
                        stats[n].no_empty_bins++;
                        nonZeroCount = 1;
                    }
                    stats[n].geo_mean_nnz += log(nonZeroCount);

                    topDownSum[x][y] += tensorLists[n][x][y][z].nonzeroCount;
                    sideSum[y][z] += tensorLists[n][x][y][z].nonzeroCount;
                    tubeSum[x][z] += tensorLists[n][x][y][z].nonzeroCount;

                    topDownTotalValues[x][y] += tensorLists[n][x][y][z].totalValues;
                    sideTotalValues[y][z] += tensorLists[n][x][y][z].totalValues;
                    tubeTotalValues[x][z] += tensorLists[n][x][y][z].totalValues;

                    topDownAbsTotalValues[x][y] += tensorLists[n][x][y][z].absTotalValues;
                    sideAbsTotalValues[y][z] += tensorLists[n][x][y][z].absTotalValues;
                    tubeAbsTotalValues[x][z] += tensorLists[n][x][y][z].absTotalValues;

                    string coord_str = "(" + to_string(x) + ", " + to_string(y) + ", " + to_string(z) + ")";
                    x_bins_json.push_back(coord_str);                                        // insert bin number
                    y_nonzero_json.push_back(tensorLists[n][x][y][z].nonzeroCount);          // insert nonzeroCount value
                    x_bins_abs_json.push_back(coord_str);                                    // insert bin number
                    y_absTotalValues_json.push_back(tensorLists[n][x][y][z].absTotalValues); // insert absTotalValues value
                    if (tensorLists[n][x][y][z].nonzeroCount != 0) {
                        nonZeros.push_back(tensorLists[n][x][y][z].nonzeroCount);
                    }
                }
            }
        }

        std::sort(nonZeros.begin(), nonZeros.end());
        if (nonZeros.size() % 2 == 0) {
            // Even number of elements
            stats[n].median_nnz = (nonZeros[nonZeros.size() / 2 - 1] + nonZeros[nonZeros.size() / 2]) / 2.0;
        } else {
            // Odd number of elements
            stats[n].median_nnz = nonZeros[nonZeros.size() / 2];
        }
        stats[n].geo_mean_nnz = stats[n].geo_mean_nnz / (stats[n].no_bins - stats[n].no_empty_bins);
        stats[n].geo_mean_nnz = exp(stats[n].geo_mean_nnz);
        stats[n].mean_nnz = (double(nnzCount)) / (stats[n].no_bins - stats[n].no_empty_bins);
       
        vector<vector<int>> *sums[3] = {&topDownSum, &sideSum, &tubeSum};
        vector<vector<int>> *totalValuesArrays[] = {&topDownTotalValues, &sideTotalValues, &tubeTotalValues};
        vector<vector<int>> *absTotalValuesArrays[] = {&topDownAbsTotalValues, &sideAbsTotalValues,
                                                       &tubeAbsTotalValues};

        // converting the nlohmann::json object to a string
        string json_str = json_array.dump();
        json_str = escapeSingleQuote(json_str);

      html_file << "<div id='orderDiv" << n
                  << "' style='display: flex; flex-direction: row; align-items: center; "
                     "justify-content: space-around; margin-bottom: 5px;'>\n";
        html_file << "<div style='writing-mode: vertical-rl; display:flex; justify-content:center; align-items:center; transform: "
                     "rotate(180deg); margin-left:20px; margin-right: 10px;'>\n";
        html_file << "<h2>" << orderings[n]->getOrderingName() << "</h2>\n";
        html_file << "</div>\n";
        html_file
                << "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: center; width: 100%;'>\n";

        // 3D scatter plot
        html_file << "<div id='myDiv" << n << "' style='width: 600px; height: 700px;'></div>\n";
        html_file
                << "<div style='display: flex; flex-direction: column; justify-content: space-between; width: 100%;'>\n"; // Change to column

        // js code for 3D scatter plot
        html_file << "<script>\n";
        html_file << "choose_" << n << "('data');\n";
         html_file << "function choose_" << n << "(choice)\n{\n";
    html_file << R"(
      try {
         var jsonData = ')" +
                     json_str + R"(';  // insert your JSON string here
         var data = JSON.parse(jsonData);

         // Extracting x, y, z coordinates and other data
         var x = data.map(d => d.binX);
         var y = data.map(d => d.binY);
         var z = data.map(d => d.binZ);
         var nonzeroCount = data.map(d => d.nonzeroCount);
         var totalValues = data.map(d => d.totalValues);
         var absTotalValues = data.map(d => d.absTotalValues);
         // Find the maximum nonZeroCount to normalize the counts
         var maxCount = Math.max(...data.map(d => d.nonzeroCount));
         // Generating hover text
         var hoverText = data.map(d => `Non-zero values: ${d.nonzeroCount}<br>Total value: ${d.totalValues}<br>Absolute total value: ${d.absTotalValues}`);

         // Creating the 3D scatter plot
         var xNonZero = [], yNonZero = [], zNonZero = [], countsNonZero = [], hoverTextNonZero = [];
         var xZero = [], yZero = [], zZero = [], hoverTextZero = [];

         for (var i = 0; i < nonzeroCount.length; i++) {
            if (nonzeroCount[i] == 0) {
               xZero.push(x[i] *)" << dims[0] / vis_dims[0] << R"( );
               yZero.push(y[i] *)" << dims[1] / vis_dims[1] << R"( );
               zZero.push(z[i] *)" << dims[2] / vis_dims[2] << R"( );
               hoverTextZero.push(hoverText[i]);
            } else {
               xNonZero.push(x[i] *)" << dims[0] / vis_dims[0] << R"( );
               yNonZero.push(y[i] *)" << dims[1] / vis_dims[1] << R"( );
               zNonZero.push(z[i] *)" << dims[2] / vis_dims[2] << R"( );
               countsNonZero.push(nonzeroCount[i]);
               hoverTextNonZero.push(hoverText[i]);
            }
         }

         // Create two separate traces
         var traceNonZero = {
         x: xNonZero,
         y: yNonZero,
         z: zNonZero,
         mode: 'markers',
         type: 'scatter3d',
         marker: {
         size: 8,
         color: countsNonZero,
         colorscale: 'Reds',
         opacity: 0.9
         },
         text: hoverTextNonZero,
         hoverinfo: 'text'
         };

         var traceZero = {
         x: xZero,
         y: yZero,
         z: zZero,
         mode: 'markers',
         type: 'scatter3d',
         marker: {
         size: 8,
         color: 'rgba(255,0,0,0)',  // Transparent red
         opacity: 0
         },
         text: hoverTextZero,
         hoverinfo: 'text'
         };

         var layout = {
         autosize: true,
         scene: {
            aspectmode: choice
         }, 
         width: 650,
         yaxis: {
            autorange: 'reversed' // This is for adjusting the main diagonal correctly
         },
         xaxis: {
            side: 'top'
         },
         height: 650,
         margin: {
         l: 10,
         r: 80,
         b: 10,
         t: 10
         }
         };

         var data = [traceNonZero];

         Plotly.newPlot('myDiv)" +
                     to_string(n) + R"(', data, layout);
      } catch (err) {
         console.log('Error parsing JSON or plotting data: ' + err.message);
      }
      }
      </script>
      )";

        html_file
                << "<div style='display: flex; flex-direction: row; justify-content: flex-start; width: 100%;'>\n"; // heatmap container

        for (int perspective = 0; perspective < 3; perspective++) { // 3 perspectives fibers rows tubes
            nlohmann::json json_arr;
            auto &sum = *sums[perspective];
            auto &totalValues = *totalValuesArrays[perspective];
            auto &absTotalValues = *absTotalValuesArrays[perspective];
            string titles[] = {"'Dimensions: X-Y'", "'Dimensions: Y-Z'", "'Dimensions: X-Z'"};
            string showColorBar = "false", width = "230";
            if(perspective == 2) {
                showColorBar = "true";
                width = "300";
                html_file << "<div id='myDiv" << n << "_" << perspective
                      << "' style='width: 300; height: 250;'></div>\n";
            } else {
                html_file << "<div id='myDiv" << n << "_" << perspective
                      << "' style='width: 230; height: 250;'></div>\n";
            }

            for (int i = 0; i < sum.size(); i++) {
                for (int j = 0; j < sum[i].size(); j++) {
                    nlohmann::json sum_json;
                    sum_json["x"] = i;
                    sum_json["y"] = j;
                    sum_json["sum"] = sum[i][j];
                    sum_json["totalValues"] = totalValues[i][j];
                    sum_json["absTotalValues"] = absTotalValues[i][j];
                    json_arr.push_back(sum_json);
                }
            }

            string json_str2 = json_arr.dump();
            json_str2 = escapeSingleQuote(json_str2);

            html_file << R"(
         <script>
         try {
            var jsonData2 = ')" + json_str2 + R"(';  // insert your JSON string here
            var data2 = JSON.parse(jsonData2);

            var x2 = [];
            var y2 = [];
            var z2 = [];
            var text = [];)";

            if(perspective == 0) {
            html_file << R"(  
            //custom hovertext creation
            for (var i = 0; i < data2.length; i++) {
               x2.push(data2[i].x *)" << dims[0] / vis_dims[0] << R"( );
               y2.push(data2[i].y *)" << dims[1] / vis_dims[1] << R"( );
               z2.push(data2[i].sum);
               text.push(
                         "x: " + data2[i].x *)" << dims[0] / vis_dims[0] << R"( +
                         "<br>y: " + data2[i].y *)" << dims[1] / vis_dims[1] << R"( +
                         "<br>NonZero: " + data2[i].sum +
                         "<br>Total Values: " + data2[i].totalValues +
                         "<br>Abs Total Values: " + data2[i].absTotalValues
                         );
            })";
            } else  if(perspective == 1) {
html_file << R"(  
            //custom hovertext creation
            for (var i = 0; i < data2.length; i++) {
               x2.push(data2[i].x *)" << dims[1] / vis_dims[2] << R"( );
               y2.push(data2[i].y *)" << dims[2] / vis_dims[2] << R"( );
               z2.push(data2[i].sum);
               text.push(
                         "y: " + data2[i].x *)" << dims[1] / vis_dims[1] << R"( +
                         "<br>z: " + data2[i].y *)" << dims[2] / vis_dims[2] << R"( +
                         "<br>NonZero: " + data2[i].sum +
                         "<br>Total Values: " + data2[i].totalValues +
                         "<br>Abs Total Values: " + data2[i].absTotalValues
                         );
            })";
            } else  if(perspective == 2) {
html_file << R"(  
            //custom hovertext creation
            for (var i = 0; i < data2.length; i++) {
               x2.push(data2[i].x *)" << dims[0] / vis_dims[0] << R"( );
               y2.push(data2[i].y *)" << dims[2] / vis_dims[2] << R"( );
               z2.push(data2[i].sum);
               text.push(
                         "x: " + data2[i].x *)" << dims[0] / vis_dims[0] << R"( +
                         "<br>z: " + data2[i].y *)" << dims[2] / vis_dims[2] << R"( +
                         "<br>NonZero: " + data2[i].sum +
                         "<br>Total Values: " + data2[i].totalValues +
                         "<br>Abs Total Values: " + data2[i].absTotalValues
                         );
            })";
            }



            html_file << R"( 
            // Creating the heatmap
            var trace2 = {
            x: x2,
            y: y2,
            z: z2,
            type: 'heatmap',
            colorscale: 'Reds',
            showscale: )" +
                         showColorBar + R"(,// set for only seen in the last heatmap
            text: text,
            hoverinfo: 'text'  // Only show the custom hover text
            };

            var layout2 = {
            title: )" + titles[perspective] +
                         R"(,
            autosize: false,
            width: )" + width +
                         R"(,// set different for the last heatmap since it has colorbar
            height: 250,
            yaxis: {
               autorange: 'reversed' // This is for adjusting the main diagonal correctly
            },
            xaxis: {
               side: 'bottom'
            },
            margin: {
            l: 50,
            r: 0,
            b: 50,
            t: 50
            }
            };

            var data2 = [trace2];

            Plotly.newPlot('myDiv)" +
                         to_string(n) + "_" + to_string(perspective) + R"(', data2, layout2);
            } catch (err) {
               console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }

        // Sorting together to create the visualization as intended.
        vector<pair<string, int>> sorted_bins;
        vector<pair<string, double>> sorted_abs_bins; // Note: Using double for absTotalValues

        for (size_t i = 0; i < x_bins_json.size(); i++) {
            sorted_bins.push_back(make_pair(x_bins_json[i].get<string>(), y_nonzero_json[i]));
        }

        for (size_t i = 0; i < x_bins_abs_json.size(); i++) {
            sorted_abs_bins.push_back(make_pair(x_bins_abs_json[i].get<string>(), y_absTotalValues_json[i]));
        }
        // sort in descending order of y values (nonzero counts)
        std::sort(sorted_bins.begin(), sorted_bins.end(),
             [](const std::pair<string, int> &a, const std::pair<string, int> &b) {
                 return a.second > b.second; // change this to a.second < b.second; for ascending order
             });
        // sort in descending order of y values (absTotalValues)
        std::sort(sorted_abs_bins.begin(), sorted_abs_bins.end(),
             [](const std::pair<string, double> &a, const std::pair<string, double> &b) {
                 return a.second > b.second; // for descending order
             });
        // create new sorted json arrays
        nlohmann::json sorted_x_bins_json, sorted_y_nonzero_json;
        nlohmann::json sorted_x_bins_abs_json, sorted_y_absTotalValues_json;

        for (const auto &bin: sorted_bins) {
            sorted_x_bins_json.push_back(bin.first);
            sorted_y_nonzero_json.push_back(bin.second);
        }
        // create new sorted json arrays
        for (const auto &bin: sorted_abs_bins) {
            sorted_x_bins_abs_json.push_back(bin.first);
            sorted_y_absTotalValues_json.push_back(bin.second);
        }
        // creating a json object with two sorted arrays
        nlohmann::json sorted_data_hist;
        nlohmann::json sorted_data_abs_hist;
        sorted_data_abs_hist["x"] = sorted_x_bins_abs_json;
        sorted_data_abs_hist["y"] = sorted_y_absTotalValues_json;
        sorted_data_hist["x"] = sorted_x_bins_json;
        sorted_data_hist["y"] = sorted_y_nonzero_json;
        // using sorted_data_hist to create the plots.

        string histogramStr = sorted_data_hist.dump();
        string histogramAbsStr = sorted_data_abs_hist.dump(); // named histograms but barPlots. note that this data is currently not in use but its duplicate is used.

        html_file << "<div id='histogram" << n
                  << "' style='width: 250px; height: 250px;'></div>\n"; // New div for histogram
        html_file << R"(
       <script>
       try {
          var histogramData = ')" +
                     histogramStr + R"(';  // insert your histogram JSON string here
          var data = JSON.parse(histogramData);

          var trace3 = {
          x: data.x,
          y: data.y,
          type: 'bar',
          marker: {
          color: 'red',
          },
          opacity: 0.7,
          };

          var layout3 = {
          title: 'Non-zero Counts Dist',
          xaxis: {
          title: 'Bin',
          type: 'category',  // treat x-axis labels as categories
          autorange: false,  // disable auto range
          range: [0, data.x.length],
          showticklabels: false
          },
          yaxis: {
          type: 'log',
          dtick: Math.log2()  // interval between ticks
          },
          autosize: false,
          width: 250,
          height: 250,
          margin: {
          l: 50,
          r: 0,
          b: 50,
          t: 50
          }
          };

          var data3 = [trace3];
          Plotly.newPlot('histogram)" +
                     to_string(n) + R"(', data3, layout3);
       } catch (err) {
          console.log('Error parsing JSON or plotting data: ' + err.message);
       }
       </script>
       )";

        html_file << "</div>\n"; // Close container for nnz heatmaps

        // New section for three new heatmaps based on absolute total values.
        html_file
                << "<div style='display: flex; flex-direction: row; justify-content: flex-start; width: 100%;'>\n"; // New heatmap container for absolute values

        for (int perspective = 0; perspective < 3; perspective++) {
            nlohmann::json json_arr_abs;
            auto &sum = *sums[perspective];
            auto &totalValues = *totalValuesArrays[perspective];
            auto &absTotalValues = *absTotalValuesArrays[perspective];
            string titles[] = {"'Dimensions: X-Y'", "'Dimensions: Y-Z'", "'Dimensions: X-Z'"};
            string showColorBar = "false", width = "230";
            if(perspective == 2) {
                showColorBar = "true";
                width = "300";
                html_file << "<div id='absDiv" << n << "_" << perspective
                      << "' style='width: 300; height: 250;'></div>\n";
            } else {
                html_file << "<div id='absDiv" << n << "_" << perspective
                      << "' style='width: 230; height: 250;'></div>\n";
            }

            for (int i = 0; i < absTotalValues.size(); i++) {
                for (int j = 0; j < absTotalValues[i].size(); j++) {
                    nlohmann::json abs_json;
                    abs_json["x"] = i;
                    abs_json["y"] = j;
                    abs_json["sum"] = sum[i][j];
                    abs_json["totalValues"] = totalValues[i][j];
                    abs_json["absTotalValues"] = absTotalValues[i][j];
                    json_arr_abs.push_back(abs_json);
                }
            }

            string json_str_abs = json_arr_abs.dump();
            json_str_abs = escapeSingleQuote(json_str_abs);

            html_file << R"(
         <script>
         try {
            var jsonDataAbs = ')" +
                         json_str_abs + R"(';  // insert your JSON string here
            var dataAbs = JSON.parse(jsonDataAbs);

            var xAbs = [];
            var yAbs = [];
            var zAbs = [];
            var textAbs = [];

            for (var i = 0; i < dataAbs.length; i++) {
               xAbs.push(dataAbs[i].x *)" << dims[0] / vis_dims[0] << R"( );
               yAbs.push(dataAbs[i].y *)" << dims[1] / vis_dims[1] << R"( );
               zAbs.push(dataAbs[i].absTotalValues);
               textAbs.push(
                         "x: " + dataAbs[i].x *)" << dims[0] / vis_dims[0] << R"( +
                         "<br>y: " + dataAbs[i].y *)" << dims[1] / vis_dims[1] << R"( +
                         "<br>NonZero: " + dataAbs[i].sum +
                         "<br>Total Values: " + dataAbs[i].totalValues +
                         "<br>Abs Total Values: " + dataAbs[i].absTotalValues
                         );
            }

            var traceAbs = {
            x: xAbs,
            y: yAbs,
            z: zAbs,
            type: 'heatmap',
            colorscale: [[0, 'rgb(247, 251, 255)'], [0.1, 'rgb(222, 235, 247)'], [0.2, 'rgb(198, 219, 239)'], [0.3, 'rgb(158, 202, 225)'], [0.4, 'rgb(107, 174, 214)'], [0.5, 'rgb(66, 146, 198)'], [0.6, 'rgb(33, 113, 181)'], [0.7, 'rgb(8, 81, 156)'], [0.8, 'rgb(8, 48, 107)'], [1, 'rgb(3, 19, 43)']], // Blues
            showscale: )" +
                         showColorBar + R"(,
            text: textAbs,
            hoverinfo: 'text'
            };

            var layoutAbs = {
            autosize: false,
            width: )" + width +
                         R"(,
            height: 250,
            yaxis: {
               autorange: 'reversed' // This is for adjusting the main diagonal correctly
            },
            xaxis: {
               side: 'bottom'
            },
            margin: {
            l: 50,
            r: 0,
            b: 50,
            t: 50
            }
            };

            var dataAbsPlot = [traceAbs];

            Plotly.newPlot('absDiv)" +
                         to_string(n) + "_" + to_string(perspective) + R"(', dataAbsPlot, layoutAbs);
            } catch (err) {
               console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }

        html_file << "<div id='histogramAbs" << n
                  << "' style='width: 250px; height: 250px;'></div>\n"; // New div for histogram of absTotalValues
        html_file << R"(
      <script>
      try {
         var histogramAbsData = ')" +
                     histogramAbsStr + R"(';  // insert your histogram JSON string here
         var dataAbs = JSON.parse(histogramAbsData);

         var traceAbs = {
         x: dataAbs.x,
         y: dataAbs.y,
         type: 'bar',
         marker: {
         color: 'blue',  // Different color for absTotalValues
         },
         opacity: 0.7,
         };

         var layoutAbs = {
         title: 'Absolute Value Dist',
         xaxis: {
         title: 'Bin',
         type: 'category',
         autorange: false,
         range: [0, dataAbs.x.length],
         showticklabels: false
         },
         yaxis: {
         type: 'log',
         dtick: Math.log2()
         },
         autosize: false,
         width: 250,
          height: 250,
          margin: {
          l: 50,
          r: 0,
          b: 50,
          t: 50
          }
         };

         var dataAbsPlot = [traceAbs];
         Plotly.newPlot('histogramAbs)" +
                     to_string(n) + R"(', dataAbsPlot, layoutAbs);
      } catch (err) {
         console.log('Error parsing JSON or plotting data: ' + err.message);
      }
      </script>
      )";
        html_file << "</div>\n"; // Close new heatmap container for absolute values

        html_file << "</div>\n"; // close heatmap container

        // the div for the calculations panel
        html_file << R"(
      <div style='
      border: 2px solid black;
      border-radius: 15px;
      box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.3);
      margin: 20px;
      margin-right: 30px;
      padding: 20px;
      background-color: rgba(255, 255, 255, 0.9);
      transition: all 0.3s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
      overflow-x: auto; //adjustments to make it visible anytime.
      box-sizing: border-box;
      width: 100%;      // Container will take full width available
      max-width: 700px;
      min-width: 330px;
      height: 100%;
      max-height: 700px;
      '
      onmouseover="this.style.boxShadow='8px 8px 15px rgba(0, 0, 0, 0.3)';"
      onmouseout="this.style.boxShadow='5px 5px 10px rgba(0, 0, 0, 0.3)';">)"
                      << stat_to_html_table(*(orderings[n]), stats[n])
                      << R"(</div>  
      )";
            html_file << "</div>\n"; // Close plotsContainer
            html_file << "</div>\n"; // Close orderDiv
        }


    html_file << R"(
    <div style='text-align: center; padding: 10px;'>
    <img src=')" + LOGO_PATH +
                 R"(' alt='Logo' style='width: 200px;'>
    <p> © 2024 SparCity</p>
    </div>
    </body>
    </html>
    )";
    html_file.close();
    logger->makeSilentLog("File " + filename + ".html" + " is generated", omp_get_wtime() - start_time);
   
    // Cleanup
    for (int n = 0; n < norder; ++n)
    {
        for (int i = 0; i < vis_dims[0]; ++i)
        {
            for (int j = 0; j < vis_dims[1]; ++j)
            {
                delete[] tensorLists[n][i][j];
            }
            delete[] tensorLists[n][i];
        }
        delete[] tensorLists[n];
    }
    delete[] tensorLists;
    delete[] vis_dims;
}

void visualizeTensors(TensorOrdering** orderings, int norder) {
    double start_time = omp_get_wtime();

    logger->makeSilentLog( "visualizeTensors is started for " + orderings[0]->getOrderingName());

    TensorBin ****tensorLists = new TensorBin ***[norder];
    TStatistic stats[norder];
    vType dims[norder][3];
    vType vis_dims[norder][3];
    std::string filename = orderings[0]->getOrderingName();
    
#pragma omp parallel 
{    
    #pragma omp for schedule(dynamic, 1)
    for(int n = 0; n < norder; n++) 
    {
        const SparseTensorCOO& tensor = dynamic_cast<const SparseTensorCOO&>(orderings[n]->getTensor());
        const std::vector<vType>& active_modes = orderings[n]->getActiveModes();

        const int tensor_order = tensor.getOrder();
        const vType* full_dims = tensor.getDims();
        const vType* nonzeros = tensor.getStorage();
        const valType* values = tensor.getValues();
        const vType nnzCount = tensor.getNNZ();

        for(int i = 0; i < 3; i++) {
            dims[n][i] = full_dims[active_modes[i]];
        }

        vType scaleFactor = std::max(std::max(dims[n][0], dims[n][1]), dims[n][2]) / MAX_DIM;
        scaleFactor = scaleFactor == 0 ? 1: scaleFactor;
        for(int i = 0; i < 3; i++) {
            vis_dims[n][i] = dims[n][i] / scaleFactor;
            if (vis_dims[n][i] < 8) vis_dims[n][i] = 8;
            if(dims[n][i] < vis_dims[n][i]) vis_dims[n][i] = dims[n][i];
        }
    
        tensorLists[n] = new TensorBin **[vis_dims[n][0]];
        for (int i = 0; i < vis_dims[n][0]; ++i) {
            tensorLists[n][i] = new TensorBin *[vis_dims[n][1]];
            for (int j = 0; j < vis_dims[n][1]; ++j) {
                tensorLists[n][i][j] = new TensorBin[vis_dims[n][2]];
            }
        }
        
        std::unordered_map<vpair, vType, pair_hash> fibers[3];
        std::vector<vType> fiberMins[3]; for(int i = 0; i < 3; i++) fiberMins[i].reserve(nnzCount/3);
        std::vector<vType> fiberMaxs[3]; for(int i = 0; i < 3; i++) fiberMaxs[i].reserve(nnzCount/3);
        std::vector<vType> fiberNNZs[3]; for(int i = 0; i < 3; i++) fiberNNZs[i].reserve(nnzCount/3);

        std::unordered_map<vpair, vType>::iterator iter; 
        /*#pragma omp critical 
        {
            std::cout << n << ": DIMS " <<  dims[n][0] << " " <<  dims[n][1] << " " << dims[n][2] << std::endl;
            std::cout << n << ": VDIMS " <<  vis_dims[n][0] << " " <<  vis_dims[n][1] << " " << vis_dims[n][2] << std::endl;
        }*/

        double start_time = omp_get_wtime();
        // Iterate over the nonzeros
        for (vType i = 0; i < nnzCount; i++) {
            vType x = nonzeros[(i * tensor_order) + active_modes[0]]; 
            vType y = nonzeros[(i * tensor_order) + active_modes[1]]; 
            vType z = nonzeros[(i * tensor_order) + active_modes[2]]; 
            valType val = values[i];
            
            vType vec_locs[3];
            for(int d = 0; d < 3; d++) {
                vpair pair; //x,y - x,z - y,z
                if(d == 2) pair.first = y; else pair.first = x;
                if(d == 0) pair.second = y; else pair.second = z; 

                std::unordered_map<vpair, vType>::iterator iter = fibers[d].find(pair);
                if(iter == fibers[d].end()) {
                    vec_locs[d] = fibers[d][pair] = fiberMins[d].size();

                    fiberMins[d].push_back(dims[n][2 - d]); //z,y,x (the missing)
                    fiberMaxs[d].push_back(0);

                    fiberNNZs[d].push_back(1);
                } else {
                    vec_locs[d] = iter->second;
                    fiberNNZs[d][iter->second]++;
                }
            }

            vType ordered_nnz[3];
            vType binIDs[3];

            vType** order = orderings[n]->getOrderedDimensions();

            ordered_nnz[0] = order[active_modes[0]][x];
            ordered_nnz[1] = order[active_modes[1]][y];
            ordered_nnz[2] = order[active_modes[2]][z];
            if(ordered_nnz[0] >= dims[n][0] || ordered_nnz[1] >= dims[n][1] || ordered_nnz[2] >= dims[n][2]) {
                std::cerr << orderings[n]->getOrderingName() << ": Error - Ordered "    << ordered_nnz[0] << " " << dims[n][0] << " | "  
                                                                                        << ordered_nnz[1] << " " << dims[n][1] << " | "  
                                                                                        << ordered_nnz[2] << " " << dims[n][2] << std::endl;
                throw std::runtime_error("b");
            }
            for(int d = 0; d < 3; d++) {
                binIDs[d] = calculateBin(ordered_nnz[d], dims[n][d], vis_dims[n][d]);
                if (binIDs[d] < 0 || binIDs[d] >= vis_dims[n][d]) {
                    std::cerr << "Unexpected bin index value. bin: " << d << ", " <<  binIDs[d] << std::endl;
                }
            }

            TensorBin& tbin = tensorLists[n][binIDs[0]][binIDs[1]][binIDs[2]];
            tbin.nonzeroCount++;
            tbin.totalValues += val;
            tbin.absTotalValues += abs(val);

            for(int d = 0; d < 3; d++) {
                if(fiberMins[d][vec_locs[d]] > ordered_nnz[2-d]) {
                    fiberMins[d][vec_locs[d]] = ordered_nnz[2-d];
                }
                if(fiberMaxs[d][vec_locs[d]] < ordered_nnz[2-d]) {
                    fiberMaxs[d][vec_locs[d]] = ordered_nnz[2-d];
                }
            }
        }
        double end_time = omp_get_wtime();
        //logger->makeSilentLog(orderings[n]->getOrderingName() + ": " + orderings[n]->getTensor().getName() + " - nonzeros are processed in ", end_time - start_time);

        for(int d = 0; d < 3; d++) {
            for(vType v = 0; v < fiberMins[d].size(); v++) {
                vType diff = fiberMaxs[d][v] - fiberMins[d][v] + 1;
                if(diff > 1) { //checks if there is more than one nonzero in this fiber
                    stats[n].fiberCounts[d]++;
                    stats[n].avgSpan[d] += diff;
                    stats[n].maxSpan[d] = std::max((int)stats[n].maxSpan[d], (int)diff);
                    stats[n].normSpan[d] += ((double)diff) / fiberNNZs[d][v];
                } else if(diff == 0) {
                    stats[n].singleNNZfiberCounts[d]++;
                }
            }
         }    

        for(int d = 0; d < 3; d++) {
            if(stats[n].fiberCounts[d] == 0) {
                stats[n].avgSpan[d] = stats[n].normSpan[d] = stats[n].maxSpan[d] = 0;
            } else {
                stats[n].avgSpan[d] /= stats[n].fiberCounts[d]; 
                stats[n].normSpan[d] /= stats[n].fiberCounts[d]; 
            }
        }

        stats[n].tensorName = orderings[n]->getTensor().getName();
        stats[n].orderingName = orderings[n]->getOrderingName();
        logger->logTensorProcessing(TENSOR_VISUALIZATION_FILES_DIR + filename + ".html", stats[n], end_time - start_time);
    }
}

    std::string filePath;

#ifdef TEST
    filePath = SparseVizTest::getSparseVizTester()->getCurrentDirectory() + filename + ".html";
#else
    filePath = TENSOR_VISUALIZATION_FILES_DIR + filename + ".html";
#endif

    std::ofstream html_file(filePath);
    html_file << R"(
   <!DOCTYPE html>
   <html>
   <head>
   <meta charset="UTF-8">
   <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
   <style>
   body {
      font-family: 'Orbitron', sans-serif;
      font-size:70%;
   }
   .header {
   display: flex;
      justify-content: space-between;
      align-items: center;
   width: 100%;
   }
   .filename {
      font-size: 18px;
   }
   .title {
      text-align: left;
   }
   .title-main {
   margin: 0;
   }
   .title-sub {
   margin: 0;
   }
   margin: 0;
   padding: 0;
      box-sizing: border-box;
   }
   .hoverlayer
   {
   z-index: 1000 !important;
   opacity: 1 !important;
   visibility: visible !important;
   }
   </style>
   <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
   <link rel="icon" href="favicon.ico" type="image/x-icon">
   </head>
   <body>
   <div class="header">
   <div class="title">
   <h1 class="title-main">SparseViz Tensor</h1>
   <h2 class="title-sub">Visualization</h2>
   </div>
   
   
    <div class="title">
   <h2>Ordering Name: )" + filename + R"(</h2><hr>)";
    //html_file << "<h3 class=\"title-sub\">Dimensions: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "</h3>";
    //html_file << "<h3 class=\"title-sub\">Nonzeros: " << nnzCount << "</h3>";

    html_file << "</div>\n"; // Close right header div
    html_file << "</div>\n"; // Close header div

    html_file << "<div id=\"aspect\">\n";
    html_file << "<button onClick=\"choose('data')\">Actual Tensor Sizes</button>\n";
    html_file << "<button onClick=\"choose('cube')\">Cube Tensors</button>\n";
    html_file << "<script>\n";
    html_file << "function choose(choice) {\n";
    for(int n = 0; n < norder; n++) {
        html_file << "choose_" << n << "(choice);\n";
    }
    html_file << "}\n";
    html_file << "</script>\n";
    html_file << "</div>\n";

    for (int n = 0; n < norder; n++) {
        vector<vector<int>> topDownSum(vis_dims[n][0], vector<int>(vis_dims[n][1], 0));
        vector<vector<int>> sideSum(vis_dims[n][1], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> tubeSum(vis_dims[n][0], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> topDownTotalValues(vis_dims[n][0], vector<int>(vis_dims[n][1], 0));
        vector<vector<int>> sideTotalValues(vis_dims[n][1], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> tubeTotalValues(vis_dims[n][0], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> topDownAbsTotalValues(vis_dims[n][0], vector<int>(vis_dims[n][1], 0));
        vector<vector<int>> sideAbsTotalValues(vis_dims[n][1], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> tubeAbsTotalValues(vis_dims[n][0], vector<int>(vis_dims[n][2], 0)); // for heatmaps
        nlohmann::json json_array;                                                        // for 3D Scatter plot
        nlohmann::json x_bins_json, y_nonzero_json;                                       // for barPlots
        nlohmann::json x_bins_abs_json, y_absTotalValues_json;                            // for barPlots
        vector<int> nonZeros;

        stats[n].no_bins =  vis_dims[n][2] * vis_dims[n][1] * vis_dims[n][0];
        stats[n].no_empty_bins = 0;
        for (int x = 0; x < vis_dims[n][0]; x++) {
            for (int y = 0; y < vis_dims[n][1]; y++) {
                for (int z = 0; z < vis_dims[n][2]; z++) {
                    nlohmann::json bin_json;
                    bin_json["binX"] = x;
                    bin_json["binY"] = y;
                    bin_json["binZ"] = z;
                    bin_json["nonzeroCount"] = tensorLists[n][x][y][z].nonzeroCount;
                    bin_json["totalValues"] = tensorLists[n][x][y][z].totalValues;
                    bin_json["absTotalValues"] = tensorLists[n][x][y][z].absTotalValues;
                    json_array.push_back(bin_json);
                    int nonZeroCount = tensorLists[n][x][y][z].nonzeroCount;

                    if (nonZeroCount == 0) {
                        stats[n].no_empty_bins++;
                        nonZeroCount = 1;
                    }
                    stats[n].geo_mean_nnz += log(nonZeroCount);

                    topDownSum[x][y] += tensorLists[n][x][y][z].nonzeroCount;
                    sideSum[y][z] += tensorLists[n][x][y][z].nonzeroCount;
                    tubeSum[x][z] += tensorLists[n][x][y][z].nonzeroCount;

                    topDownTotalValues[x][y] += tensorLists[n][x][y][z].totalValues;
                    sideTotalValues[y][z] += tensorLists[n][x][y][z].totalValues;
                    tubeTotalValues[x][z] += tensorLists[n][x][y][z].totalValues;

                    topDownAbsTotalValues[x][y] += tensorLists[n][x][y][z].absTotalValues;
                    sideAbsTotalValues[y][z] += tensorLists[n][x][y][z].absTotalValues;
                    tubeAbsTotalValues[x][z] += tensorLists[n][x][y][z].absTotalValues;

                    string coord_str = "(" + to_string(x) + ", " + to_string(y) + ", " + to_string(z) + ")";
                    x_bins_json.push_back(coord_str);                                        // insert bin number
                    y_nonzero_json.push_back(tensorLists[n][x][y][z].nonzeroCount);          // insert nonzeroCount value
                    x_bins_abs_json.push_back(coord_str);                                    // insert bin number
                    y_absTotalValues_json.push_back(tensorLists[n][x][y][z].absTotalValues); // insert absTotalValues value
                    if (tensorLists[n][x][y][z].nonzeroCount != 0) {
                        nonZeros.push_back(tensorLists[n][x][y][z].nonzeroCount);
                    }
                }
            }
        }

        std::sort(nonZeros.begin(), nonZeros.end());
        if (nonZeros.size() % 2 == 0) { // Even number of elements
            stats[n].median_nnz = (nonZeros[nonZeros.size() / 2 - 1] + nonZeros[nonZeros.size() / 2]) / 2.0;
        } else { // Odd number of elements
            stats[n].median_nnz = nonZeros[nonZeros.size() / 2];
        }
        stats[n].geo_mean_nnz = stats[n].geo_mean_nnz / (stats[n].no_bins - stats[n].no_empty_bins);
        stats[n].geo_mean_nnz = exp(stats[n].geo_mean_nnz);
        stats[n].mean_nnz = (double(orderings[n]->getTensor().getNNZ())) / (stats[n].no_bins - stats[n].no_empty_bins);
       
        vector<vector<int>> *sums[3] = {&topDownSum, &sideSum, &tubeSum};
        vector<vector<int>> *totalValuesArrays[] = {&topDownTotalValues, &sideTotalValues, &tubeTotalValues};
        vector<vector<int>> *absTotalValuesArrays[] = {&topDownAbsTotalValues, &sideAbsTotalValues, &tubeAbsTotalValues};

        // converting the nlohmann::json object to a string
        string json_str = json_array.dump();
        json_str = escapeSingleQuote(json_str);

      html_file << "<div id='orderDiv" << n
                  << "' style='display: flex; flex-direction: row; align-items: center; "
                     "justify-content: space-around; margin-bottom: 5px;'>\n";
        html_file << "<div style='writing-mode: vertical-rl; display:flex; justify-content:center; text-align: center; align-items:center; transform: "
                     "rotate(180deg); margin-left:20px; margin-right: 10px;'>\n";
        html_file << "<h2>Tensor: " << orderings[n]->getTensor().getName() << " - mods(" << std::to_string(orderings[n]->getActiveModes()[0]) << ", " << std::to_string(orderings[n]->getActiveModes()[1]) << ", " << std::to_string(orderings[n]->getActiveModes()[2]) << ")"  
         << "<br>\n(dims: " << std::to_string(dims[n][0]) << " x  " << std::to_string(dims[n][1]) << " x  " << std::to_string(dims[n][2]) << ")" << "<br>\n(nnz: " << std::to_string(orderings[n]->getTensor().getNNZ()) << ")</h3>\n";
        html_file << "</div>\n";
        html_file << "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: center; width: 100%;'>\n";

        // 3D scatter plot
        html_file << "<div id='myDiv" << n << "' style='width: 600px; height: 700px;'></div>\n";
        html_file << "<div style='display: flex; flex-direction: column; justify-content: space-between; width: 100%;'>\n"; // Change to column

        // js code for 3D scatter plot
        html_file << "<script>\n";
        html_file << "choose_" << n << "('data');\n";
        html_file << "function choose_" << n << "(choice)\n{\n";
    html_file << R"(
      try {
         var jsonData = ')" +
                     json_str + R"(';  // insert your JSON string here
         var data = JSON.parse(jsonData);

         // Extracting x, y, z coordinates and other data
         var x = data.map(d => d.binX);
         var y = data.map(d => d.binY);
         var z = data.map(d => d.binZ);
         var nonzeroCount = data.map(d => d.nonzeroCount);
         var totalValues = data.map(d => d.totalValues);
         var absTotalValues = data.map(d => d.absTotalValues);
         // Find the maximum nonZeroCount to normalize the counts
         var maxCount = Math.max(...data.map(d => d.nonzeroCount));
         // Generating hover text
         var hoverText = data.map(d => `Non-zero values: ${d.nonzeroCount}<br>Total value: ${d.totalValues}<br>Absolute total value: ${d.absTotalValues}`);

         // Creating the 3D scatter plot
         var xNonZero = [], yNonZero = [], zNonZero = [], countsNonZero = [], hoverTextNonZero = [];
         var xZero = [], yZero = [], zZero = [], hoverTextZero = [];

         for (var i = 0; i < nonzeroCount.length; i++) {
            if (nonzeroCount[i] == 0) {
               xZero.push(x[i] *)" << dims[n][0] / vis_dims[n][0] << R"( );
               yZero.push(y[i] *)" << dims[n][1] / vis_dims[n][1] << R"( );
               zZero.push(z[i] *)" << dims[n][2] / vis_dims[n][2] << R"( );
               hoverTextZero.push(hoverText[i]);
            } else {
               xNonZero.push(x[i] *)" << dims[n][0] / vis_dims[n][0] << R"( );
               yNonZero.push(y[i] *)" << dims[n][1] / vis_dims[n][1] << R"( );
               zNonZero.push(z[i] *)" << dims[n][2] / vis_dims[n][2] << R"( );
               countsNonZero.push(nonzeroCount[i]);
               hoverTextNonZero.push(hoverText[i]);
            }
         }

         // Create two separate traces
         var traceNonZero = {
         x: xNonZero,
         y: yNonZero,
         z: zNonZero,
         mode: 'markers',
         type: 'scatter3d',
         marker: {
         size: 8,
         color: countsNonZero,
         colorscale: 'Reds',
         opacity: 0.9
         },
         text: hoverTextNonZero,
         hoverinfo: 'text'
         };

         var traceZero = {
         x: xZero,
         y: yZero,
         z: zZero,
         mode: 'markers',
         type: 'scatter3d',
         marker: {
         size: 8,
         color: 'rgba(255,0,0,0)',  // Transparent red
         opacity: 0
         },
         text: hoverTextZero,
         hoverinfo: 'text'
         };

         var layout = {
         autosize: true,
         scene: {
            aspectmode: choice
         }, 
         width: 650,
         yaxis: {
            autorange: 'reversed' // This is for adjusting the main diagonal correctly
         },
         xaxis: {
            side: 'top'
         },
         height: 650,
         margin: {
         l: 10,
         r: 80,
         b: 10,
         t: 10
         }
         };

         var data = [traceNonZero];

         Plotly.newPlot('myDiv)" +
                     to_string(n) + R"(', data, layout);
      } catch (err) {
         console.log('Error parsing JSON or plotting data: ' + err.message);
      }
      }
      </script>
      )";

        html_file
                << "<div style='display: flex; flex-direction: row; justify-content: flex-start; width: 100%;'>\n"; // heatmap container

        for (int perspective = 0; perspective < 3; perspective++) { // 3 perspectives fibers rows tubes
            nlohmann::json json_arr;
            auto &sum = *sums[perspective];
            auto &totalValues = *totalValuesArrays[perspective];
            auto &absTotalValues = *absTotalValuesArrays[perspective];
            string titles[] = {"'Dimensions: X-Y'", "'Dimensions: Y-Z'", "'Dimensions: X-Z'"};
            string showColorBar = "false", width = "230";
            if(perspective == 2) {
                showColorBar = "true";
                width = "300";
                html_file << "<div id='myDiv" << n << "_" << perspective << "' style='width: 300; height: 250;'></div>\n";
            } else {
                html_file << "<div id='myDiv" << n << "_" << perspective << "' style='width: 230; height: 250;'></div>\n";
            }

            for (int i = 0; i < sum.size(); i++) {
                for (int j = 0; j < sum[i].size(); j++) {
                    nlohmann::json sum_json;
                    sum_json["x"] = i;
                    sum_json["y"] = j;
                    sum_json["sum"] = sum[i][j];
                    sum_json["totalValues"] = totalValues[i][j];
                    sum_json["absTotalValues"] = absTotalValues[i][j];
                    json_arr.push_back(sum_json);
                }
            }

            string json_str2 = json_arr.dump();
            json_str2 = escapeSingleQuote(json_str2);

            html_file << R"(
         <script>
         try {
            var jsonData2 = ')" + json_str2 + R"(';  // insert your JSON string here
            var data2 = JSON.parse(jsonData2);

            var x2 = [];
            var y2 = [];
            var z2 = [];
            var text = [];)";

            if(perspective == 0) {
                html_file << R"(  
                //custom hovertext creation
                for (var i = 0; i < data2.length; i++) {
                x2.push(data2[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( );
                y2.push(data2[i].y *)" << dims[n][1] / vis_dims[n][1] << R"( );
                z2.push(data2[i].sum);
                text.push(
                            "x: " + data2[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( +
                            "<br>y: " + data2[i].y *)" << dims[n][1] / vis_dims[n][1] << R"( +
                            "<br>NonZero: " + data2[i].sum +
                            "<br>Total Values: " + data2[i].totalValues +
                            "<br>Abs Total Values: " + data2[i].absTotalValues
                            );
                })";
            } else  if(perspective == 1) {
                html_file << R"(  
                //custom hovertext creation
                for (var i = 0; i < data2.length; i++) {
                x2.push(data2[i].x *)" << dims[n][1] / vis_dims[n][2] << R"( );
                y2.push(data2[i].y *)" << dims[n][2] / vis_dims[n][2] << R"( );
                z2.push(data2[i].sum);
                text.push(
                            "y: " + data2[i].x *)" << dims[n][1] / vis_dims[n][1] << R"( +
                            "<br>z: " + data2[i].y *)" << dims[n][2] / vis_dims[n][2] << R"( +
                            "<br>NonZero: " + data2[i].sum +
                            "<br>Total Values: " + data2[i].totalValues +
                            "<br>Abs Total Values: " + data2[i].absTotalValues
                            );
                })";
            } else  if(perspective == 2) {
                html_file << R"(  
                //custom hovertext creation
                for (var i = 0; i < data2.length; i++) {
                x2.push(data2[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( );
                y2.push(data2[i].y *)" << dims[n][2] / vis_dims[n][2] << R"( );
                z2.push(data2[i].sum);
                text.push(
                            "x: " + data2[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( +
                            "<br>z: " + data2[i].y *)" << dims[n][2] / vis_dims[n][2] << R"( +
                            "<br>NonZero: " + data2[i].sum +
                            "<br>Total Values: " + data2[i].totalValues +
                            "<br>Abs Total Values: " + data2[i].absTotalValues
                            );
                })";
            }



            html_file << R"( 
            // Creating the heatmap
            var trace2 = {
            x: x2,
            y: y2,
            z: z2,
            type: 'heatmap',
            colorscale: 'Reds',
            showscale: )" +
                         showColorBar + R"(,// set for only seen in the last heatmap
            text: text,
            hoverinfo: 'text'  // Only show the custom hover text
            };

            var layout2 = {
            title: )" + titles[perspective] +
                         R"(,
            autosize: false,
            width: )" + width +
                         R"(,// set different for the last heatmap since it has colorbar
            height: 250,
            yaxis: {
               autorange: 'reversed' // This is for adjusting the main diagonal correctly
            },
            xaxis: {
               side: 'bottom'
            },
            margin: {
            l: 50,
            r: 0,
            b: 50,
            t: 50
            }
            };

            var data2 = [trace2];

            Plotly.newPlot('myDiv)" + to_string(n) + "_" + to_string(perspective) + R"(', data2, layout2);
            } catch (err) {
               console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }

        // Sorting together to create the visualization as intended.
        vector<pair<string, int>> sorted_bins;
        vector<pair<string, double>> sorted_abs_bins; // Note: Using double for absTotalValues

        for (size_t i = 0; i < x_bins_json.size(); i++) {
            sorted_bins.push_back(make_pair(x_bins_json[i].get<string>(), y_nonzero_json[i]));
        }

        for (size_t i = 0; i < x_bins_abs_json.size(); i++) {
            sorted_abs_bins.push_back(make_pair(x_bins_abs_json[i].get<string>(), y_absTotalValues_json[i]));
        }
        // sort in descending order of y values (nonzero counts)
        std::sort(sorted_bins.begin(), sorted_bins.end(),
             [](const std::pair<string, int> &a, const std::pair<string, int> &b) {
                 return a.second > b.second; // change this to a.second < b.second; for ascending order
             });
        // sort in descending order of y values (absTotalValues)
        std::sort(sorted_abs_bins.begin(), sorted_abs_bins.end(),
             [](const std::pair<string, double> &a, const std::pair<string, double> &b) {
                 return a.second > b.second; // for descending order
             });
        // create new sorted json arrays
        nlohmann::json sorted_x_bins_json, sorted_y_nonzero_json;
        nlohmann::json sorted_x_bins_abs_json, sorted_y_absTotalValues_json;

        for (const auto &bin: sorted_bins) {
            sorted_x_bins_json.push_back(bin.first);
            sorted_y_nonzero_json.push_back(bin.second);
        }
        // create new sorted json arrays
        for (const auto &bin: sorted_abs_bins) {
            sorted_x_bins_abs_json.push_back(bin.first);
            sorted_y_absTotalValues_json.push_back(bin.second);
        }
        // creating a json object with two sorted arrays
        nlohmann::json sorted_data_hist;
        nlohmann::json sorted_data_abs_hist;
        sorted_data_abs_hist["x"] = sorted_x_bins_abs_json;
        sorted_data_abs_hist["y"] = sorted_y_absTotalValues_json;
        sorted_data_hist["x"] = sorted_x_bins_json;
        sorted_data_hist["y"] = sorted_y_nonzero_json;
        // using sorted_data_hist to create the plots.

        string histogramStr = sorted_data_hist.dump();
        string histogramAbsStr = sorted_data_abs_hist.dump(); // named histograms but barPlots. note that this data is currently not in use but its duplicate is used.

        html_file << "<div id='histogram" << n
                  << "' style='width: 250px; height: 250px;'></div>\n"; // New div for histogram
        html_file << R"(
       <script>
       try {
          var histogramData = ')" +
                     histogramStr + R"(';  // insert your histogram JSON string here
          var data = JSON.parse(histogramData);

          var trace3 = {
          x: data.x,
          y: data.y,
          type: 'bar',
          marker: {
          color: 'red',
          },
          opacity: 0.7,
          };

          var layout3 = {
          title: 'Non-zero Counts Dist',
          xaxis: {
          title: 'Bin',
          type: 'category',  // treat x-axis labels as categories
          autorange: false,  // disable auto range
          range: [0, data.x.length],
          showticklabels: false
          },
          yaxis: {
          type: 'log',
          dtick: Math.log2()  // interval between ticks
          },
          autosize: false,
          width: 250,
          height: 250,
          margin: {
          l: 50,
          r: 0,
          b: 50,
          t: 50
          }
          };

          var data3 = [trace3];
          Plotly.newPlot('histogram)" + to_string(n) + R"(', data3, layout3);
       } catch (err) {
          console.log('Error parsing JSON or plotting data: ' + err.message);
       }
       </script>
       )";

        html_file << "</div>\n"; // Close container for nnz heatmaps

        // New section for three new heatmaps based on absolute total values.
        html_file << "<div style='display: flex; flex-direction: row; justify-content: flex-start; width: 100%;'>\n"; // New heatmap container for absolute values

        for (int perspective = 0; perspective < 3; perspective++) {
            nlohmann::json json_arr_abs;
            auto &sum = *sums[perspective];
            auto &totalValues = *totalValuesArrays[perspective];
            auto &absTotalValues = *absTotalValuesArrays[perspective];
            string titles[] = {"'Dimensions: X-Y'", "'Dimensions: Y-Z'", "'Dimensions: X-Z'"};
            string showColorBar = "false", width = "230";
            if(perspective == 2) {
                showColorBar = "true";
                width = "300";
                html_file << "<div id='absDiv" << n << "_" << perspective << "' style='width: 300; height: 250;'></div>\n";
            } else {
                html_file << "<div id='absDiv" << n << "_" << perspective << "' style='width: 230; height: 250;'></div>\n";
            }

            for (int i = 0; i < absTotalValues.size(); i++) {
                for (int j = 0; j < absTotalValues[i].size(); j++) {
                    nlohmann::json abs_json;
                    abs_json["x"] = i;
                    abs_json["y"] = j;
                    abs_json["sum"] = sum[i][j];
                    abs_json["totalValues"] = totalValues[i][j];
                    abs_json["absTotalValues"] = absTotalValues[i][j];
                    json_arr_abs.push_back(abs_json);
                }
            }

            string json_str_abs = json_arr_abs.dump();
            json_str_abs = escapeSingleQuote(json_str_abs);

            html_file << R"(
         <script>
         try {
            var jsonDataAbs = ')" +
                         json_str_abs + R"(';  // insert your JSON string here
            var dataAbs = JSON.parse(jsonDataAbs);

            var xAbs = [];
            var yAbs = [];
            var zAbs = [];
            var textAbs = [];

            for (var i = 0; i < dataAbs.length; i++) {
               xAbs.push(dataAbs[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( );
               yAbs.push(dataAbs[i].y *)" << dims[n][1] / vis_dims[n][1] << R"( );
               zAbs.push(dataAbs[i].absTotalValues);
               textAbs.push(
                         "x: " + dataAbs[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( +
                         "<br>y: " + dataAbs[i].y *)" << dims[n][1] / vis_dims[n][1] << R"( +
                         "<br>NonZero: " + dataAbs[i].sum +
                         "<br>Total Values: " + dataAbs[i].totalValues +
                         "<br>Abs Total Values: " + dataAbs[i].absTotalValues
                         );
            }

            var traceAbs = {
            x: xAbs,
            y: yAbs,
            z: zAbs,
            type: 'heatmap',
            colorscale: [[0, 'rgb(247, 251, 255)'], [0.1, 'rgb(222, 235, 247)'], [0.2, 'rgb(198, 219, 239)'], [0.3, 'rgb(158, 202, 225)'], [0.4, 'rgb(107, 174, 214)'], [0.5, 'rgb(66, 146, 198)'], [0.6, 'rgb(33, 113, 181)'], [0.7, 'rgb(8, 81, 156)'], [0.8, 'rgb(8, 48, 107)'], [1, 'rgb(3, 19, 43)']], // Blues
            showscale: )" +
                         showColorBar + R"(,
            text: textAbs,
            hoverinfo: 'text'
            };

            var layoutAbs = {
            autosize: false,
            width: )" + width +
                         R"(,
            height: 250,
            yaxis: {
               autorange: 'reversed' // This is for adjusting the main diagonal correctly
            },
            xaxis: {
               side: 'bottom'
            },
            margin: {
            l: 50,
            r: 0,
            b: 50,
            t: 50
            }
            };

            var dataAbsPlot = [traceAbs];

            Plotly.newPlot('absDiv)" + to_string(n) + "_" + to_string(perspective) + R"(', dataAbsPlot, layoutAbs);
            } catch (err) {
               console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }

        html_file << "<div id='histogramAbs" << n << "' style='width: 250px; height: 250px;'></div>\n"; // New div for histogram of absTotalValues
        html_file << R"(
      <script>
      try {
         var histogramAbsData = ')" + histogramAbsStr + R"(';  // insert your histogram JSON string here
         var dataAbs = JSON.parse(histogramAbsData);

         var traceAbs = {
         x: dataAbs.x,
         y: dataAbs.y,
         type: 'bar',
         marker: {
         color: 'blue',  // Different color for absTotalValues
         },
         opacity: 0.7,
         };

         var layoutAbs = {
         title: 'Absolute Value Dist',
         xaxis: {
         title: 'Bin',
         type: 'category',
         autorange: false,
         range: [0, dataAbs.x.length],
         showticklabels: false
         },
         yaxis: {
         type: 'log',
         dtick: Math.log2()
         },
         autosize: false,
         width: 250,
          height: 250,
          margin: {
          l: 50,
          r: 0,
          b: 50,
          t: 50
          }
         };

         var dataAbsPlot = [traceAbs];
         Plotly.newPlot('histogramAbs)" +
                     to_string(n) + R"(', dataAbsPlot, layoutAbs);
      } catch (err) {
         console.log('Error parsing JSON or plotting data: ' + err.message);
      }
      </script>
      )";
        html_file << "</div>\n"; // Close new heatmap container for absolute values

        html_file << "</div>\n"; // close heatmap container

        // the div for the calculations panel
        html_file << R"(
      <div style='
      border: 2px solid black;
      border-radius: 15px;
      box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.3);
      margin: 20px;
      margin-right: 30px;
      padding: 20px;
      background-color: rgba(255, 255, 255, 0.9);
      transition: all 0.3s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
      overflow-x: auto; //adjustments to make it visible anytime.
      box-sizing: border-box;
      width: 100%;      // Container will take full width available
      max-width: 700px;
      min-width: 330px;
      height: 100%;
      max-height: 700px;
      '
      onmouseover="this.style.boxShadow='8px 8px 15px rgba(0, 0, 0, 0.3)';"
      onmouseout="this.style.boxShadow='5px 5px 10px rgba(0, 0, 0, 0.3)';">)"
                      << stat_to_html_table(*(orderings[n]), stats[n])
                      << R"(</div>  
      )";
            html_file << "</div>\n"; // Close plotsContainer
            html_file << "</div>\n"; // Close orderDiv
        }


    html_file << R"(
    <div style='text-align: center; padding: 10px;'>
    <img src=')" + LOGO_PATH +
                 R"(' alt='Logo' style='width: 200px;'>
    <p> © 2024 SparCity</p>
    </div>
    </body>
    </html>
    )";
    html_file.close();
    logger->makeSilentLog("File " + filename + ".html" + " is generated", omp_get_wtime() - start_time);
   
    // Cleanup
    for (int n = 0; n < norder; ++n)
    {
        for (int i = 0; i < vis_dims[n][0]; ++i)
        {
            for (int j = 0; j < vis_dims[n][1]; ++j)
            {
                delete[] tensorLists[n][i][j];
            }
            delete[] tensorLists[n][i];
        }
        delete[] tensorLists[n];
    }
    delete[] tensorLists;
}

void visualizeFullSparseTensor(TensorOrdering* ordering) {
    double start_time = omp_get_wtime();

    logger->makeSilentLog( "visualizeTensors is started for " + ordering->getOrderingName() + " " + ordering->getTensor().getName());
    
    int tensor_rank = ordering->getTensor().getOrder();

    int norder = 0;
    std::vector<std::vector<vType> > active_modes_s;
    for(unsigned int i = 0; i < tensor_rank; i++) {
        for(unsigned int j = i+1; j < tensor_rank; j++) {
            for(unsigned int k = j+1; k < tensor_rank; k++) {
                std::vector<vType> active_modes = {i, j, k};
                active_modes_s.push_back(active_modes);
                norder++;
            }
        }
    }

    TensorBin ****tensorLists = new TensorBin ***[norder];
    TStatistic stats[norder];
    vType dims[norder][3];
    vType vis_dims[norder][3];
    std::string filename = ordering->getOrderingName() + "_" + ordering->getTensor().getName() + ".FULL";

#pragma omp parallel 
{    
    #pragma omp for schedule(dynamic, 1)
    for(int n = 0; n < norder; n++) 
    {
        const SparseTensorCOO& tensor = dynamic_cast<const SparseTensorCOO&>(ordering->getTensor());
        const std::vector<vType>& active_modes = active_modes_s[n];

        const int tensor_order = tensor.getOrder();
        const vType* full_dims = tensor.getDims();
        const vType* nonzeros = tensor.getStorage();
        const valType* values = tensor.getValues();
        const vType nnzCount = tensor.getNNZ();

        for(int i = 0; i < 3; i++) {
            dims[n][i] = full_dims[active_modes[i]];
        }

        vType scaleFactor = std::max(std::max(dims[n][0], dims[n][1]), dims[n][2]) / MAX_DIM;
        scaleFactor = scaleFactor == 0 ? 1: scaleFactor;
        for(int i = 0; i < 3; i++) {
            vis_dims[n][i] = dims[n][i] / scaleFactor;
            if (vis_dims[n][i] < 8) vis_dims[n][i] = 8;
            if(dims[n][i] < vis_dims[n][i]) vis_dims[n][i] = dims[n][i];
        }
    
        tensorLists[n] = new TensorBin **[vis_dims[n][0]];
        for (int i = 0; i < vis_dims[n][0]; ++i) {
            tensorLists[n][i] = new TensorBin *[vis_dims[n][1]];
            for (int j = 0; j < vis_dims[n][1]; ++j) {
                tensorLists[n][i][j] = new TensorBin[vis_dims[n][2]];
            }
        }
        
        std::unordered_map<vpair, vType, pair_hash> fibers[3];
        std::vector<vType> fiberMins[3]; for(int i = 0; i < 3; i++) fiberMins[i].reserve(nnzCount/3);
        std::vector<vType> fiberMaxs[3]; for(int i = 0; i < 3; i++) fiberMaxs[i].reserve(nnzCount/3);
        std::vector<vType> fiberNNZs[3]; for(int i = 0; i < 3; i++) fiberNNZs[i].reserve(nnzCount/3);

        std::unordered_map<vpair, vType>::iterator iter; 
        //#pragma omp critical 
        //{
        //    std::cout << n << ": DIMS " <<  dims[n][0] << " " <<  dims[n][1] << " " << dims[n][2] << std::endl;
        //    std::cout << n << ": VDIMS " <<  vis_dims[n][0] << " " <<  vis_dims[n][1] << " " << vis_dims[n][2] << std::endl;
        //}

        double start_time = omp_get_wtime();
        // Iterate over the nonzeros
        for (vType i = 0; i < nnzCount; i++) {
            vType x = nonzeros[(i * tensor_order) + active_modes[0]]; 
            vType y = nonzeros[(i * tensor_order) + active_modes[1]]; 
            vType z = nonzeros[(i * tensor_order) + active_modes[2]]; 
            valType val = values[i];
            
            vType vec_locs[3];
            for(int d = 0; d < 3; d++) {
                vpair pair; //x,y - x,z - y,z
                if(d == 2) pair.first = y; else pair.first = x;
                if(d == 0) pair.second = y; else pair.second = z; 

                std::unordered_map<vpair, vType>::iterator iter = fibers[d].find(pair);
                if(iter == fibers[d].end()) {
                    vec_locs[d] = fibers[d][pair] = fiberMins[d].size();

                    fiberMins[d].push_back(dims[n][2 - d]); //z,y,x (the missing)
                    fiberMaxs[d].push_back(0);

                    fiberNNZs[d].push_back(1);
                } else {
                    vec_locs[d] = iter->second;
                    fiberNNZs[d][iter->second]++;
                }
            }

            vType ordered_nnz[3];
            vType binIDs[3];

            vType** order = ordering->getOrderedDimensions();

            ordered_nnz[0] = order[active_modes[0]][x];
            ordered_nnz[1] = order[active_modes[1]][y];
            ordered_nnz[2] = order[active_modes[2]][z];
            if(ordered_nnz[0] >= dims[n][0] || ordered_nnz[1] >= dims[n][1] || ordered_nnz[2] >= dims[n][2]) {
                std::cerr << ordering->getOrderingName() << ": Error - Ordered "    << ordered_nnz[0] << " " << dims[n][0] << " | "  
                                                                                    << ordered_nnz[1] << " " << dims[n][1] << " | "  
                                                                                    << ordered_nnz[2] << " " << dims[n][2] << std::endl;
                throw std::runtime_error("c");
            }
            for(int d = 0; d < 3; d++) {
                binIDs[d] = calculateBin(ordered_nnz[d], dims[n][d], vis_dims[n][d]);
                if (binIDs[d] < 0 || binIDs[d] >= vis_dims[n][d]) {
                    std::cerr << "Unexpected bin index value. bin: " << d << ", " <<  binIDs[d] << std::endl;
                }
            }

            TensorBin& tbin = tensorLists[n][binIDs[0]][binIDs[1]][binIDs[2]];
            tbin.nonzeroCount++;
            tbin.totalValues += val;
            tbin.absTotalValues += abs(val);

            for(int d = 0; d < 3; d++) {
                if(fiberMins[d][vec_locs[d]] > ordered_nnz[2-d]) {
                    fiberMins[d][vec_locs[d]] = ordered_nnz[2-d];
                }
                if(fiberMaxs[d][vec_locs[d]] < ordered_nnz[2-d]) {
                    fiberMaxs[d][vec_locs[d]] = ordered_nnz[2-d];
                }
            }
        }
        double end_time = omp_get_wtime();
        //logger->makeSilentLog(ordering->getOrderingName() + ": " + ordering->getTensor().getName() + " - nonzeros are processed in ", end_time - start_time);

        for(int d = 0; d < 3; d++) {
            for(vType v = 0; v < fiberMins[d].size(); v++) {
                vType diff = fiberMaxs[d][v] - fiberMins[d][v] + 1;
                if(diff > 1) { //checks if there is more than one nonzero in this fiber
                    stats[n].fiberCounts[d]++;
                    stats[n].avgSpan[d] += diff;
                    stats[n].maxSpan[d] = std::max((int)stats[n].maxSpan[d], (int)diff);
                    stats[n].normSpan[d] += ((double)diff) / fiberNNZs[d][v];
                } else if(diff == 0) {
                    stats[n].singleNNZfiberCounts[d]++;
                }
            }
         }    

        for(int d = 0; d < 3; d++) {
            if(stats[n].fiberCounts[d] == 0) {
                stats[n].avgSpan[d] = stats[n].normSpan[d] = stats[n].maxSpan[d] = 0;
            } else {
                stats[n].avgSpan[d] /= stats[n].fiberCounts[d]; 
                stats[n].normSpan[d] /= stats[n].fiberCounts[d]; 
            }
        }

        stats[n].tensorName = ordering->getTensor().getName();
        stats[n].orderingName = ordering->getOrderingName();
        logger->logTensorProcessing(TENSOR_VISUALIZATION_FILES_DIR + filename + ".html", stats[n], end_time - start_time);
    }
}

    std::string filePath;

#ifdef TEST
    filePath = SparseVizTest::getSparseVizTester()->getCurrentDirectory() + filename + ".html";
#else
    filePath = TENSOR_VISUALIZATION_FILES_DIR + filename + ".html";
#endif

    std::ofstream html_file(filePath);
    html_file << R"(
   <!DOCTYPE html>
   <html>
   <head>
   <meta charset="UTF-8">
   <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
   <style>
   body {
      font-family: 'Orbitron', sans-serif;
      font-size:70%;
   }
   .header {
   display: flex;
      justify-content: space-between;
      align-items: center;
   width: 100%;
   }
   .filename {
      font-size: 18px;
   }
   .title {
      text-align: left;
   }
   .title-main {
   margin: 0;
   }
   .title-sub {
   margin: 0;
   }
   margin: 0;
   padding: 0;
      box-sizing: border-box;
   }
   .hoverlayer
   {
   z-index: 1000 !important;
   opacity: 1 !important;
   visibility: visible !important;
   }
   </style>
   <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
   <link rel="icon" href="favicon.ico" type="image/x-icon">
   </head>
   <body>
   <div class="header">
   <div class="title">
   <h1 class="title-main">SparseViz Tensor</h1>
   <h2 class="title-sub">Visualization</h2>
   </div>
   
   
    <div class="title">
   <h2>Ordering: )" + ordering->getOrderingName() + R"(</h2><hr>)";
    html_file << "<h2>Tensor: " << ordering->getTensor().getName() << "</h2>";
    html_file << "<h3 class=\"title-sub\">Dimensions: "; 
    html_file << ordering->getTensor().getDims()[0]; 
    for(int i = 1; i < ordering->getTensor().getOrder(); i++) {
        html_file << " x " << ordering->getTensor().getDims()[i];
     }
     html_file <<  "</h3>";
    html_file << "<h3 class=\"title-sub\">Nonzeros: " << ordering->getTensor().getNNZ() << "</h3>";

    html_file << "</div>\n"; // Close right header div
    html_file << "</div>\n"; // Close header div

    html_file << "<div id=\"aspect\">\n";
    html_file << "<button onClick=\"choose('data')\">Actual Tensor Sizes</button>\n";
    html_file << "<button onClick=\"choose('cube')\">Cube Tensors</button>\n";
    html_file << "<script>\n";
    html_file << "function choose(choice) {\n";
    for(int n = 0; n < norder; n++) {
        html_file << "choose_" << n << "(choice);\n";
    }
    html_file << "}\n";
    html_file << "</script>\n";
    html_file << "</div>\n";

    for (int n = 0; n < norder; n++) {
        vector<vector<int>> topDownSum(vis_dims[n][0], vector<int>(vis_dims[n][1], 0));
        vector<vector<int>> sideSum(vis_dims[n][1], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> tubeSum(vis_dims[n][0], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> topDownTotalValues(vis_dims[n][0], vector<int>(vis_dims[n][1], 0));
        vector<vector<int>> sideTotalValues(vis_dims[n][1], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> tubeTotalValues(vis_dims[n][0], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> topDownAbsTotalValues(vis_dims[n][0], vector<int>(vis_dims[n][1], 0));
        vector<vector<int>> sideAbsTotalValues(vis_dims[n][1], vector<int>(vis_dims[n][2], 0));
        vector<vector<int>> tubeAbsTotalValues(vis_dims[n][0], vector<int>(vis_dims[n][2], 0)); // for heatmaps
        nlohmann::json json_array;                                                        // for 3D Scatter plot
        nlohmann::json x_bins_json, y_nonzero_json;                                       // for barPlots
        nlohmann::json x_bins_abs_json, y_absTotalValues_json;                            // for barPlots
        vector<int> nonZeros;

        stats[n].no_bins =  vis_dims[n][2] * vis_dims[n][1] * vis_dims[n][0];
        stats[n].no_empty_bins = 0;
        for (int x = 0; x < vis_dims[n][0]; x++) {
            for (int y = 0; y < vis_dims[n][1]; y++) {
                for (int z = 0; z < vis_dims[n][2]; z++) {
                    nlohmann::json bin_json;
                    bin_json["binX"] = x;
                    bin_json["binY"] = y;
                    bin_json["binZ"] = z;
                    bin_json["nonzeroCount"] = tensorLists[n][x][y][z].nonzeroCount;
                    bin_json["totalValues"] = tensorLists[n][x][y][z].totalValues;
                    bin_json["absTotalValues"] = tensorLists[n][x][y][z].absTotalValues;
                    json_array.push_back(bin_json);
                    int nonZeroCount = tensorLists[n][x][y][z].nonzeroCount;

                    if (nonZeroCount == 0) {
                        stats[n].no_empty_bins++;
                        nonZeroCount = 1;
                    }
                    stats[n].geo_mean_nnz += log(nonZeroCount);

                    topDownSum[x][y] += tensorLists[n][x][y][z].nonzeroCount;
                    sideSum[y][z] += tensorLists[n][x][y][z].nonzeroCount;
                    tubeSum[x][z] += tensorLists[n][x][y][z].nonzeroCount;

                    topDownTotalValues[x][y] += tensorLists[n][x][y][z].totalValues;
                    sideTotalValues[y][z] += tensorLists[n][x][y][z].totalValues;
                    tubeTotalValues[x][z] += tensorLists[n][x][y][z].totalValues;

                    topDownAbsTotalValues[x][y] += tensorLists[n][x][y][z].absTotalValues;
                    sideAbsTotalValues[y][z] += tensorLists[n][x][y][z].absTotalValues;
                    tubeAbsTotalValues[x][z] += tensorLists[n][x][y][z].absTotalValues;

                    string coord_str = "(" + to_string(x) + ", " + to_string(y) + ", " + to_string(z) + ")";
                    x_bins_json.push_back(coord_str);                                        // insert bin number
                    y_nonzero_json.push_back(tensorLists[n][x][y][z].nonzeroCount);          // insert nonzeroCount value
                    x_bins_abs_json.push_back(coord_str);                                    // insert bin number
                    y_absTotalValues_json.push_back(tensorLists[n][x][y][z].absTotalValues); // insert absTotalValues value
                    if (tensorLists[n][x][y][z].nonzeroCount != 0) {
                        nonZeros.push_back(tensorLists[n][x][y][z].nonzeroCount);
                    }
                }
            }
        }

        std::sort(nonZeros.begin(), nonZeros.end());
        if (nonZeros.size() % 2 == 0) { // Even number of elements
            stats[n].median_nnz = (nonZeros[nonZeros.size() / 2 - 1] + nonZeros[nonZeros.size() / 2]) / 2.0;
        } else { // Odd number of elements
            stats[n].median_nnz = nonZeros[nonZeros.size() / 2];
        }
        stats[n].geo_mean_nnz = stats[n].geo_mean_nnz / (stats[n].no_bins - stats[n].no_empty_bins);
        stats[n].geo_mean_nnz = exp(stats[n].geo_mean_nnz);
        stats[n].mean_nnz = (double(ordering->getTensor().getNNZ())) / (stats[n].no_bins - stats[n].no_empty_bins);
       
        vector<vector<int>> *sums[3] = {&topDownSum, &sideSum, &tubeSum};
        vector<vector<int>> *totalValuesArrays[] = {&topDownTotalValues, &sideTotalValues, &tubeTotalValues};
        vector<vector<int>> *absTotalValuesArrays[] = {&topDownAbsTotalValues, &sideAbsTotalValues, &tubeAbsTotalValues};

        // converting the nlohmann::json object to a string
        string json_str = json_array.dump();
        json_str = escapeSingleQuote(json_str);

      html_file << "<div id='orderDiv" << n
                  << "' style='display: flex; flex-direction: row; align-items: center; "
                     "justify-content: space-around; margin-bottom: 5px;'>\n";
        html_file << "<div style='writing-mode: vertical-rl; display:flex; justify-content:center; text-align: center; align-items:center; transform: "
                     "rotate(180deg); margin-left:20px; margin-right: 10px;'>\n";
        html_file << "<h2>mods: " << std::to_string(active_modes_s[n][0]) << ", " << std::to_string(active_modes_s[n][1]) << ", " << std::to_string(active_modes_s[n][2]) << ""  
         << "<br>\n(dims: " << std::to_string(dims[n][0]) << " x  " << std::to_string(dims[n][1]) << " x  " << std::to_string(dims[n][2]) << ")</h3>\n";
        html_file << "</div>\n";
        html_file << "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: center; width: 100%;'>\n";

        // 3D scatter plot
        html_file << "<div id='myDiv" << n << "' style='width: 600px; height: 700px;'></div>\n";
        html_file << "<div style='display: flex; flex-direction: column; justify-content: space-between; width: 100%;'>\n"; // Change to column

        // js code for 3D scatter plot
        html_file << "<script>\n";
        html_file << "choose_" << n << "('data');\n";
        html_file << "function choose_" << n << "(choice)\n{\n";
    html_file << R"(
      try {
         var jsonData = ')" +
                     json_str + R"(';  // insert your JSON string here
         var data = JSON.parse(jsonData);

         // Extracting x, y, z coordinates and other data
         var x = data.map(d => d.binX);
         var y = data.map(d => d.binY);
         var z = data.map(d => d.binZ);
         var nonzeroCount = data.map(d => d.nonzeroCount);
         var totalValues = data.map(d => d.totalValues);
         var absTotalValues = data.map(d => d.absTotalValues);
         // Find the maximum nonZeroCount to normalize the counts
         var maxCount = Math.max(...data.map(d => d.nonzeroCount));
         // Generating hover text
         var hoverText = data.map(d => `Non-zero values: ${d.nonzeroCount}<br>Total value: ${d.totalValues}<br>Absolute total value: ${d.absTotalValues}`);

         // Creating the 3D scatter plot
         var xNonZero = [], yNonZero = [], zNonZero = [], countsNonZero = [], hoverTextNonZero = [];
         var xZero = [], yZero = [], zZero = [], hoverTextZero = [];

         for (var i = 0; i < nonzeroCount.length; i++) {
            if (nonzeroCount[i] == 0) {
               xZero.push(x[i] *)" << dims[n][0] / vis_dims[n][0] << R"( );
               yZero.push(y[i] *)" << dims[n][1] / vis_dims[n][1] << R"( );
               zZero.push(z[i] *)" << dims[n][2] / vis_dims[n][2] << R"( );
               hoverTextZero.push(hoverText[i]);
            } else {
               xNonZero.push(x[i] *)" << dims[n][0] / vis_dims[n][0] << R"( );
               yNonZero.push(y[i] *)" << dims[n][1] / vis_dims[n][1] << R"( );
               zNonZero.push(z[i] *)" << dims[n][2] / vis_dims[n][2] << R"( );
               countsNonZero.push(nonzeroCount[i]);
               hoverTextNonZero.push(hoverText[i]);
            }
         }

         // Create two separate traces
         var traceNonZero = {
         x: xNonZero,
         y: yNonZero,
         z: zNonZero,
         mode: 'markers',
         type: 'scatter3d',
         marker: {
         size: 8,
         color: countsNonZero,
         colorscale: 'Reds',
         opacity: 0.9
         },
         text: hoverTextNonZero,
         hoverinfo: 'text'
         };

         var traceZero = {
         x: xZero,
         y: yZero,
         z: zZero,
         mode: 'markers',
         type: 'scatter3d',
         marker: {
         size: 8,
         color: 'rgba(255,0,0,0)',  // Transparent red
         opacity: 0
         },
         text: hoverTextZero,
         hoverinfo: 'text'
         };

         var layout = {
         autosize: true,
         scene: {
            aspectmode: choice
         }, 
         width: 650,
         yaxis: {
            autorange: 'reversed' // This is for adjusting the main diagonal correctly
         },
         xaxis: {
            side: 'top'
         },
         height: 650,
         margin: {
         l: 10,
         r: 80,
         b: 10,
         t: 10
         }
         };

         var data = [traceNonZero];

         Plotly.newPlot('myDiv)" +
                     to_string(n) + R"(', data, layout);
      } catch (err) {
         console.log('Error parsing JSON or plotting data: ' + err.message);
      }
      }
      </script>
      )";

        html_file
                << "<div style='display: flex; flex-direction: row; justify-content: flex-start; width: 100%;'>\n"; // heatmap container

        for (int perspective = 0; perspective < 3; perspective++) { // 3 perspectives fibers rows tubes
            nlohmann::json json_arr;
            auto &sum = *sums[perspective];
            auto &totalValues = *totalValuesArrays[perspective];
            auto &absTotalValues = *absTotalValuesArrays[perspective];
            string titles[] = {"'Dimensions: X-Y'", "'Dimensions: Y-Z'", "'Dimensions: X-Z'"};
            string showColorBar = "false", width = "230";
            if(perspective == 2) {
                showColorBar = "true";
                width = "300";
                html_file << "<div id='myDiv" << n << "_" << perspective << "' style='width: 300; height: 250;'></div>\n";
            } else {
                html_file << "<div id='myDiv" << n << "_" << perspective << "' style='width: 230; height: 250;'></div>\n";
            }

            for (int i = 0; i < sum.size(); i++) {
                for (int j = 0; j < sum[i].size(); j++) {
                    nlohmann::json sum_json;
                    sum_json["x"] = i;
                    sum_json["y"] = j;
                    sum_json["sum"] = sum[i][j];
                    sum_json["totalValues"] = totalValues[i][j];
                    sum_json["absTotalValues"] = absTotalValues[i][j];
                    json_arr.push_back(sum_json);
                }
            }

            string json_str2 = json_arr.dump();
            json_str2 = escapeSingleQuote(json_str2);

            html_file << R"(
         <script>
         try {
            var jsonData2 = ')" + json_str2 + R"(';  // insert your JSON string here
            var data2 = JSON.parse(jsonData2);

            var x2 = [];
            var y2 = [];
            var z2 = [];
            var text = [];)";

            if(perspective == 0) {
                html_file << R"(  
                //custom hovertext creation
                for (var i = 0; i < data2.length; i++) {
                x2.push(data2[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( );
                y2.push(data2[i].y *)" << dims[n][1] / vis_dims[n][1] << R"( );
                z2.push(data2[i].sum);
                text.push(
                            "x: " + data2[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( +
                            "<br>y: " + data2[i].y *)" << dims[n][1] / vis_dims[n][1] << R"( +
                            "<br>NonZero: " + data2[i].sum +
                            "<br>Total Values: " + data2[i].totalValues +
                            "<br>Abs Total Values: " + data2[i].absTotalValues
                            );
                })";
            } else  if(perspective == 1) {
                html_file << R"(  
                //custom hovertext creation
                for (var i = 0; i < data2.length; i++) {
                x2.push(data2[i].x *)" << dims[n][1] / vis_dims[n][2] << R"( );
                y2.push(data2[i].y *)" << dims[n][2] / vis_dims[n][2] << R"( );
                z2.push(data2[i].sum);
                text.push(
                            "y: " + data2[i].x *)" << dims[n][1] / vis_dims[n][1] << R"( +
                            "<br>z: " + data2[i].y *)" << dims[n][2] / vis_dims[n][2] << R"( +
                            "<br>NonZero: " + data2[i].sum +
                            "<br>Total Values: " + data2[i].totalValues +
                            "<br>Abs Total Values: " + data2[i].absTotalValues
                            );
                })";
            } else  if(perspective == 2) {
                html_file << R"(  
                //custom hovertext creation
                for (var i = 0; i < data2.length; i++) {
                x2.push(data2[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( );
                y2.push(data2[i].y *)" << dims[n][2] / vis_dims[n][2] << R"( );
                z2.push(data2[i].sum);
                text.push(
                            "x: " + data2[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( +
                            "<br>z: " + data2[i].y *)" << dims[n][2] / vis_dims[n][2] << R"( +
                            "<br>NonZero: " + data2[i].sum +
                            "<br>Total Values: " + data2[i].totalValues +
                            "<br>Abs Total Values: " + data2[i].absTotalValues
                            );
                })";
            }



            html_file << R"( 
            // Creating the heatmap
            var trace2 = {
            x: x2,
            y: y2,
            z: z2,
            type: 'heatmap',
            colorscale: 'Reds',
            showscale: )" +
                         showColorBar + R"(,// set for only seen in the last heatmap
            text: text,
            hoverinfo: 'text'  // Only show the custom hover text
            };

            var layout2 = {
            title: )" + titles[perspective] +
                         R"(,
            autosize: false,
            width: )" + width +
                         R"(,// set different for the last heatmap since it has colorbar
            height: 250,
            yaxis: {
               autorange: 'reversed' // This is for adjusting the main diagonal correctly
            },
            xaxis: {
               side: 'bottom'
            },
            margin: {
            l: 50,
            r: 0,
            b: 50,
            t: 50
            }
            };

            var data2 = [trace2];

            Plotly.newPlot('myDiv)" + to_string(n) + "_" + to_string(perspective) + R"(', data2, layout2);
            } catch (err) {
               console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }

        // Sorting together to create the visualization as intended.
        vector<pair<string, int>> sorted_bins;
        vector<pair<string, double>> sorted_abs_bins; // Note: Using double for absTotalValues

        for (size_t i = 0; i < x_bins_json.size(); i++) {
            sorted_bins.push_back(make_pair(x_bins_json[i].get<string>(), y_nonzero_json[i]));
        }

        for (size_t i = 0; i < x_bins_abs_json.size(); i++) {
            sorted_abs_bins.push_back(make_pair(x_bins_abs_json[i].get<string>(), y_absTotalValues_json[i]));
        }
        // sort in descending order of y values (nonzero counts)
        std::sort(sorted_bins.begin(), sorted_bins.end(),
             [](const std::pair<string, int> &a, const std::pair<string, int> &b) {
                 return a.second > b.second; // change this to a.second < b.second; for ascending order
             });
        // sort in descending order of y values (absTotalValues)
        std::sort(sorted_abs_bins.begin(), sorted_abs_bins.end(),
             [](const std::pair<string, double> &a, const std::pair<string, double> &b) {
                 return a.second > b.second; // for descending order
             });
        // create new sorted json arrays
        nlohmann::json sorted_x_bins_json, sorted_y_nonzero_json;
        nlohmann::json sorted_x_bins_abs_json, sorted_y_absTotalValues_json;

        for (const auto &bin: sorted_bins) {
            sorted_x_bins_json.push_back(bin.first);
            sorted_y_nonzero_json.push_back(bin.second);
        }
        // create new sorted json arrays
        for (const auto &bin: sorted_abs_bins) {
            sorted_x_bins_abs_json.push_back(bin.first);
            sorted_y_absTotalValues_json.push_back(bin.second);
        }
        // creating a json object with two sorted arrays
        nlohmann::json sorted_data_hist;
        nlohmann::json sorted_data_abs_hist;
        sorted_data_abs_hist["x"] = sorted_x_bins_abs_json;
        sorted_data_abs_hist["y"] = sorted_y_absTotalValues_json;
        sorted_data_hist["x"] = sorted_x_bins_json;
        sorted_data_hist["y"] = sorted_y_nonzero_json;
        // using sorted_data_hist to create the plots.

        string histogramStr = sorted_data_hist.dump();
        string histogramAbsStr = sorted_data_abs_hist.dump(); // named histograms but barPlots. note that this data is currently not in use but its duplicate is used.

        html_file << "<div id='histogram" << n
                  << "' style='width: 250px; height: 250px;'></div>\n"; // New div for histogram
        html_file << R"(
       <script>
       try {
          var histogramData = ')" +
                     histogramStr + R"(';  // insert your histogram JSON string here
          var data = JSON.parse(histogramData);

          var trace3 = {
          x: data.x,
          y: data.y,
          type: 'bar',
          marker: {
          color: 'red',
          },
          opacity: 0.7,
          };

          var layout3 = {
          title: 'Non-zero Counts Dist',
          xaxis: {
          title: 'Bin',
          type: 'category',  // treat x-axis labels as categories
          autorange: false,  // disable auto range
          range: [0, data.x.length],
          showticklabels: false
          },
          yaxis: {
          type: 'log',
          dtick: Math.log2()  // interval between ticks
          },
          autosize: false,
          width: 250,
          height: 250,
          margin: {
          l: 50,
          r: 0,
          b: 50,
          t: 50
          }
          };

          var data3 = [trace3];
          Plotly.newPlot('histogram)" + to_string(n) + R"(', data3, layout3);
       } catch (err) {
          console.log('Error parsing JSON or plotting data: ' + err.message);
       }
       </script>
       )";

        html_file << "</div>\n"; // Close container for nnz heatmaps

        // New section for three new heatmaps based on absolute total values.
        html_file << "<div style='display: flex; flex-direction: row; justify-content: flex-start; width: 100%;'>\n"; // New heatmap container for absolute values

        for (int perspective = 0; perspective < 3; perspective++) {
            nlohmann::json json_arr_abs;
            auto &sum = *sums[perspective];
            auto &totalValues = *totalValuesArrays[perspective];
            auto &absTotalValues = *absTotalValuesArrays[perspective];
            string titles[] = {"'Dimensions: X-Y'", "'Dimensions: Y-Z'", "'Dimensions: X-Z'"};
            string showColorBar = "false", width = "230";
            if(perspective == 2) {
                showColorBar = "true";
                width = "300";
                html_file << "<div id='absDiv" << n << "_" << perspective << "' style='width: 300; height: 250;'></div>\n";
            } else {
                html_file << "<div id='absDiv" << n << "_" << perspective << "' style='width: 230; height: 250;'></div>\n";
            }

            for (int i = 0; i < absTotalValues.size(); i++) {
                for (int j = 0; j < absTotalValues[i].size(); j++) {
                    nlohmann::json abs_json;
                    abs_json["x"] = i;
                    abs_json["y"] = j;
                    abs_json["sum"] = sum[i][j];
                    abs_json["totalValues"] = totalValues[i][j];
                    abs_json["absTotalValues"] = absTotalValues[i][j];
                    json_arr_abs.push_back(abs_json);
                }
            }

            string json_str_abs = json_arr_abs.dump();
            json_str_abs = escapeSingleQuote(json_str_abs);

            html_file << R"(
         <script>
         try {
            var jsonDataAbs = ')" +
                         json_str_abs + R"(';  // insert your JSON string here
            var dataAbs = JSON.parse(jsonDataAbs);

            var xAbs = [];
            var yAbs = [];
            var zAbs = [];
            var textAbs = [];

            for (var i = 0; i < dataAbs.length; i++) {
               xAbs.push(dataAbs[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( );
               yAbs.push(dataAbs[i].y *)" << dims[n][1] / vis_dims[n][1] << R"( );
               zAbs.push(dataAbs[i].absTotalValues);
               textAbs.push(
                         "x: " + dataAbs[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( +
                         "<br>y: " + dataAbs[i].y *)" << dims[n][1] / vis_dims[n][1] << R"( +
                         "<br>NonZero: " + dataAbs[i].sum +
                         "<br>Total Values: " + dataAbs[i].totalValues +
                         "<br>Abs Total Values: " + dataAbs[i].absTotalValues
                         );
            }

            var traceAbs = {
            x: xAbs,
            y: yAbs,
            z: zAbs,
            type: 'heatmap',
            colorscale: [[0, 'rgb(247, 251, 255)'], [0.1, 'rgb(222, 235, 247)'], [0.2, 'rgb(198, 219, 239)'], [0.3, 'rgb(158, 202, 225)'], [0.4, 'rgb(107, 174, 214)'], [0.5, 'rgb(66, 146, 198)'], [0.6, 'rgb(33, 113, 181)'], [0.7, 'rgb(8, 81, 156)'], [0.8, 'rgb(8, 48, 107)'], [1, 'rgb(3, 19, 43)']], // Blues
            showscale: )" +
                         showColorBar + R"(,
            text: textAbs,
            hoverinfo: 'text'
            };

            var layoutAbs = {
            autosize: false,
            width: )" + width +
                         R"(,
            height: 250,
            yaxis: {
               autorange: 'reversed' // This is for adjusting the main diagonal correctly
            },
            xaxis: {
               side: 'bottom'
            },
            margin: {
            l: 50,
            r: 0,
            b: 50,
            t: 50
            }
            };

            var dataAbsPlot = [traceAbs];

            Plotly.newPlot('absDiv)" + to_string(n) + "_" + to_string(perspective) + R"(', dataAbsPlot, layoutAbs);
            } catch (err) {
               console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }

        html_file << "<div id='histogramAbs" << n << "' style='width: 250px; height: 250px;'></div>\n"; // New div for histogram of absTotalValues
        html_file << R"(
      <script>
      try {
         var histogramAbsData = ')" + histogramAbsStr + R"(';  // insert your histogram JSON string here
         var dataAbs = JSON.parse(histogramAbsData);

         var traceAbs = {
         x: dataAbs.x,
         y: dataAbs.y,
         type: 'bar',
         marker: {
         color: 'blue',  // Different color for absTotalValues
         },
         opacity: 0.7,
         };

         var layoutAbs = {
         title: 'Absolute Value Dist',
         xaxis: {
         title: 'Bin',
         type: 'category',
         autorange: false,
         range: [0, dataAbs.x.length],
         showticklabels: false
         },
         yaxis: {
         type: 'log',
         dtick: Math.log2()
         },
         autosize: false,
         width: 250,
          height: 250,
          margin: {
          l: 50,
          r: 0,
          b: 50,
          t: 50
          }
         };

         var dataAbsPlot = [traceAbs];
         Plotly.newPlot('histogramAbs)" +
                     to_string(n) + R"(', dataAbsPlot, layoutAbs);
      } catch (err) {
         console.log('Error parsing JSON or plotting data: ' + err.message);
      }
      </script>
      )";
        html_file << "</div>\n"; // Close new heatmap container for absolute values

        html_file << "</div>\n"; // close heatmap container

        // the div for the calculations panel
        html_file << R"(
      <div style='
      border: 2px solid black;
      border-radius: 15px;
      box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.3);
      margin: 20px;
      margin-right: 30px;
      padding: 20px;
      background-color: rgba(255, 255, 255, 0.9);
      transition: all 0.3s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
      overflow-x: auto; //adjustments to make it visible anytime.
      box-sizing: border-box;
      width: 100%;      // Container will take full width available
      max-width: 700px;
      min-width: 330px;
      height: 100%;
      max-height: 700px;
      '
      onmouseover="this.style.boxShadow='8px 8px 15px rgba(0, 0, 0, 0.3)';"
      onmouseout="this.style.boxShadow='5px 5px 10px rgba(0, 0, 0, 0.3)';">)"
                      << stat_to_html_table(*(ordering), stats[n])
                      << R"(</div>  
      )";
            html_file << "</div>\n"; // Close plotsContainer
            html_file << "</div>\n"; // Close orderDiv
        }


    html_file << R"(
    <div style='text-align: center; padding: 10px;'>
    <img src=')" + LOGO_PATH +
                 R"(' alt='Logo' style='width: 200px;'>
    <p> © 2024 SparCity</p>
    </div>
    </body>
    </html>
    )";
    html_file.close();
    logger->makeSilentLog("File " + filename + ".html" + " is generated", omp_get_wtime() - start_time);
   
    // Cleanup
    for (int n = 0; n < norder; ++n)
    {
        for (int i = 0; i < vis_dims[n][0]; ++i)
        {
            for (int j = 0; j < vis_dims[n][1]; ++j)
            {
                delete[] tensorLists[n][i][j];
            }
            delete[] tensorLists[n][i];
        }
        delete[] tensorLists[n];
    }
    delete[] tensorLists;
}
