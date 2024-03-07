#include "MatrixVisualizer.h"
#include <sstream>
#include <cstring>
#include "SparseVizTest.h"

std::string stat_to_html_table(const MatrixOrdering &o, const Statistic &stat)
{
    std::stringstream stream;
    std::string temp;

    std::ostringstream table;
    table << "<div class=\"container\">\n";
    table << "<table class=\"responsive-table\">\n";
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
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Stat</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Average</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Maximum</th>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">Normalized</th>\n";
    table << "</tr>\n";
    table << "</thead>\n";
    table << "<tbody>\n";
    table << "<tr>\n";
    table << "<th scope=\"row\" bgcolor=\"#D3D3D3\" color=\"white\" align=\"left\">Bandwidth</th>\n";
    stream << std::fixed << std::setprecision(0) << stat.avgBandwidth;
    table << "<td data-title=\"Average\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.maxBandwidth;
    table << "<td data-title=\"<Maximum>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.normalizedBandwidth;
    table << "<td data-title=\"<Normalized>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    table << "</tr>\n";
    table << "<tr>\n";
    table << "<th scope=\"row\" bgcolor=\"#D3D3D3\" color=\"white\" align=\"left\">Row Span</th>\n";
    stream << std::fixed << std::setprecision(0) << stat.avgRowSpan;
    table << "<td data-title=\"Average\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.maxRowSpan;
    table << "<td data-title=\"<Maximum>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.normalizedRowSpan;
    table << "<td data-title=\"<Normalized>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    table << "</tr>\n";
    table << "<tr>\n";
    table << "<th scope=\"row\" bgcolor=\"#D3D3D3\" color=\"white\" align=\"left\">Col Span</th>\n";
    stream << std::fixed << std::setprecision(0) << stat.avgColSpan;
    table << "<td data-title=\"Average\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.maxColSpan;
    table << "<td data-title=\"<Maximum>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    stream << std::fixed << std::setprecision(0) << stat.normalizedColSpan;
    table << "<td data-title=\"<Normalized>\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
    stream.str(std::string());
    table << "</tr>\n";
    table << "<thead>\n";
    table << "<tr>\n";
    table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">RBE</th>\n";
    for (int i = 0; i < NROWBLOCKS; i++)
    {
        stream << std::fixed << std::setprecision(0) << rowBlockSizes[i];
        table << "<th align=\"left\" bgcolor=\"#FFD3D3\" color=\"white\" scope=\"col\">" << stream.str() << "</th>\n";
        stream.str(std::string());
    }
    table << "</tr>\n";
    table << "</thead>\n";
    table << "<tbody>\n";
    table << "<tr>\n";
    table << "<th scope=\"row\" RBE=\"#D3D3D3\" color=\"white\" align=\"left\">"
          << ""
          << "</th>\n";
    for (int i = 0; i < NROWBLOCKS; i++)
    {
        stream << std::fixed << std::setprecision(0) << rowBlockSizes[i];
        std::string temp = stream.str();
        stream.str(std::string());
        stream << std::fixed << std::setprecision(2) << stat.rowBlockEfficiency[i];
        table << "<td data-title=\"" << temp << "\" data-type=\"number\" align=\"right\">" << stream.str() << "</td>\n";
        stream.str(std::string());
    }
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
        table << "<caption style=\"background-color:#FF3030; color:white;\"	> CPU KERNEL EXECUTION TIMES  </caption>\n";
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

void visualizeMatrixOrderings(MatrixOrdering **orderings, int norder)
{
    const SparseMatrix &matrix = orderings[0]->getMatrix();
    const std::string &filename = matrix.getName();

    unsigned int nnz = matrix.getNNZCount();
    const vType *rowPtr = matrix.getPtr();
    const vType *colInd = matrix.getInd();
    const valType *sortedValues = matrix.getValues();
    bool isSymmetric = matrix.isPatternSymmetric();

    unsigned int *dims = new unsigned int[2];
    dims[0] = matrix.getRowCount();
    dims[1] = matrix.getColCount();

    unsigned int scaleFactor = std::max(dims[0], dims[1]) / MAX_DIM;
    scaleFactor = scaleFactor == 0 ? 1 : scaleFactor;
    unsigned int *vis_dims = new unsigned int[2];
    vis_dims[0] = dims[0] / scaleFactor;
    vis_dims[1] = dims[1] / scaleFactor;
    if (vis_dims[0] < 8)
        vis_dims[0] = 8;
    if (vis_dims[1] < 8)
        vis_dims[1] = 8;

    // Initializations
    MatrixBin ***matrixBins = new MatrixBin **[norder];
    for (int n = 0; n < norder; n++)
    {
        matrixBins[n] = new MatrixBin *[vis_dims[1]];
        for (int i = 0; i < vis_dims[1]; i++)
        {
            matrixBins[n][i] = new MatrixBin[vis_dims[0]];
            memset(matrixBins[n][i], 0, vis_dims[0] * sizeof(int));
        }
    }

    MatrixBin ***matrixBinsHalf = new MatrixBin **[norder];
    for (int n = 0; n < norder; n++)
    {
        matrixBinsHalf[n] = new MatrixBin *[(vis_dims[1] + 1) / 2];
        for (int i = 0; i < ((vis_dims[1] + 1) / 2); i++)
        {
            matrixBinsHalf[n][i] = new MatrixBin[(vis_dims[0] + 1) / 2];
            memset(matrixBinsHalf[n][i], 0, ((vis_dims[0] + 1) / 2) * sizeof(int));
        }
    }

    MatrixBin ***matrixBinsDouble = new MatrixBin **[norder];
    for (int n = 0; n < norder; n++)
    {
        matrixBinsDouble[n] = new MatrixBin *[vis_dims[1] * 2];
        for (int i = 0; i < (vis_dims[1] * 2); i++)
        {
            matrixBinsDouble[n][i] = new MatrixBin[vis_dims[0] * 2];
            memset(matrixBinsDouble[n][i], 0, (vis_dims[0] * 2) * sizeof(int));
        }
    }
    Statistic *stats = new Statistic[norder]();
    int *minX = new int[dims[1]];
    int *maxX = new int[dims[1]];

    int rowBlockDistinctCounters[NROWBLOCKS];
    int rowBlockTotalCounters[NROWBLOCKS];
    int *colLastSeen = new int[dims[1]];
    int *rowPerm = new int[dims[0]];

    int *colDegrees = new int[dims[1]];
    memset(colDegrees, 0, sizeof(int) * dims[1]);
    for (int row = 0; row < dims[0]; ++row)
    {
        for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; ++idx)
        {
            int col = colInd[idx];
            colDegrees[col]++;
        }
    }

    double start_time = omp_get_wtime();

    for (int n = 0; n < norder; n++)
    {
        const vType *orderedRow = orderings[n]->getRowIPermutation();
        const vType *orderedCol = orderings[n]->getColIPermutation();

        for (int col = 0; col < dims[1]; col++)
        {
            minX[col] = dims[0];
            maxX[col] = -1;
            colLastSeen[col] = -1;
        }

        int d0 = dims[0];
        int d1 = dims[1];
        int dmax = std::max(d0, d1);
        double d0d1ratio = 1;
        double d1d0ratio = 1;
        if (d0 > d1)
        {
            d0d1ratio = ((double)d0) / d1;
        }
        else
        {
            d1d0ratio = ((double)d1) / d0;
        }

        for (int row = 0; row < dims[0]; ++row)
        {
            int orderedX = orderedRow[row];
            double binXp = ((double)orderedX) / dims[0];
            int binX = binXp * vis_dims[0];
            int binXhalf = binXp * ((vis_dims[0] + 1) / 2);
            int binXdouble = binXp * (vis_dims[0] * 2);

            int minY = dims[1];
            int maxY = -1;
            int nonzeroCountForRow = 0;
            

            for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; ++idx)
            {
                int col = colInd[idx];
                float value = sortedValues[idx];

                int orderedY = orderedCol[col];
                double binYp = ((double)orderedY) / dims[1];
                int binY = binYp * vis_dims[1];
                int binYhalf = binYp * ((vis_dims[1] + 1) / 2);
                int binYdouble = binYp * (vis_dims[1] * 2);

                matrixBins[n][binY][binX].nonzeroCount += 1;
                matrixBins[n][binY][binX].totalValues += value;
                matrixBins[n][binY][binX].absTotalValues += fabs(value);

                matrixBinsHalf[n][binYhalf][binXhalf].nonzeroCount += 1;
                matrixBinsHalf[n][binYhalf][binXhalf].totalValues += value;
                matrixBinsHalf[n][binYhalf][binXhalf].absTotalValues += fabs(value);

                matrixBinsDouble[n][binYdouble][binXdouble].nonzeroCount += 1;
                matrixBinsDouble[n][binYdouble][binXdouble].totalValues += value;
                matrixBinsDouble[n][binYdouble][binXdouble].absTotalValues += fabs(value);

                minY = std::min(minY, orderedY);
                maxY = std::max(maxY, orderedY);
                minX[col] = std::min(minX[col], orderedX);
                maxX[col] = std::max(maxX[col], orderedX);

                int fixedBW = std::abs(d0d1ratio * orderedY - d1d0ratio * orderedX);
                stats[n].avgBandwidth += fixedBW;
                stats[n].maxBandwidth = std::max(stats[n].maxBandwidth, fixedBW);
                int normalizer = std::abs(dmax - (d0d1ratio * orderedY + d1d0ratio * orderedX));
                stats[n].normalizedBandwidth += ((double)fixedBW) / std::max((int)(rowPtr[row + 1] - rowPtr[row + 1]), (int)(colDegrees[col]));
            }

            if (rowPtr[row + 1] - rowPtr[row] != 0)
            { 
                if (rowPtr[row + 1] - rowPtr[row] == 1)
                {
                    stats[n].normalizedRowSpan += 1;
                    stats[n].avgRowSpan += 1;
                    stats[n].maxRowSpan = std::max(stats[n].maxRowSpan, 1);
                } else {
                    stats[n].normalizedRowSpan += (maxY - minY) / ((double)(rowPtr[row + 1] - rowPtr[row]));
                    stats[n].avgRowSpan += (maxY - minY);
                    stats[n].maxRowSpan = std::max(stats[n].maxRowSpan, (maxY - minY));
                }
            }
        }
        stats[n].avgRowSpan = stats[n].avgRowSpan / dims[0];
        stats[n].normalizedRowSpan = stats[n].normalizedRowSpan / dims[0];
        stats[n].avgBandwidth = stats[n].avgBandwidth / rowPtr[dims[0]];
        stats[n].normalizedBandwidth = stats[n].normalizedBandwidth / rowPtr[dims[0]];

        for (int col = 0; col < dims[1]; col++)
        {
            if (colDegrees[col] != 0)
            { // column has at least one nonzero
                if(colDegrees[col] == 1) {
                    stats[n].normalizedColSpan += 1;
                    stats[n].avgColSpan += 1;
                    stats[n].maxColSpan = std::max(stats[n].maxColSpan, 1);
                } else {
                    stats[n].normalizedColSpan += (maxX[col] - minX[col]) / ((double)(colDegrees[col]));
                    stats[n].avgColSpan += maxX[col] - minX[col];
                    stats[n].maxColSpan = std::max(stats[n].maxColSpan, (maxX[col] - minX[col]));
                }
            }
        }

        stats[n].avgColSpan = stats[n].avgColSpan / dims[1];
        stats[n].normalizedColSpan = stats[n].normalizedColSpan / dims[1];

        memset(rowBlockDistinctCounters, 0, sizeof(int) * NROWBLOCKS);
        memset(rowBlockTotalCounters, 0, sizeof(int) * NROWBLOCKS);
        for (int i = 0; i < dims[0]; i++)
            rowPerm[orderedRow[i]] = i;
        for (int i = 0; i < dims[1]; i++)
            colLastSeen[i] = -1;

        for (int i = 0; i < dims[0]; ++i)
        {
            int row = rowPerm[i];
            for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; ++idx)
            {
                int col = colInd[idx];
                int orderedY = orderedCol[col];

                int prevLastSeen = colLastSeen[orderedY];
                for (int x = 0; x < NROWBLOCKS; x++)
                {
                    if (prevLastSeen == -1 || ((i / rowBlockSizes[x]) != (prevLastSeen / rowBlockSizes[x])))
                    {
                        rowBlockDistinctCounters[x]++;
                    }
                }
                colLastSeen[orderedY] = i;
            }

            for (int x = 0; x < NROWBLOCKS; x++)
            {
                rowBlockTotalCounters[x] += rowPtr[row + 1] - rowPtr[row];
                if (((i + 1) % rowBlockSizes[x]) == 0)
                {
                    if (rowBlockDistinctCounters[x] != 0)
                    {
                        //std::cout << x << " " << rowBlockTotalCounters[x] << " " << rowBlockDistinctCounters[x] << " " << rowBlockTotalCounters[x] / rowBlockDistinctCounters[x] << std::endl;
                        stats[n].rowBlockEfficiency[x] += (((double)(rowBlockTotalCounters[x])) / rowBlockDistinctCounters[x]);
                        rowBlockTotalCounters[x] = rowBlockDistinctCounters[x] = 0;
                    }
                }
            }
        }

        for (int x = 0; x < NROWBLOCKS; x++)
        {
            if (rowBlockDistinctCounters[x] != 0)
            {
                stats[n].rowBlockEfficiency[x] += (((double)(rowBlockTotalCounters[x])) / rowBlockDistinctCounters[x]);
            }
            rowBlockTotalCounters[x] = rowBlockDistinctCounters[x] = 0;
            stats[n].rowBlockEfficiency[x] /= ((dims[0] + rowBlockSizes[x] - 1) / rowBlockSizes[x]);
            //stats[n].rowBlockEfficiency[x] /= rowBlockSizes[x];
        } 
    }

    double end_time = omp_get_wtime();

    for (int i = 0; i != norder; ++i)
    {
        stats[i].matrixName = orderings[i]->getMatrix().getName();
        stats[i].orderingName = orderings[i]->getOrderingName();
        logger.logMatrixProcessing(MATRIX_VISUALIZATION_FILES_DIR + filename + ".html", stats[i], end_time - start_time);
    }

    std::string filePath;

#ifdef TEST
    filePath = SparseVizTest::getSparseVizTester()->getCurrentDirectory() + filename + ".html";
#else
    filePath = MATRIX_VISUALIZATION_FILES_DIR + filename + ".html";
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
      font-size:80%;
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
   <link rel="icon" href=')" +
                     FAVICON_PATH + R"(' type="image/x-icon">
   </head>
   <body>
   <div class="header">
   <div class="title">
   <h1 class="title-main">SparseViz Matrix</h1>
   <h2 class="title-sub">Visualization</h2>
   </div>
   
   <div class="title">
   <h2>Matrix Name: )" + filename + R"(</h2><hr>)";
    html_file << "<h3 class=\"title-sub\">Dimensions: " << dims[0] << " x " << dims[1] << "</h3>";
    html_file << "<h3 class=\"title-sub\">Nonzeros: " << nnz << "</h3>";
    
    html_file << "</div>"; // Close right header div
    html_file << "</div>"; // Close header div

    for (int n = 0; n < norder; n++)
    {
        html_file << "<div id='orderDiv" << n
                  << "' style='display: flex; flex-direction: row; align-items: center; "
                     "justify-content: space-around; margin-bottom: 5px;'>\n";
        html_file << "<div style='writing-mode: vertical-rl; display:flex; justify-content:center; align-items:center; transform: "
                     "rotate(180deg); margin-left:20px; margin-right: 10px;'>\n";
        html_file << "<h3>" << orderings[n]->getOrderingName() << "</h3>\n";
        html_file << "</div>\n";
        // plots container to align in row.
        html_file
            << "<div id='plotsContainer" << n
            << "' style='display: flex; flex-direction: row; justify-content: "
               "space-between; align-items: center; width: 100%;'>\n";
        nlohmann::json json_arr; // for 2D heat map
        nlohmann::json json_arr_half;
        nlohmann::json json_arr_double;
        nlohmann::json x_bins_json, y_nonzero_json;            // for barPlots
        nlohmann::json x_bins_abs_json, y_absTotalValues_json; // for barPlots

        // all calculations done here in this loop.
        std::vector<int> nonZeros;
        stats[n].no_bins = vis_dims[1] * vis_dims[0];
        stats[n].no_empty_bins = 0;
        for (int x = 0; x < vis_dims[1]; x++)
        {
            for (int y = 0; y < vis_dims[0]; y++)
            {
                nlohmann::json bin_json;
                bin_json["x"] = x;
                bin_json["y"] = y;
                bin_json["sum"] = matrixBins[n][x][y].nonzeroCount;
                bin_json["totalValues"] = matrixBins[n][x][y].totalValues;
                bin_json["absTotalValues"] = matrixBins[n][x][y].absTotalValues;
                json_arr.push_back(bin_json);

                int nonZeroCount = matrixBins[n][x][y].nonzeroCount;
                if (nonZeroCount == 0)
                {
                    stats[n].no_empty_bins++;
                    nonZeroCount = 1;
                }
                stats[n].geo_mean_nnz += log(nonZeroCount);

                std::string coord_str = "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
                x_bins_json.push_back(coord_str);                                    // insert bin number
                y_nonzero_json.push_back(matrixBins[n][x][y].nonzeroCount);          // insert nonzeroCount value
                x_bins_abs_json.push_back(coord_str);                                // insert bin number
                y_absTotalValues_json.push_back(matrixBins[n][x][y].absTotalValues); // insert absTotalValues value
                if (matrixBins[n][x][y].nonzeroCount != 0) {
                    nonZeros.push_back(matrixBins[n][x][y].nonzeroCount);
                }
            }
        }

        for (int x = 0; x < ((vis_dims[1] + 1) / 2); x++)
        {
            for (int y = 0; y < ((vis_dims[0] + 1) / 2); y++)
            {
                nlohmann::json bin_json_half;
                bin_json_half["x"] = x;
                bin_json_half["y"] = y;
                bin_json_half["sum"] = matrixBinsHalf[n][x][y].nonzeroCount;
                bin_json_half["totalValues"] = matrixBinsHalf[n][x][y].totalValues;
                bin_json_half["absTotalValues"] = matrixBinsHalf[n][x][y].absTotalValues;
                json_arr_half.push_back(bin_json_half);
            }
        }

        for (int x = 0; x < (vis_dims[1] * 2); x++)
        {
            for (int y = 0; y < (vis_dims[0] * 2); y++)
            {
                nlohmann::json bin_json_double;
                bin_json_double["x"] = x;
                bin_json_double["y"] = y;
                bin_json_double["sum"] = matrixBinsDouble[n][x][y].nonzeroCount;
                bin_json_double["totalValues"] = matrixBinsDouble[n][x][y].totalValues;
                bin_json_double["absTotalValues"] = matrixBinsDouble[n][x][y].absTotalValues;
                json_arr_double.push_back(bin_json_double);
            }
        }

        sort(nonZeros.begin(), nonZeros.end());
        if (nonZeros.size() % 2 == 0)
        { // Even number of elements
            stats[n].median_nnz = (nonZeros[nonZeros.size() / 2 - 1] + nonZeros[nonZeros.size() / 2]) / 2.0;
        }
        else
        { // Odd number of elements
            stats[n].median_nnz = nonZeros[nonZeros.size() / 2];
        }
        stats[n].geo_mean_nnz = stats[n].geo_mean_nnz / (stats[n].no_bins - stats[n].no_empty_bins);
        stats[n].geo_mean_nnz = exp(stats[n].geo_mean_nnz);
        stats[n].mean_nnz = (double(nnz)) / (stats[n].no_bins - stats[n].no_empty_bins);

        std::string json_str2 = json_arr.dump();
        json_str2 = escapeSingleQuote(json_str2);

        std::string json_str2_half = json_arr_half.dump();
        json_str2_half = escapeSingleQuote(json_str2_half);

        std::string json_str2_double = json_arr_double.dump();
        json_str2_double = escapeSingleQuote(json_str2_double);
        
        bool is_red = true;
        if (CHART_TYPE == "ABS") {
            is_red = false;
        }

        html_file << "<div id='myDiv" << n << "' style='width:600px; height:500px;'></div>\n";
        html_file << R"(
            <script>
            try {
                var jsonData2 = ')" << json_str2 << R"(';  // insert your JSON string here
                var data2 = JSON.parse(jsonData2);

                var x2 = [];
                var y2 = [];
                var z2 = [];
                var text = [];

                for (var i = 0; i < data2.length; i++) {
                    x2.push(data2[i].x *)" << dims[0] / vis_dims[0] << R"( );
                    y2.push(data2[i].y *)" << dims[1] / vis_dims[1] << R"( );)";
        if (CHART_TYPE == "NNZ")
        {
            html_file << R"(z2.push(data2[i].sum);)";
        }
        else if (CHART_TYPE == "ABS")
        {
            html_file << R"(z2.push(data2[i].absTotalValues);)";
        }
        else
        {   
            html_file << R"(z2.push(data2[i].sum / data2[i].sum);)";
        }

        html_file << R"(text.push(
                            "x: " + data2[i].x *)" << dims[0] / vis_dims[0] << R"( +
                            "<br>y: " + data2[i].y *)" << dims[1] / vis_dims[1] << R"( +
                            "<br>NonZero: " + data2[i].sum +
                            "<br>Total Values: " + data2[i].totalValues +
                            "<br>Abs Total Values: " + data2[i].absTotalValues
                            );
                }

                // Creating the heatmap
                var trace2 = {
                x: x2,
                y: y2,
                z: z2,
                visible: true, // initially  shown
                type: 'heatmap',
                colorscale: )" << calculateColorscale(EXPONENTIAL_COLORSCALE, is_red) << R"(,
                showscale: true,
                text: text,
                hoverinfo: 'text'  // Only show the custom hover text
                };

                var jsonData2_half = ')" << json_str2_half << R"(';  // insert your JSON string here
                var data3 = JSON.parse(jsonData2_half);

                var x2h = [];
                var y2h = [];
                var z2h = [];
                var texth = [];

                for (var i = 0; i < data3.length; i++) {
                    x2h.push(data3[i].x *)" << dims[0] / ((vis_dims[0] + 1) / 2) << R"( );
                    y2h.push(data3[i].y *)" << dims[1] / ((vis_dims[1] + 1) / 2) << R"( );)";
        if (CHART_TYPE == "NNZ")
        {
            html_file << R"(z2h.push(data3[i].sum);)";
        }
        else if (CHART_TYPE == "ABS")
        {
            html_file << R"(z2h.push(data3[i].absTotalValues);)";
        }
        else
        {
            html_file << R"(z2h.push(data3[i].sum / data3[i].sum);)";
        }
        html_file << R"(texth.push(
                            "x: " + data3[i].x *)" << dims[0] / ((vis_dims[0] + 1) / 2) << R"( +
                            "<br>y: " + data3[i].y *)" << dims[1] / ((vis_dims[1] + 1) / 2) << R"( +
                            "<br>NonZero: " + data3[i].sum +
                            "<br>Total Values: " + data3[i].totalValues +
                            "<br>Abs Total Values: " + data3[i].absTotalValues
                            );
                }

                var trace3 = {
                x: x2h,
                y: y2h,
                z: z2h,
                visible: false, // initially not shown
                type: 'heatmap',
                colorscale: )" << calculateColorscale(EXPONENTIAL_COLORSCALE, is_red) << R"(,
                showscale: true,
                text: texth,
                hoverinfo: 'text'  // Only show the custom hover text
                };

                var jsonData2_double = ')" << json_str2_double << R"(';  // insert your JSON string here
                var data4 = JSON.parse(jsonData2_double);

                var x2d = [];
                var y2d = [];
                var z2d = [];
                var textd = [];

                for (var i = 0; i < data4.length; i++) {
                    x2d.push(data4[i].x *)" << (dims[0] / (2 * vis_dims[0])) << R"( );
                    y2d.push(data4[i].y *)" << (dims[1] / (2 * vis_dims[1])) << R"( );)";
        if (CHART_TYPE == "NNZ")
        {
            html_file << R"(z2d.push(data4[i].sum);)";
        }
        else if (CHART_TYPE == "ABS")
        {
            html_file << R"(z2d.push(data4[i].absTotalValues);)";
        }
        else
        {
            html_file << R"(z2d.push(data4[i].sum / data4[i].sum );)";
        }
        html_file << R"(textd.push(
                            "x: " + data4[i].x *)" << (dims[0] / (2 * vis_dims[0])) << R"( +
                            "<br>y: " + data4[i].y *)" << (dims[1] / (2 * vis_dims[1])) << R"( +
                            "<br>NonZero: " + data4[i].sum +
                            "<br>Total Values: " + data4[i].totalValues +
                            "<br>Abs Total Values: " + data4[i].absTotalValues
                            );
                }

                var trace4 = {
                x: x2d,
                y: y2d,
                z: z2d,
                visible: false, // initially not shown
                type: 'heatmap',
                colorscale: )" << calculateColorscale(EXPONENTIAL_COLORSCALE, is_red) << R"(,
                showscale: true,
                text: textd,
                autosize: 'false',
                hoverinfo: 'text'  // Only show the custom hover text
                };

                var layout2 = {
                plot_bgcolor: '#FFFFFF',
                autosize: false,
                width: 500,
                yaxis: {
                    autorange: 'reversed', // This is for adjusting the main diagonal correctly
                    scaleanchor: 'x',
                    type: 'category',
                    constrain: 'domain'
                },
                xaxis: {
                    side: 'top',
                    type: 'category',
                    constrain: 'domain'
                },
                height: 500,
                sliders: [{
                    active: 1,
                    direction: 'center',
                    showactive: true,
                    currentvalue: {
                    xanchor: 'center',
                    yanchor: 'bottom',
                    prefix: 'Resolution: ',
                    font: {
                    color: '#888',
                    size: 10
                    }
                    },
                    steps: [{
                    label: '50\%',
                    method: 'restyle',
                    args: ['visible', [false, true, false]],
                    }, {
                    label: '100\%',
                    method: 'restyle',
                    args: ['visible', [true, false, false]],
                    }, {
                    label: '200\%',
                    method: 'restyle',
                    args: ['visible', [false, false, true]],
                    }]
                }]
                };
                var data2 = [trace2, trace3, trace4];

                Plotly.newPlot('myDiv)" + std::to_string(n) + R"(', data2, layout2);
                } catch (err) {
                    console.log('Error parsing JSON or plotting data: ' + err.message);
                }
                </script>
                )";

        if (CHART_TYPE != "ABS")
        {
            std::vector<std::pair<std::string, int>> sorted_bins;
            for (size_t i = 0; i < x_bins_json.size(); i++) {
                sorted_bins.push_back(make_pair(x_bins_json[i].get<std::string>(), y_nonzero_json[i]));
            }
            sort(sorted_bins.begin(), sorted_bins.end(),
                 [](const std::pair<std::string, int> &a, const std::pair<std::string, int> &b) {
                     return a.second > b.second; // change this to a.second < b.second;
                     // for ascending order
                 });
            nlohmann::json sorted_x_bins_json, sorted_y_nonzero_json;
            for (const auto &bin : sorted_bins) {
                sorted_x_bins_json.push_back(bin.first);
                sorted_y_nonzero_json.push_back(bin.second);
            }
            nlohmann::json sorted_data_hist;
            sorted_data_hist["x"] = sorted_x_bins_json;
            sorted_data_hist["y"] = sorted_y_nonzero_json;

            std::string histogramStr = sorted_data_hist.dump();
            html_file << "<div id='histogram" << n
                      << "' style='width:450px; height:300px;'></div>\n";
            html_file << R"(
            <script>
            try {
                var histogramData = ')" + histogramStr + R"(';  // insert your histogram JSON string here
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
                title: 'Bin NNZCount',
                font: {
                color: '#000000',
                size: 10
                },
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
                width: 300,
                height: 300,
                margin: {
                l: 50,
                r: 20,
                b: 50,
                t: 50
                }
                };

                var data3 = [trace3];

                Plotly.newPlot('histogram)" +
                             std::to_string(n) + R"(', data3, layout3);
            } catch (err) {
                console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }
        else if (CHART_TYPE == "ABS")
        {
            std::vector<std::pair<std::string, double>> sorted_abs_bins; // Note: Using double for absTotalValues
            for (size_t i = 0; i < x_bins_abs_json.size(); i++) {
                sorted_abs_bins.push_back(make_pair(x_bins_abs_json[i].get<std::string>(), y_absTotalValues_json[i]));
            }
            sort(sorted_abs_bins.begin(), sorted_abs_bins.end(),
                [](const std::pair<std::string, double> &a,
                    const std::pair<std::string, double> &b)
                {
                    return a.second > b.second; // for descending order
                });
            nlohmann::json sorted_x_bins_abs_json, sorted_y_absTotalValues_json;
            for (const auto &bin : sorted_abs_bins)
            {
                sorted_x_bins_abs_json.push_back(bin.first);
                sorted_y_absTotalValues_json.push_back(bin.second);
            }
            nlohmann::json sorted_data_abs_hist;
            sorted_data_abs_hist["x"] = sorted_x_bins_abs_json;
            sorted_data_abs_hist["y"] = sorted_y_absTotalValues_json;
            // using the sorted_data_hist
            std::string histogramAbsStr = sorted_data_abs_hist.dump();
            html_file << "<div id='histogramAbs" << n
                      << "' style='width:450px; height:300px;'></div>\n";
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
                opacity: 0.5,
                };

                var layoutAbs = {
                title: 'Bin Absolute Values',
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
                width: 350,
                height: 300,
                margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50
                }
                };

                var dataAbsPlot = [traceAbs];

                Plotly.newPlot('histogramAbs)" +
                             std::to_string(n) + R"(', dataAbsPlot, layoutAbs);
            } catch (err) {
                console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }
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

    // CleanUp
    for (int n = 0; n < norder; n++)
    {
        for (int i = 0; i < vis_dims[1]; i++)
        {
            delete[] matrixBins[n][i];
        }
        delete[] matrixBins[n];
    }
    delete[] matrixBins;
    for (int n = 0; n < norder; n++)
    {
        for (int i = 0; i < ((vis_dims[1] + 1) / 2); i++)
        {
            delete[] matrixBinsHalf[n][i];
        }
        delete[] matrixBinsHalf[n];
    }
    delete[] matrixBinsHalf;
    for (int n = 0; n < norder; n++)
    {
        for (int i = 0; i < (vis_dims[1] * 2); i++)
        {
            delete[] matrixBinsDouble[n][i];
        }
        delete[] matrixBinsDouble[n];
    }
    delete[] matrixBinsDouble;
    delete[] stats;
    delete[] minX;
    delete[] maxX;
    delete[] dims;
    delete[] vis_dims;
    delete[] colLastSeen;
    delete[] rowPerm;
    delete[] colDegrees;
}

void visualizeMatrices(MatrixOrdering **orderings, int norder)
{   
    const std::string &filename = orderings[0]->getOrderingName();

    std::vector<const SparseMatrix *> matrices(norder);
    std::vector<unsigned int> nnzs(norder);
    std::vector<const vType *> rowPtrs(norder);
    std::vector<const vType *> colInds(norder);
    std::vector<const valType *> sortedValues(norder);
    std::vector<bool> symmetries(norder);
    std::vector<std::vector<unsigned int>> dims(norder, std::vector<unsigned int>(2));
    std::vector<std::vector<unsigned int>> vis_dims(norder, std::vector<unsigned int>(2));
    int maxDim0 = 0, maxDim1 = 0;

    for (int i = 0; i != norder; ++i)
    {
        const SparseMatrix &matrix = orderings[i]->getMatrix();
        matrices[i] = &matrix;
        nnzs[i] = matrix.getNNZCount();
        rowPtrs[i] = matrix.getPtr();
        colInds[i] = matrix.getInd();
        symmetries[i] = matrix.isPatternSymmetric();
        sortedValues[i] = matrix.getValues();
        dims[i][0] = matrix.getRowCount();
        dims[i][1] = matrix.getColCount();

        maxDim0 = std::max((int)dims[i][0], maxDim0);
        maxDim1 = std::max((int)dims[i][1], maxDim1);
        
        unsigned int scaleFactor = std::max(dims[i][0], dims[i][1]) / MAX_DIM;
        scaleFactor = scaleFactor == 0 ? 1 : scaleFactor;
        vis_dims[i][0] = dims[i][0] / scaleFactor;
        vis_dims[i][1] = dims[i][1] / scaleFactor;
        if (vis_dims[i][0] < 8)
            vis_dims[i][0] = 8;
        if (vis_dims[i][1] < 8)
            vis_dims[i][1] = 8;
    }

    // Initializations
    MatrixBin ***matrixBins = new MatrixBin **[norder];
    for (int n = 0; n < norder; n++)
    {
        matrixBins[n] = new MatrixBin *[vis_dims[n][1]];
        for (int i = 0; i < vis_dims[n][1]; i++)
        {
            matrixBins[n][i] = new MatrixBin[vis_dims[n][0]];
        }
    }

    MatrixBin ***matrixBinsHalf = new MatrixBin **[norder];
    for (int n = 0; n < norder; n++)
    {
        matrixBinsHalf[n] = new MatrixBin *[(vis_dims[n][1] + 1) / 2];
        for (int i = 0; i < ((vis_dims[n][1] + 1) / 2); i++)
        {
            matrixBinsHalf[n][i] = new MatrixBin[(vis_dims[n][0] + 1) / 2];
        }
    }

    MatrixBin ***matrixBinsDouble = new MatrixBin **[norder];
    for (int n = 0; n < norder; n++)
    {
        matrixBinsDouble[n] = new MatrixBin *[vis_dims[n][1] * 2];
        for (int i = 0; i < (vis_dims[n][1] * 2); i++)
        {
            matrixBinsDouble[n][i] = new MatrixBin[vis_dims[n][0] * 2];
        }
    }

    Statistic *stats = new Statistic[norder]();
    int *minX = new int[maxDim1];
    int *maxX = new int[maxDim1];

    int rowBlockDistinctCounters[NROWBLOCKS];
    int rowBlockTotalCounters[NROWBLOCKS];
    int *colLastSeen = new int[maxDim1];
    int *rowPerm = new int[maxDim0];

    int *colDegrees = new int[maxDim1];

    double start_time = omp_get_wtime();

    for (int n = 0; n < norder; n++)
    {    
        memset(colDegrees, 0, sizeof(int) * dims[n][1]);
        for (int row = 0; row < dims[n][0]; ++row)
        {
            for (int idx = rowPtrs[n][row]; idx < rowPtrs[n][row + 1]; ++idx)
            {
                int col = colInds[n][idx];
                colDegrees[col]++;
            }
        }

        for (int col = 0; col < dims[n][1]; col++)
        {
            minX[col] = dims[n][0];
            maxX[col] = -1;
            colLastSeen[col] = -1;
        }

        const vType *orderedRow = orderings[n]->getRowIPermutation();
        const vType *orderedCol = orderings[n]->getColIPermutation();
        
        int d0 = dims[n][0];
        int d1 = dims[n][1];
        int dmax = std::max(d0, d1);
        double d0d1ratio = 1;
        double d1d0ratio = 1;
        if (d0 > d1)
        {
            d0d1ratio = ((double)d0) / d1;
        }
        else
        {
            d1d0ratio = ((double)d1) / d0;
        }  

        for (int row = 0; row < dims[n][0]; ++row)
        {
            int orderedX = orderedRow[row];
            double binXp = ((double)orderedX) / dims[n][0];
            int binX = binXp * vis_dims[n][0];
            int binXhalf = binXp * ((vis_dims[n][0] + 1) / 2);
            int binXdouble = binXp * (vis_dims[n][0] * 2);

            int minY = dims[n][0];
            int maxY = 0;
            int nonzeroCountForRow = 0;

            for (int idx = rowPtrs[n][row]; idx < rowPtrs[n][row + 1]; ++idx)
            {
                int col = colInds[n][idx];
                float value = sortedValues[n][idx];

                int orderedY = orderedCol[col];
                double binYp = ((double)orderedY) / dims[n][1];
                int binY = binYp * vis_dims[n][1];
                int binYhalf = binYp * ((vis_dims[n][1] + 1) / 2);
                int binYdouble = binYp * (vis_dims[n][1] * 2);

                matrixBins[n][binY][binX].nonzeroCount += 1;
                matrixBins[n][binY][binX].totalValues += value;
                matrixBins[n][binY][binX].absTotalValues += fabs(value);

                matrixBinsHalf[n][binYhalf][binXhalf].nonzeroCount += 1;
                matrixBinsHalf[n][binYhalf][binXhalf].totalValues += value;
                matrixBinsHalf[n][binYhalf][binXhalf].absTotalValues += fabs(value);

                matrixBinsDouble[n][binYdouble][binXdouble].nonzeroCount += 1;
                matrixBinsDouble[n][binYdouble][binXdouble].totalValues += value;
                matrixBinsDouble[n][binYdouble][binXdouble].absTotalValues += fabs(value);
                
                minY = std::min(minY, orderedY);
                maxY = std::max(maxY, orderedY);
                minX[col] = std::min(minX[col], orderedX);
                maxX[col] = std::max(maxX[col], orderedX);

                int fixedBW = std::abs(d0d1ratio * orderedY - d1d0ratio * orderedX);
                stats[n].avgBandwidth += fixedBW;
                stats[n].maxBandwidth = std::max(stats[n].maxBandwidth, fixedBW);
                int normalizer = std::abs(dmax - (d0d1ratio * orderedY + d1d0ratio * orderedX));
                stats[n].normalizedBandwidth += ((double)fixedBW) / std::max((int)(rowPtrs[n][row + 1] - rowPtrs[n][row + 1]), (int)(colDegrees[col]));
            }

            if (rowPtrs[n][row + 1] - rowPtrs[n][row] != 0)
            { 
                if (rowPtrs[n][row + 1] - rowPtrs[n][row] == 1)
                {
                    stats[n].normalizedRowSpan += 1;
                    stats[n].avgRowSpan += 1;
                    stats[n].maxRowSpan = std::max(stats[n].maxRowSpan, 1);
                } else {
                    stats[n].normalizedRowSpan += (maxY - minY) / ((double)(rowPtrs[n][row + 1] - rowPtrs[n][row]));
                    stats[n].avgRowSpan += (maxY - minY);
                    stats[n].maxRowSpan = std::max(stats[n].maxRowSpan, (maxY - minY));
                }
            }
        }

        stats[n].avgRowSpan = stats[n].avgRowSpan / dims[n][0];
        stats[n].normalizedRowSpan = stats[n].normalizedRowSpan / dims[n][0];
        stats[n].avgBandwidth = stats[n].avgBandwidth / rowPtrs[n][dims[n][0]];
        stats[n].normalizedBandwidth = stats[n].normalizedBandwidth / rowPtrs[n][dims[n][0]];

       for (int col = 0; col < dims[n][1]; col++)
        {
            if (colDegrees[col] != 0)
            { // column has at least one nonzero
                if(colDegrees[col] == 1) {
                    stats[n].normalizedColSpan += 1;
                    stats[n].avgColSpan += 1;
                    stats[n].maxColSpan = std::max(stats[n].maxColSpan, 1);
                } else {
                    stats[n].normalizedColSpan += (maxX[col] - minX[col]) / ((double)(colDegrees[col]));
                    stats[n].avgColSpan += maxX[col] - minX[col];
                    stats[n].maxColSpan = std::max(stats[n].maxColSpan, (maxX[col] - minX[col]));
                }
            }
        }

        stats[n].avgColSpan = stats[n].avgColSpan / dims[n][1];
        stats[n].normalizedColSpan = stats[n].normalizedColSpan / dims[n][1];

        memset(rowBlockDistinctCounters, 0, sizeof(int) * NROWBLOCKS);
        memset(rowBlockTotalCounters, 0, sizeof(int) * NROWBLOCKS);
        for (int i = 0; i < dims[n][0]; i++)
            rowPerm[orderedRow[i]] = i;
        for (int i = 0; i < dims[n][1]; i++)
            colLastSeen[i] = -1;

        for (int i = 0; i < dims[n][0]; ++i)
        {
            int row = rowPerm[i];
            for (int idx = rowPtrs[n][row]; idx < rowPtrs[n][row + 1]; ++idx)
            {
                int col = colInds[n][idx];
                int orderedY = orderedCol[col];

                int prevLastSeen = colLastSeen[orderedY];
                for (int x = 0; x < NROWBLOCKS; x++)
                {
                    if (prevLastSeen == -1 || ((i / rowBlockSizes[x]) != (prevLastSeen / rowBlockSizes[x])))
                    {
                        rowBlockDistinctCounters[x]++;
                    }
                }
                colLastSeen[orderedY] = i;
            }

            for (int x = 0; x < NROWBLOCKS; x++)
            {
                rowBlockTotalCounters[x] += rowPtrs[n][row + 1] - rowPtrs[n][row];
                if (((i + 1) % rowBlockSizes[x]) == 0)
                {
                    if (rowBlockDistinctCounters[x] != 0)
                    {
                        stats[n].rowBlockEfficiency[x] += (((double)(rowBlockTotalCounters[x])) / rowBlockDistinctCounters[x]);
                        rowBlockTotalCounters[x] = rowBlockDistinctCounters[x] = 0;
                    }
                }
            }
        }

        for (int x = 0; x < NROWBLOCKS; x++)
        {
            if (rowBlockDistinctCounters[x] != 0)
            {
                stats[n].rowBlockEfficiency[x] += (((double)(rowBlockTotalCounters[x])) / rowBlockDistinctCounters[x]);
            }
            rowBlockTotalCounters[x] = rowBlockDistinctCounters[x] = 0;
            stats[n].rowBlockEfficiency[x] /= ((dims[n][0] + rowBlockSizes[x] - 1) / rowBlockSizes[x]);
            //stats[n].rowBlockEfficiency[x] /= rowBlockSizes[x];
        }
    }

    double end_time = omp_get_wtime();

    for (int i = 0; i != norder; ++i)
    {
        stats[i].matrixName = orderings[i]->getMatrix().getName();
        stats[i].orderingName = orderings[i]->getOrderingName();
        logger.logMatrixProcessing(MATRIX_VISUALIZATION_FILES_DIR + filename + ".html", stats[i], end_time - start_time);
    }

    std::string filePath;

#ifdef TEST
    filePath = SparseVizTest::getSparseVizTester()->getCurrentDirectory() + filename + ".html";
#else
    filePath = MATRIX_VISUALIZATION_FILES_DIR + filename + ".html";
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
      font-size:80%;
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
   <link rel="icon" href=')" +
                     FAVICON_PATH + R"(' type="image/x-icon">
   </head>
   <body>
   <div class="header">
   <div class="title">
   <h1 class="title-main">SparseViz Matrix</h1>
   <h2 class="title-sub">Visualization</h2>
   </div>
   <div>
   <h2>Ordering Name: )" + "" + R"(</h2><hr>)";
    html_file << "<h3 class=\"title-sub\">" << filename << "</h3>";
    html_file << "</div>\n"; // Close right header div
    html_file << "</div>\n"; // Close header div

    for (int n = 0; n < norder; n++) {
        html_file
            << "<div id='orderDiv" << n
            << "' style='display: flex; flex-direction: row; align-items: center; "
               "justify-content: space-around; margin-bottom: 5px;'>\n";
        html_file << "<div style='writing-mode: vertical-rl; display:flex; justify-content:center; align-items:center; transform: "
                     "rotate(180deg); margin-left:20px; margin-right: 10px;'>\n";
        html_file << "<h3>" << orderings[n]->getMatrix().getName() << "<br>\n(dims: " << std::to_string(dims[n][0]) << " x  " << std::to_string(dims[n][1]) << ")" << "<br>\n(nnz: " << std::to_string(nnzs[n]) << ")</h3>\n";
        html_file << "</div>\n";
        // plots container to align in row.
        html_file
            << "<div id='plotsContainer" << n
            << "' style='display: flex; flex-direction: row; justify-content: "
               "space-between; align-items: center; width: 100%;'>\n";
        nlohmann::json json_arr; // for 2D heat map
        nlohmann::json json_arr_half;
        nlohmann::json json_arr_double;
        nlohmann::json x_bins_json, y_nonzero_json;            // for barPlots
        nlohmann::json x_bins_abs_json, y_absTotalValues_json; // for barPlots

        // all calculations done here in this loop.
        std::vector<int> nonZeros;
        stats[n].no_bins = vis_dims[n][1] * vis_dims[n][0];

        for (int x = 0; x < vis_dims[n][1]; x++)
        {
            for (int y = 0; y < vis_dims[n][0]; y++)
            {
                nlohmann::json bin_json;
                bin_json["x"] = x;
                bin_json["y"] = y;
                bin_json["sum"] = matrixBins[n][x][y].nonzeroCount;
                bin_json["totalValues"] = matrixBins[n][x][y].totalValues;
                bin_json["absTotalValues"] = matrixBins[n][x][y].absTotalValues;
                json_arr.push_back(bin_json);

                int nonZeroCount = matrixBins[n][x][y].nonzeroCount;
                if (nonZeroCount == 0)
                {
                    stats[n].no_empty_bins++;
                    nonZeroCount = 1;
                }
                stats[n].geo_mean_nnz += log(nonZeroCount);

                std::string coord_str = "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
                x_bins_json.push_back(coord_str); // insert bin number
                y_nonzero_json.push_back(matrixBins[n][x][y].nonzeroCount); // insert nonzeroCount value
                x_bins_abs_json.push_back(coord_str);  // insert bin number
                y_absTotalValues_json.push_back(matrixBins[n][x][y].absTotalValues); // insert absTotalValues value
                if (matrixBins[n][x][y].nonzeroCount != 0)
                    nonZeros.push_back(matrixBins[n][x][y].nonzeroCount);
            }
        }

        for (int x = 0; x < ((vis_dims[n][1] + 1) / 2); x++)
        {
            for (int y = 0; y < ((vis_dims[n][0] + 1) / 2); y++)
            {
                nlohmann::json bin_json_half;
                bin_json_half["x"] = x;
                bin_json_half["y"] = y;
                bin_json_half["sum"] = matrixBinsHalf[n][x][y].nonzeroCount;
                bin_json_half["totalValues"] = matrixBinsHalf[n][x][y].totalValues;
                bin_json_half["absTotalValues"] = matrixBinsHalf[n][x][y].absTotalValues;
                json_arr_half.push_back(bin_json_half);
            }
        }

        for (int x = 0; x < (vis_dims[n][1] * 2); x++)
        {
            for (int y = 0; y < (vis_dims[n][0] * 2); y++)
            {
                nlohmann::json bin_json_double;
                bin_json_double["x"] = x;
                bin_json_double["y"] = y;
                bin_json_double["sum"] = matrixBinsDouble[n][x][y].nonzeroCount;
                bin_json_double["totalValues"] = matrixBinsDouble[n][x][y].totalValues;
                bin_json_double["absTotalValues"] = matrixBinsDouble[n][x][y].absTotalValues;
                json_arr_double.push_back(bin_json_double);
            }
        }

        sort(nonZeros.begin(), nonZeros.end());
        if (nonZeros.size() % 2 == 0)
        {  // Even number of elements
            stats[n].median_nnz = (nonZeros[nonZeros.size() / 2 - 1] + nonZeros[nonZeros.size() / 2]) / 2.0;
        }
        else
        {  // Odd number of elements
            stats[n].median_nnz = nonZeros[nonZeros.size() / 2];
        }
        stats[n].geo_mean_nnz = stats[n].geo_mean_nnz / (stats[n].no_bins - stats[n].no_empty_bins);
        stats[n].geo_mean_nnz = exp(stats[n].geo_mean_nnz);
        stats[n].mean_nnz = (double(nnzs[n])) / (stats[n].no_bins - stats[n].no_empty_bins);
        //std::cout << (double(nnzs[n])) << " !!! " << stats[n].no_bins << " !!! " << stats[n].no_empty_bins << std::endl;

        std::string json_str2 = json_arr.dump();
        json_str2 = escapeSingleQuote(json_str2);

        std::string json_str2_half = json_arr_half.dump();
        json_str2_half = escapeSingleQuote(json_str2_half);

        std::string json_str2_double = json_arr_double.dump();
        json_str2_double = escapeSingleQuote(json_str2_double);

        bool is_red = true;
        if (CHART_TYPE == "ABS") {
            is_red = false;
        }

            html_file << "<div id='myDiv" << n << "' style='width:600px; height:500px;'></div>\n";
            html_file << R"(
            <script>
            try {
                var jsonData2 = ')" << json_str2 << R"(';  // insert your JSON string here
                var data2 = JSON.parse(jsonData2);

                var x2 = [];
                var y2 = [];
                var z2 = [];
                var text = [];

                for (var i = 0; i < data2.length; i++) {
                    x2.push(data2[i].x *)" << dims[n][0] / vis_dims[n][0] << R"( );
                    y2.push(data2[i].y *)" << dims[n][1] / vis_dims[n][1] << R"( );)";
        if (CHART_TYPE == "NNZ")
        {
            html_file << R"(z2.push(data2[i].sum);)";
        }
        else if (CHART_TYPE == "ABS")
        {
            html_file << R"(z2.push(data2[i].absTotalValues);)";
        }
        else
        {   
            html_file << R"(z2.push(data2[i].sum / data2[i].sum);)";
        }

        html_file << R"(text.push(
                            "x: " + data2[i].x *)"
                      << dims[n][0] / vis_dims[n][0] << R"( +
                            "<br>y: " + data2[i].y *)"
                      << dims[n][1] / vis_dims[n][1] << R"( +
                            "<br>NonZero: " + data2[i].sum +
                            "<br>Total Values: " + data2[i].totalValues +
                            "<br>Abs Total Values: " + data2[i].absTotalValues
                            );
                }

                // Creating the heatmap
                var trace2 = {
                x: x2,
                y: y2,
                z: z2,
                visible: true, // initially  shown
                type: 'heatmap',
                colorscale: )" << calculateColorscale(EXPONENTIAL_COLORSCALE, is_red) << R"(,
                showscale: true,
                text: text,
                hoverinfo: 'text'  // Only show the custom hover text
                };

                var jsonData2_half = ')" << json_str2_half << R"(';  // insert your JSON string here
                var data3 = JSON.parse(jsonData2_half);

                var x2h = [];
                var y2h = [];
                var z2h = [];
                var texth = [];

                for (var i = 0; i < data3.length; i++) {
                    x2h.push(data3[i].x *)"
                      << dims[n][0] / ((vis_dims[n][0] + 1) / 2) << R"( );
                    y2h.push(data3[i].y *)"
                      << dims[n][1] / ((vis_dims[n][1] + 1) / 2) << R"( );)";
        if (CHART_TYPE == "NNZ")
        {
            html_file << R"(z2h.push(data3[i].sum);)";
        }
        else if (CHART_TYPE == "ABS")
        {
            html_file << R"(z2h.push(data3[i].absTotalValues);)";
        }
        else
        {
            html_file << R"(z2h.push(data3[i].sum / data3[i].sum);)";
        }
        html_file << R"(texth.push(
                            "x: " + data3[i].x *)"
                      << dims[n][0] / ((vis_dims[n][0] + 1) / 2) << R"( +
                            "<br>y: " + data3[i].y *)"
                      << dims[n][1] / ((vis_dims[n][1] + 1) / 2) << R"( +
                            "<br>NonZero: " + data3[i].sum +
                            "<br>Total Values: " + data3[i].totalValues +
                            "<br>Abs Total Values: " + data3[i].absTotalValues
                            );
                }

                var trace3 = {
                x: x2h,
                y: y2h,
                z: z2h,
                visible: false, // initially not shown
                type: 'heatmap',
                colorscale: )" << calculateColorscale(EXPONENTIAL_COLORSCALE, is_red) << R"(,
                showscale: true,
                text: texth,
                hoverinfo: 'text'  // Only show the custom hover text
                };

                var jsonData2_double = ')" << json_str2_double << R"(';  // insert your JSON string here
                var data4 = JSON.parse(jsonData2_double);

                var x2d = [];
                var y2d = [];
                var z2d = [];
                var textd = [];

                for (var i = 0; i < data4.length; i++) {
                    x2d.push(data4[i].x *)"
                      << (dims[n][0] / (2 * vis_dims[n][0])) << R"( );
                    y2d.push(data4[i].y *)"
                      << (dims[n][1] / (2 * vis_dims[n][1])) << R"( );)";
        if (CHART_TYPE == "NNZ")
        {
            html_file << R"(z2d.push(data4[i].sum);)";
        }
        else if (CHART_TYPE == "ABS")
        {
            html_file << R"(z2d.push(data4[i].absTotalValues);)";
        }
        else
        {
            html_file << R"(z2d.push(data4[i].sum / data4[i].sum );)";
        }
        html_file << R"(textd.push(
                            "x: " + data4[i].x *)"
                      << (dims[n][0] / (2 * vis_dims[n][0])) << R"( +
                            "<br>y: " + data4[i].y *)"
                      << (dims[n][1] / (2 * vis_dims[n][1])) << R"( +
                            "<br>NonZero: " + data4[i].sum +
                            "<br>Total Values: " + data4[i].totalValues +
                            "<br>Abs Total Values: " + data4[i].absTotalValues
                            );
                }

                var trace4 = {
                x: x2d,
                y: y2d,
                z: z2d,
                visible: false, // initially not shown
                type: 'heatmap',
                colorscale: )"
                      << calculateColorscale(EXPONENTIAL_COLORSCALE, is_red)
                      << R"(,
                showscale: true,
                text: textd,
                autosize: 'false',
                hoverinfo: 'text'  // Only show the custom hover text
                };

                var layout2 = {
                plot_bgcolor: '#FFFFFF',
                autosize: false,
                width: 500,
                yaxis: {
                    autorange: 'reversed', // This is for adjusting the main diagonal correctly
                    scaleanchor: 'x',
                    type: 'category',
                    constrain: 'domain'

                },
                xaxis: {
                    side: 'top',
                    type: 'category',
                    constrain: 'domain'
                },
                height: 500,
                sliders: [{
                    active: 1,
                    direction: 'center',
                    showactive: true,
                    currentvalue: {
                    xanchor: 'center',
                    yanchor: 'bottom',
                    prefix: 'Resolution: ',
                    font: {
                    color: '#888',
                    size: 10
                    }
                    },
                    steps: [{
                    label: '50\%',
                    method: 'restyle',
                    args: ['visible', [false, true, false]],
                    }, {
                    label: '100\%',
                    method: 'restyle',
                    args: ['visible', [true, false, false]],
                    }, {
                    label: '200\%',
                    method: 'restyle',
                    args: ['visible', [false, false, true]],
                    }]
                }]
                };
                var data2 = [trace2, trace3, trace4];

                Plotly.newPlot('myDiv)" +
                             std::to_string(n) +
                             R"(', data2, layout2);
                } catch (err) {
                    console.log('Error parsing JSON or plotting data: ' + err.message);
                }
                </script>
                )";
          
        if (CHART_TYPE != "ABS")
        {
            std::vector<std::pair<std::string, int>> sorted_bins;
            for (size_t i = 0; i < x_bins_json.size(); i++) {
                sorted_bins.push_back(make_pair(x_bins_json[i].get<std::string>(), y_nonzero_json[i]));
            }
            sort(sorted_bins.begin(), sorted_bins.end(),
                 [](const std::pair<std::string, int> &a, const std::pair<std::string, int> &b) {
                     return a.second > b.second; // change this to a.second < b.second;
                     // for ascending order
                 });
            nlohmann::json sorted_x_bins_json, sorted_y_nonzero_json;
            for (const auto &bin : sorted_bins) {
                sorted_x_bins_json.push_back(bin.first);
                sorted_y_nonzero_json.push_back(bin.second);
            }
            nlohmann::json sorted_data_hist;
            sorted_data_hist["x"] = sorted_x_bins_json;
            sorted_data_hist["y"] = sorted_y_nonzero_json;

            std::string histogramStr = sorted_data_hist.dump();
            html_file << "<div id='histogram" << n
                      << "' style='width:450px; height:300px;'></div>\n";
            html_file << R"(
            <script>
            try {
                var histogramData = ')" + histogramStr + R"(';  // insert your histogram JSON string here
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
                title: 'Bin NNZCount',
                font: {
                color: '#000000',
                size: 10
                },
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
                width: 300,
                height: 300,
                margin: {
                l: 50,
                r: 20,
                b: 50,
                t: 50
                }
                };

                var data3 = [trace3];

                Plotly.newPlot('histogram)" +
                             std::to_string(n) + R"(', data3, layout3);
            } catch (err) {
                console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }
        else if (CHART_TYPE == "ABS")
        {
            std::vector<std::pair<std::string, double>> sorted_abs_bins; // Note: Using double for absTotalValues
            for (size_t i = 0; i < x_bins_abs_json.size(); i++) {
                sorted_abs_bins.push_back(make_pair(x_bins_abs_json[i].get<std::string>(), y_absTotalValues_json[i]));
            }
            sort(sorted_abs_bins.begin(), sorted_abs_bins.end(),
                [](const std::pair<std::string, double> &a,
                    const std::pair<std::string, double> &b)
                {
                    return a.second > b.second; // for descending order
                });
            nlohmann::json sorted_x_bins_abs_json, sorted_y_absTotalValues_json;
            for (const auto &bin : sorted_abs_bins)
            {
                sorted_x_bins_abs_json.push_back(bin.first);
                sorted_y_absTotalValues_json.push_back(bin.second);
            }
            nlohmann::json sorted_data_abs_hist;
            sorted_data_abs_hist["x"] = sorted_x_bins_abs_json;
            sorted_data_abs_hist["y"] = sorted_y_absTotalValues_json;
            // using the sorted_data_hist
            std::string histogramAbsStr = sorted_data_abs_hist.dump();
            html_file << "<div id='histogramAbs" << n
                      << "' style='width:450px; height:300px;'></div>\n";
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
                opacity: 0.5,
                };

                var layoutAbs = {
                title: 'Bin Absolute Values',
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
                width: 350,
                height: 300,
                margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50
                }
                };

                var dataAbsPlot = [traceAbs];

                Plotly.newPlot('histogramAbs)" +
                             std::to_string(n) + R"(', dataAbsPlot, layoutAbs);
            } catch (err) {
                console.log('Error parsing JSON or plotting data: ' + err.message);
            }
            </script>
            )";
        }
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

    // CleanUp
    for (int n = 0; n < norder; n++)
    {
        for (int i = 0; i < vis_dims[n][1]; i++)
        {
            delete[] matrixBins[n][i];
        }
        delete[] matrixBins[n];
    }
    delete[] matrixBins;
    for (int n = 0; n < norder; n++)
    {
        for (int i = 0; i < ((vis_dims[n][1] + 1) / 2); i++)
        {
            delete[] matrixBinsHalf[n][i];
        }
        delete[] matrixBinsHalf[n];
    }
    delete[] matrixBinsHalf;
    for (int n = 0; n < norder; n++)
    {
        for (int i = 0; i < (vis_dims[n][1] * 2); i++)
        {
            delete[] matrixBinsDouble[n][i];
        }
        delete[] matrixBinsDouble[n];
    }
    delete[] matrixBinsDouble;

    delete[] stats;
    delete[] minX;
    delete[] maxX;
    delete[] colLastSeen;
    delete[] rowPerm;
    delete[] colDegrees;
}
