/**
 * implementation of band selection algorithm proposed by Sun, Kang et.al.
 * detail description of the algorithm can be found in the paper:
 * "A robust and efficient band selection method using graph representation for hyperspectral imagery"
 * ****************************************************************************
 * Created by Yanzhao
 * date: 2018-5-10
 */

#include <iostream>
#include <gdal_priv.h>
#include <boost/program_options.hpp>
#include "common.h"

namespace po = boost::program_options;
using namespace std;

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("src", po::value<string>(), "Source Hyperspectral Image path")
        ("band_count,K", po::value<int>(), "Selected band count")
        ("sigma", po::value<float>(), "parameter sigma")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || 1 == argc) {
        cout << desc << "\n";
        return 1;
    }
    string src_path;
    int selected_band_count;
    float sigma;

    if (vm.count("band_count")) {
        selected_band_count = vm["band_count"].as<int>();
    }
    else {
        cout << "Selected band count was not set.\n";
    }

    if (vm.count("src")) {
        src_path = vm["src"].as<string>();
    }
    else {
        cout << "Source Hyperspectral Image path was not set.\n";
    }

    if (vm.count("sigma")) {
        sigma = vm["sigma"].as<float>();
    }
    else {
        cout << "parameter sigma was not set.\n";
    }

    // read data
    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
    GDALDataset* dataset = (GDALDataset*)GDALOpen(src_path.c_str(), GA_ReadOnly);
    int rows = dataset->GetRasterYSize();
    int cols = dataset->GetRasterXSize();
    int bands = dataset->GetRasterCount();
    uint16_t *data_buf = new uint16_t[rows * cols * bands];
    GDALDataType type = dataset->GetRasterBand(1)->GetRasterDataType();
    dataset->RasterIO(GF_Read, 0, 0, cols, rows, data_buf, cols, rows, GDT_UInt16, bands, nullptr, 0, 0, 0);
    
    //存储每个波段向量的长度
    double *magnitude = new double[bands];
    for (int i = 0; i <bands; ++i) {
        magnitude[i] = 0.0;
        uint16_t* band = data_buf + i * rows * cols;
        for (int pixidx = 0; pixidx < rows * cols; ++pixidx) {
            magnitude[i] += band[pixidx] * band[pixidx];
        }
        magnitude[i] = sqrt(magnitude[i]);
    }

    // compute adjacency mat
    double *adjacency_mat = new double[bands * bands];
    for (int i = 0; i < bands; ++i) {
        adjacency_mat[i * bands + i] = 0;
        uint16_t* band_i = data_buf + i * rows * cols;
        for (int j = i + 1; j < bands; ++j) {
            double correlation = 0.0;
            uint16_t* band_j = data_buf + j * rows * cols;
            for (int pixidx = 0; pixidx < rows * cols; ++pixidx) {
                correlation += band_i[pixidx] * band_j[pixidx];
            }
            correlation = correlation / magnitude[i] / magnitude[j];
            adjacency_mat[i * bands + j] = adjacency_mat[j * bands + i] = exp((correlation - 1) / sigma / sigma);
        }
    }

    //compute degree for each band
    double *degree = new double[bands];
    for (int i = 0; i < bands; ++i) {
        for (int j = 0; j < bands; ++j) {
            degree[i] = adjacency_mat[i * bands + j];
        }
    }

    //SFS solution
    vector<int> selected_bands;
    bool *sel_flag = new bool[bands];
    //first choice
    double max_degree = -1.0;
    int selected_band = -1;
    for (int i = 0; i < bands; ++i) {
        sel_flag[i] = false;
        if(degree[i] > max_degree) {
            max_degree = degree[i];
            selected_band = i;
        }
    }
    selected_bands.push_back(selected_band);
    sel_flag[selected_band] = true;
    double degree_sum = degree[selected_band];
    double adjancency_sum = 0;
    for (int iter = 1; iter < selected_band_count; ++iter) {
        double degree_sum_max = -1.0;
        double adjancency_sum_max = -1.0;
        double criterion_max = -1.0;
        int band_sel = -1;
        for (int band_idx = 0; band_idx < bands; ++band_idx) {
            if (sel_flag[band_idx]) continue;
            //add new band degree and adjancency to the old ones to compute criterion
            double degree_sum_new = degree_sum + degree[band_idx];
            double adjancency_sum_new = adjancency_sum;
            for (int k : selected_bands) {
                adjancency_sum_new += adjacency_mat[band_idx * bands + k];
            }
            double criterion_new = degree_sum_new / adjancency_sum_new;
            //select the max criterion
            if (criterion_new > criterion_max) {
                criterion_max = criterion_new;
                degree_sum_max = degree_sum_new;
                adjancency_sum_max = adjancency_sum_new;
                band_sel = band_idx;
            }
        }
        selected_bands.push_back(band_sel);
        sel_flag[band_sel] = true;
        degree_sum = degree_sum_max;
        adjancency_sum = adjancency_sum_max;
    }

    sort(selected_bands.begin(), selected_bands.end());
    for (int sel_band : selected_bands) {
        cout << sel_band << endl;
    }

    //SBS solution

    release_array(data_buf);
    release_array(adjacency_mat);
    release_array(magnitude);
    release_array(degree);
    release_array(sel_flag);
}