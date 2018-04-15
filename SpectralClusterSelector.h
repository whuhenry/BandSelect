//
// Created by henry on 18-4-12.
// 实现该论文中的算法《基于谱聚类与类间可分性因子的高光谱波段选择》
//

#ifndef BANDSELECT_SPECTRALCLUSTERSELECTOR_H
#define BANDSELECT_SPECTRALCLUSTERSELECTOR_H

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <gdal_priv.h>

struct BandInfo {
    unsigned short min;
    unsigned short max;
    double entropy;
};

class SpectralClusterSelector {
public:
    int selected_count_;
    std::string src_img_path_, dst_img_path_;
    void solve();

private:
    unsigned short *img_data_;
    int rows_, cols_, bands_;
    BandInfo* bands_info_;
    GDALDataset *dataset_;
    void load_data();
};



void compute_similar_matrix_gpu(unsigned short *h_img_data, double *h_similar_matrix,
                                BandInfo* h_bands_info, int rows, int cols, int band_count);

__global__ void compute_joint_histogram(unsigned short *d_img_data, int *d_joint_histogram,
                                        int band_idx1, int band_idx2, int pixel_count, int cols, int rows, int band_count);

__global__ void compute_joint_entropy(int* d_joint_histogram, double *d_joint_entropy_partial);

#endif //BANDSELECT_SPECTRALCLUSTERSELECTOR_H
