//
// Created by henry on 18-4-12.
// 实现该论文中的算法《基于谱聚类与类间可分性因子的高光谱波段选择》
//

#ifndef BANDSELECT_SPECTRALCLUSTERSELECTOR_H
#define BANDSELECT_SPECTRALCLUSTERSELECTOR_H

#include <string>
#include <vector>
#include <gdal_priv.h>
#include <Eigen/Core>

#include <cuda_runtime.h>

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
    GDALDataset *dataset_;
    void load_data();

    void kmeans(std::vector<Eigen::VectorXd> data, std::vector<int>& out_label, std::vector<Eigen::VectorXd>& out_center);
};



void compute_similar_matrix_gpu(unsigned short *h_img_data, double *h_joint_entropy,
                                int rows, int cols, int band_count);

__global__ void compute_joint_histogram(unsigned short *d_img_data, int *d_joint_histogram,
                                        int band_idx1, int band_idx2, int pixel_count, int cols, int rows, int band_count);

__global__ void compute_joint_entropy(int* d_joint_histogram, double *d_joint_entropy_partial, int pixel_count);

__global__ void sum_array(double* d_array_in, double* d_array_out, int len);

#endif //BANDSELECT_SPECTRALCLUSTERSELECTOR_H
