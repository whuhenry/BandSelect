#include "SpectralClusterSelector.h"
#include <device_launch_parameters.h>
#include <stdio.h>

void compute_similar_matrix_gpu(unsigned short *h_img_data, double *h_similar_matrix,
                                                         BandInfo* h_bands_info, int pixel_count, int band_count) {
    dim3 thread_size(32, 32);
    dim3 block_size;
    block_size.x = (band_count + thread_size.x - 1) / thread_size.x;
    block_size.y = (band_count + thread_size.y - 1) / thread_size.y;
    unsigned short *d_img_data;
    double *d_similar_matrix;
    BandInfo* d_bands_info;
    cudaMalloc(&d_img_data, pixel_count* band_count * sizeof(unsigned short));
    cudaMemcpy(d_img_data, h_img_data, pixel_count* band_count * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMalloc(&d_similar_matrix, band_count * band_count * sizeof(double));
    cudaMalloc(&d_bands_info, band_count * sizeof(BandInfo));
    cudaMemcpy(d_bands_info, h_bands_info, band_count * sizeof(BandInfo), cudaMemcpyHostToDevice);
    compute_similar_matrix_kernel<<<block_size, thread_size>>>(d_img_data, d_similar_matrix, d_bands_info,
                                                               pixel_count, band_count);
}


__global__ void compute_similar_matrix_kernel(unsigned short *img_data,
                                       double *similar_matrix,
                                       BandInfo* bands_info,
                                       int pixel_count, int band_count) {
    int band_x = blockIdx.x * blockDim.x + threadIdx.x;
    int band_y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("x = %d, y = %d \n", blockIdx.x, blockIdx.y);
    if (band_x >= band_count || band_y >= band_count || band_x > band_y) {
        return;
    }
    int count_x = bands_info[band_x].max - bands_info[band_x].min + 1;
    int count_y = bands_info[band_y].max - bands_info[band_y].min + 1;
    int *histogram_2d = new int[count_x * count_y];
    int h_x, h_y;
    for (int pixel_idx = 0; pixel_idx < pixel_count; ++pixel_idx) {
        h_x = img_data[band_x * pixel_count + pixel_idx] - bands_info[band_x].min;
        h_y = img_data[band_y * pixel_count + pixel_idx] - bands_info[band_y].min;
        ++histogram_2d[h_x * count_y + h_y];
    }
    double joint_entropy;
    for (int x_idx = 0; x_idx < count_x; ++x_idx) {
        for (int y_idx = 0; y_idx < count_y; ++y_idx) {
            double probability = (double)histogram_2d[x_idx * count_y + y_idx] / (double)pixel_count;
            joint_entropy -= probability * log2(probability);
        }
    }

    similar_matrix[band_y * band_count + band_x]
            = bands_info[band_x].entropy + bands_info[band_x].entropy - joint_entropy;
    similar_matrix[band_x * band_count + band_y] = similar_matrix[band_y * band_count + band_x];
}