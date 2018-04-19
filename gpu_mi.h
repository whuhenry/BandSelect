#include <cuda_runtime.h>

void compute_similar_matrix_gpu(unsigned short *h_img_data, double *h_joint_entropy,
    int rows, int cols, int band_count);

__global__ void compute_joint_histogram(unsigned short *d_img_data, int *d_joint_histogram,
    int band_idx1, int band_idx2, int pixel_count, int cols, int rows, int band_count);

__global__ void compute_joint_entropy(int* d_joint_histogram, double *d_joint_entropy_partial, int pixel_count);

__global__ void sum_array(double* d_array_in, double* d_array_out, int len);