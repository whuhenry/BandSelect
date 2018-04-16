#include "SpectralClusterSelector.h"
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdio.h>

const int thread_per_dim = 32;
const int thread_per_block = thread_per_dim * thread_per_dim;
const int max_intensity = 4096;
const int max_intensity_square = max_intensity * max_intensity;

void compute_similar_matrix_gpu(unsigned short *h_img_data, double *h_joint_entropy,
                                int rows, int cols, int band_count) {
    dim3 thread_size(thread_per_dim, thread_per_dim);
    dim3 block_size;
    block_size.x = (cols + thread_size.x - 1) / thread_size.x;
    block_size.y = (rows + thread_size.y - 1) / thread_size.y;
    int entropy_block_size = (max_intensity_square + thread_per_block - 1) / thread_per_block;
    unsigned short *d_img_data;
    cudaMalloc(&d_img_data, rows * cols * band_count * sizeof(unsigned short));
    cudaMemcpy(d_img_data, h_img_data, rows * cols * band_count * sizeof(unsigned short), cudaMemcpyHostToDevice);
    int *d_joint_histogram;
    cudaMalloc(&d_joint_histogram, max_intensity * max_intensity * sizeof(int));
    double *d_joint_entropy_partial, entropy_sum;
    cudaMalloc(&d_joint_entropy_partial, max_intensity_square * sizeof(double));

    for(int i = 0; i < band_count; ++i) {
        for (int j = i; j < band_count; ++j) {
            cudaMemset(d_joint_histogram, 0, max_intensity * max_intensity * sizeof(int));
            compute_joint_histogram<<<block_size, thread_size>>>(d_img_data,
                                                                 d_joint_histogram,
                                                                 i, j,
                                                                 rows * cols, cols, rows,
                                                                 band_count);

            compute_joint_entropy<<<entropy_block_size, thread_per_block>>>(d_joint_histogram,
                                                                       d_joint_entropy_partial, rows * cols);

//            double* tmp_buf = new double[max_intensity_square];
//            cudaMemcpy(tmp_buf, d_joint_entropy_partial, max_intensity_square *sizeof(double), cudaMemcpyDeviceToHost);
//            double sum = 0;
//            for(int k = 0; k < max_intensity_square; ++k)
//            {
//                sum += tmp_buf[k];
//            }

            int d_joint_entropy_partial_len = max_intensity_square;
            int grid_dim = entropy_block_size;
            while (grid_dim > 0){
                sum_array<<<grid_dim, thread_per_block>>>(d_joint_entropy_partial,
                        d_joint_entropy_partial,
                        d_joint_entropy_partial_len);

                if (grid_dim == 1) {
                    break;
                }
                d_joint_entropy_partial_len = grid_dim;
                grid_dim = (grid_dim + thread_per_block - 1) / thread_per_block;
            };

            cudaMemcpy(&entropy_sum, d_joint_entropy_partial, sizeof(double), cudaMemcpyDeviceToHost);

            h_joint_entropy[i * band_count + j] = h_joint_entropy[j * band_count + i] = entropy_sum;

        }
    }

    cudaFree(d_joint_entropy_partial);
    cudaFree(d_joint_histogram);
    cudaFree(d_img_data);
}

__global__ void compute_joint_histogram(unsigned short *d_img_data, int *d_joint_histogram,
                                        int band_idx1, int band_idx2, int pixel_count, int cols, int rows, int band_count) {
    int pixel_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_idx_x < cols && pixel_idx_y < rows) {
        unsigned short pixel_idx1 = d_img_data[band_idx1 * pixel_count + pixel_idx_y * cols + pixel_idx_x];
        unsigned short pixel_idx2 = d_img_data[band_idx2 * pixel_count + pixel_idx_y * cols + pixel_idx_x];
        atomicAdd(d_joint_histogram + pixel_idx1 * band_count + pixel_idx2, 1);
    }

}

__global__ void compute_joint_entropy(int* d_joint_histogram, double *d_joint_entropy_partial, int pixel_count) {
    int his_idx = blockIdx.x * thread_per_block + threadIdx.x;

    if (his_idx < max_intensity_square && 0 != d_joint_histogram[his_idx]) {
        double probability = d_joint_histogram[his_idx] / (double)pixel_count;
        d_joint_entropy_partial[his_idx] = -probability * log2(probability);
    } else {
        d_joint_entropy_partial[his_idx] = 0.0;
    }
}

__global__ void sum_array(double* d_array_in, double* d_array_out, int len) {
    __shared__ double partial_sum[thread_per_block];
    int tid = threadIdx.x;
    int array_idx = blockIdx.x * blockDim.x + tid;

    if (array_idx > len) {
       partial_sum[tid] = 0.0;
    } else {
        partial_sum[tid] = d_array_in[array_idx];
    }
    __syncthreads();

    for(int stride = thread_per_block / 2; stride > 0; stride/=2)
    {
        if(tid < stride) {
            partial_sum[tid]+= partial_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_array_out[blockIdx.x] = partial_sum[0];
    }

}