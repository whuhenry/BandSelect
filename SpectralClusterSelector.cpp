//
// Created by henry on 18-4-12.
//

#include "SpectralClusterSelector.h"

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/log/trivial.hpp>
#include <random>

#include "gpu_mi.h"
#include <complex.h>


void SpectralClusterSelector::solve() {
    BOOST_LOG_TRIVIAL(info) << "load data start";
    load_data();
    BOOST_LOG_TRIVIAL(info) << "load data finished";

    //compute similar matrix

    //int pixel_count = rows_ * cols_;
    //int h_x, h_y;
    //double minmaxi[2], minmaxj[2];
    //for (int i = 0; i < bands_; ++i) {
    //    dataset_->GetRasterBand(i + 1)->ComputeRasterMinMax(FALSE, minmaxi);
    //    int count_x = minmaxi[1] - minmaxi[0] + 1;
    //    BOOST_LOG_TRIVIAL(info) << i << " bands start";
    //    for (int j = i + 1; j < bands_; ++j) {
    //        dataset_->GetRasterBand(j + 1)->ComputeRasterMinMax(FALSE, minmaxj);
    //        int count_y = minmaxj[1] - minmaxj[0] + 1;
    //        int *histogram_2d = new int[count_x * count_y];
    //        for(int k = 0; k < count_y * count_x; ++k)
    //        {
    //            histogram_2d[k] = 0;
    //        }
    //        for (int pixel_idx = 0; pixel_idx < pixel_count; ++pixel_idx) {
    //            h_x = img_data_[i * pixel_count + pixel_idx] - minmaxi[0];
    //            h_y = img_data_[j * pixel_count + pixel_idx] - minmaxj[0];
    //            ++histogram_2d[h_x * count_y + h_y];
    //        }
    //        double joint_entropy = 0;
    //        for (int x_idx = 0; x_idx < count_x; ++x_idx) {
    //            for (int y_idx = 0; y_idx < count_y; ++y_idx) {
    //                if (0 != histogram_2d[x_idx * count_y + y_idx]) {
    //                    double probability = histogram_2d[x_idx * count_y + y_idx] / (double)pixel_count;
    //                    joint_entropy -= probability * log2(probability);
    //                }
    //            }
    //        }
    //        delete[] histogram_2d;
    //        histogram_2d = nullptr;
    //        //similar_matrix(i, j) = bands_info_[i].entropy + bands_info_[j].entropy - joint_entropy;
    //        //diag_matrix(i, i) += similar_matrix(i, j);
    //    }
    //    BOOST_LOG_TRIVIAL(info) << i << " bands finished";
    //}
//    Eigen::MatrixXd laplacian_matrix = diag_matrix - similar_matrix;

    double* similar_matrix_buf = new double[bands_ * bands_];
    if (true) {
        BOOST_LOG_TRIVIAL(info) << "cuda start";
        compute_similar_matrix_gpu(img_data_, similar_matrix_buf, rows_, cols_, bands_);
        BOOST_LOG_TRIVIAL(info) << "cuda finished";

        std::ofstream ofs;
        ofs.open("tmp_s_matrix", std::iostream::out);
        if (ofs.is_open()) {
            for (int i = 0; i < bands_ * bands_; ++i) {
                ofs << similar_matrix_buf[i] << std::endl;
            }
        }
        ofs.close();
    } else {
        std::ifstream ifs;
        ifs.open("tmp_s_matrix", std::iostream::in);
        if (ifs.is_open()) {
            for (int i = 0; i < bands_ * bands_; ++i) {
                ifs >> similar_matrix_buf[i];
            }
        }
        ifs.close();
    }

    int valid_band_count = 0;
    double valid_threads_hold = 8.0;
    for (int i = 0; i < bands_; ++i)
    {
        if(similar_matrix_buf[i * bands_ + i] > valid_threads_hold)
        {
            valid_band_count++;
        }
    }

    Eigen::MatrixXd similar_matrix(valid_band_count, valid_band_count);
    similar_matrix.setZero();
    Eigen::MatrixXd diag_matrix(valid_band_count, valid_band_count);
    diag_matrix.setZero();
    for (int i = 0; i < valid_band_count; ++i) {
        if(similar_matrix_buf[i * bands_ + i] <= valid_threads_hold)
        {
            continue;
        }
        for (int j = 0; j < valid_band_count; ++j) {
            if (similar_matrix_buf[j * bands_ + j] <= valid_threads_hold)
            {
                continue;
            }
            similar_matrix(i, j) = similar_matrix_buf[bands_ * i + j];
        }
    }

    //int count[65536] = {0};
    //for(int i = 0; i < rows_ * cols_; ++i)
    //{
    //    ++count[img_data_[i]];
    //}
    //double entropy = 0;
    //for(int i = 0; i < 65536; ++i)
    //{
    //    if(count[i] != 0)
    //    {
    //        double p = count[i] / (double)(rows_ * cols_);
    //        entropy -= p * log2(p);
    //    }
    //}

    //std::cout << similar_matrix << std::endl;
    for (int i = 0; i < valid_band_count; ++i) {
        for (int j = 0; j < valid_band_count; ++j) {
            similar_matrix(i, j) = similar_matrix(i, i) + similar_matrix(j, j) - similar_matrix(i, j);
            diag_matrix(i, i) += similar_matrix(i, j);
            if(similar_matrix(i, j) < 0)
            {
                break;
            }
        }
    }
    Eigen::MatrixXd laplacian_matrix = diag_matrix - similar_matrix;

    Eigen::EigenSolver<Eigen::MatrixXd> es(laplacian_matrix);
    auto eigen_value_mat = es.eigenvalues();
    auto eigen_vector_mat = es.eigenvectors();
    std::vector<std::pair<double, Eigen::VectorXd>> eigen_result;
    for (int i = 0; i < eigen_value_mat.rows(); ++i) {
        Eigen::VectorXd eigen_vector = eigen_vector_mat.col(i).real();
        eigen_result.push_back(std::make_pair(eigen_value_mat[i].real(), eigen_vector));
    }
    std::nth_element(eigen_result.begin(), eigen_result.begin() + selected_count_, eigen_result.end(),
                     [](const std::pair<double, Eigen::VectorXd>& val1, const std::pair<double, Eigen::VectorXd> &val2)
                     {
                         return val1.first < val2.first;
                     });

//    for (int i = 0; i < selected_count_; ++i) {
//        std::cout << eigen_result[i].first << " " << eigen_result[i].second.transpose() << std::endl;
//    }

    std::vector<Eigen::VectorXd> data;
    for (int i = 0; i < valid_band_count; ++i) {
        Eigen::VectorXd tmp_data(selected_count_);
        for (int j = 0; j < selected_count_; ++j) {
            tmp_data[j] = eigen_result[j].second[i];
        }
        tmp_data.normalize();
        data.push_back(tmp_data);
    }

    std::vector<int> label;
    std::vector<Eigen::VectorXd> center;
    kmeans(data, label, center);
    for (int i = 0; i < selected_count_; ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int band_idx = -1;
        for(int k = 0; k < valid_band_count; ++k)
        {
            if(label[k] == i)
            {
                double distance = (data[k] - center[i]).norm();
                if(distance < min_distance)
                {
                    min_distance = distance;
                    band_idx = k;
                }
            }
        }
        std::cout << "label :" << i << "; band_idx :" << band_idx << std::endl;
    }

    if (nullptr != img_data_) {
        delete[] img_data_;
        img_data_ = nullptr;
    }

    GDALClose(dataset_);
    dataset_ = nullptr;
}

void SpectralClusterSelector::load_data() {
    GDALAllRegister();
    dataset_ = (GDALDataset*)GDALOpen(src_img_path_.c_str(), GA_ReadOnly);
    rows_ = dataset_->GetRasterYSize();
    cols_ = dataset_->GetRasterXSize();
    bands_ = dataset_->GetRasterCount();
    img_data_ = new uint16_t[rows_ * cols_ * bands_];

    std::string interleave = dataset_->GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE");
    if (interleave == "LINE") {
        //read method for BIL image
//        uint16_t *tmp_buf = new uint16_t[cols_ * bands_];
//        for (int i = 0; i < rows_; ++i){
//            dataset_->RasterIO(GF_Read, 0, i, cols_, 1, tmp_buf, cols_, 1, GDT_UInt16, bands_, nullptr, 0, 0, 0);
//            for (int k = 0; k < bands_; ++k) {
//                for (int j = 0; j < cols_; ++j) {
//                    img_data_[k * rows_ * cols_ + i * cols_ + j] = tmp_buf[k * cols_ + j];
//                }
//            }
//        }
//        delete[] tmp_buf;
//        tmp_buf = nullptr;
        dataset_->RasterIO(GF_Read, 0, 0, cols_, rows_, img_data_, cols_, rows_, GDT_UInt16, bands_, nullptr, 0, 0, 0);
    } else if (interleave == "BAND") {
        //read method for BSQ image
        dataset_->RasterIO(GF_Read, 0, 0, cols_, rows_, img_data_, cols_, rows_, GDT_UInt16, bands_, nullptr, 0, 0, 0);
    }

}

void SpectralClusterSelector::kmeans(std::vector<Eigen::VectorXd> data,
                                     std::vector<int>& out_label,
                                     std::vector<Eigen::VectorXd>& out_center) {
    int valid_band_count = data.size();
    out_label.resize(valid_band_count);

    //choose center random
    std::default_random_engine generator(clock());
    std::uniform_int_distribution<int> distribution(0, valid_band_count - 1);
    out_center.push_back(data[distribution(generator)]);
    double sum_distance, max_sum_distance = -1.0;
    int center_idx;
    bool *selected = new bool[valid_band_count];
    for(int i = 0; i< valid_band_count; ++i)
    {
        selected[i] = false;
    }
    for (int i = 1; i < selected_count_; ++i) {
        center_idx = -1;
        max_sum_distance = -1.0;
        for (int j = 0; j < valid_band_count; ++j) {
            if(selected[j])
            {
                continue;
            }
            sum_distance = 0;
            for (int k = 0; k < i; ++k) {
                sum_distance += (data[j] - out_center[k]).norm();
            }
            if (sum_distance > max_sum_distance) {
                max_sum_distance = sum_distance;
                center_idx = j;
            }
        }
        out_center.push_back(data[center_idx]);
        selected[center_idx] = true;
    }

    delete[] selected;

    std::vector<double> delta_center;
    delta_center.resize(selected_count_);

    int mark, iter_num = 0;
    double min_distance, max_delta;
    do {
        for(int i = 0; i < valid_band_count; ++i) {
            mark = -1;
            min_distance = std::numeric_limits<double>::max();
            for (int j = 0; j < selected_count_; ++j) {
                double distance = (data[i] - out_center[j]).norm();
                if (distance < min_distance) {
                    min_distance = distance;
                    mark = j;
                }
            }
            out_label[i] = mark;
        }

        std::vector<Eigen::VectorXd> new_center;
        std::vector<int> label_count;
        for (int i = 0; i < selected_count_; ++i) {
            Eigen::VectorXd tmp_center(selected_count_);
            tmp_center.setZero();
            new_center.push_back(tmp_center);
            label_count.push_back(0);
        }
        for (int i = 0; i < valid_band_count; ++i) {
            new_center[out_label[i]] += data[i];
            ++label_count[out_label[i]];
        }

        max_delta = -1.0;
        for (int i = 0; i < selected_count_; ++i) {
            new_center[i] /= label_count[i];
            double tmp_dis = (new_center[i] - out_center[i]).norm();
            if (tmp_dis > max_delta) {
                max_delta = tmp_dis;
            }
        }
        out_center.swap(new_center);
        ++iter_num;
        if (iter_num > 20) {
            break;
        }
    } while( max_delta > 0.000000001);
}
