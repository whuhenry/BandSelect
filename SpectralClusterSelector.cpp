//
// Created by henry on 18-4-12.
//

#include "SpectralClusterSelector.h"

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/log/trivial.hpp>
#include <opencv2/core.hpp>

void SpectralClusterSelector::solve() {
    BOOST_LOG_TRIVIAL(info) << "load data start";
    load_data();
    BOOST_LOG_TRIVIAL(info) << "load data finished";

    //compute similar matrix
//    Eigen::MatrixXd similar_matrix(bands_, bands_);
//    similar_matrix.setZero();
//    Eigen::MatrixXd diag_matrix(bands_, bands_);
//    diag_matrix.setZero();
//    int h_x, h_y;
//    for (int i = 0; i < bands_; ++i) {
//        int count_x = bands_info_[i].max - bands_info_[i].min + 1;
//        BOOST_LOG_TRIVIAL(info) << i << " bands start";
//        for (int j = i + 1; j < bands_; ++j) {
//            int count_y = bands_info_[j].max - bands_info_[j].min + 1;
//            int *histogram_2d = new int[count_x * count_y];
//            for (int pixel_idx = 0; pixel_idx < pixel_count; ++pixel_idx) {
//                h_x = img_data_[i * pixel_count + pixel_idx] - bands_info_[i].min;
//                h_y = img_data_[j * pixel_count + pixel_idx] - bands_info_[j].min;
//                ++histogram_2d[h_x * count_y + h_y];
//            }
//            double joint_entropy = 0;
//            for (int x_idx = 0; x_idx < count_x; ++x_idx) {
//                for (int y_idx = 0; y_idx < count_y; ++y_idx) {
//                    if (0 != histogram_2d[x_idx * count_y + y_idx]) {
//                        double probability = histogram_2d[x_idx * count_y + y_idx] / (double)pixel_count;
//                        joint_entropy -= probability * log2(probability);
//                    }
//                }
//            }
//            delete[] histogram_2d;
//            histogram_2d = nullptr;
//            similar_matrix(i, j) = bands_info_[i].entropy + bands_info_[j].entropy - joint_entropy;
//            diag_matrix(i, i) += similar_matrix(i, j);
//        }
//        BOOST_LOG_TRIVIAL(info) << i << " bands finished";
//    }
//    Eigen::MatrixXd laplacian_matrix = diag_matrix - similar_matrix;

    double* similar_matrix_buf = new double[bands_ * bands_];
    if (false) {
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

    Eigen::MatrixXd similar_matrix(bands_, bands_);
    Eigen::MatrixXd diag_matrix(bands_, bands_);
    diag_matrix.setZero();
    for (int i = 0; i < bands_; ++i) {
        for (int j = 0; j < bands_; ++j) {
            similar_matrix(i, j) = similar_matrix_buf[i * bands_ + j];
        }
    }
    //std::cout << similar_matrix << std::endl;
    for (int i = 0; i < bands_; ++i) {
        for (int j = 0; j < bands_; ++j) {
            similar_matrix(i, j) = similar_matrix(i, i) + similar_matrix(j, j) - similar_matrix(i, j);
            diag_matrix(i, i) += similar_matrix(i, j);
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

    for (int i = 0; i < selected_count_; ++i) {
        std::cout << eigen_result[i].first << " " << eigen_result[i].second.transpose() << std::endl;
    }

    cv::Mat points(bands_, selected_count_, CV_32F);
    cv::Mat best_labels, center;
    for(int i = 0; i < bands_; ++i) {
        for (int j = 0; j < selected_count_; ++j) {
            points.at<float>(i, j) = eigen_result[j].second[i];
        }
    }
    double t = cv::kmeans(points, selected_count_, best_labels,
                          cv::TermCriteria(),
                          10, cv::KMEANS_RANDOM_CENTERS, center);

    std::cout << best_labels;

    if (nullptr != img_data_) {
        delete[] img_data_;
        img_data_ = nullptr;
    }
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

    GDALClose(dataset_);
    dataset_ = nullptr;
}
