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

#endif //BANDSELECT_SPECTRALCLUSTERSELECTOR_H
