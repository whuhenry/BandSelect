#include <iostream>
#include <boost/program_options.hpp>

#include "SpectralClusterSelector.h"

namespace po = boost::program_options;
using namespace std;

int main(int argc, char* argv[]) {
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("src", po::value<string>(), "Source Hyperspectral Image path")
            ("dst", po::value<string>(), "Destination Hyperspectral Image path")
            ("band_count,K", po::value<int>(), "Selected band count")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    SpectralClusterSelector spectral_cluster_selector;

    if (vm.count("band_count")) {
        spectral_cluster_selector.selected_count_ = vm["band_count"].as<int>();
    } else {
        cout << "Selected band count was not set.\n";
    }

    if (vm.count("src")) {
        spectral_cluster_selector.src_img_path_ = vm["src"].as<string>();
    } else {
        cout << "Source Hyperspectral Image path was not set.\n";
    }

    if (vm.count("dst")) {
        spectral_cluster_selector.dst_img_path_ = vm["dst"].as<string>();
    } else {
        cout << "Source Hyperspectral Image path was not set.\n";
    }

    spectral_cluster_selector.solve();

    return 0;
}