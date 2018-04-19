cmake_minimum_required(VERSION 3.5)
project(BandSelect)

set(CMAKE_CXX_STANDARD 11)

find_package(GDAL REQUIRED)
if (GDAL_FOUND)
    include_directories(${GDAL_INCLUDE_DIR})
endif()

find_package(CUDA REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(Boost_USE_STATIC_LIBS        ON)
find_package(Boost REQUIRED COMPONENTS program_options log)

cuda_add_executable(BandSelect main.cpp SpectralClusterSelector.cpp SpectralClusterSelector.h gpu_mi.cu gpu_mi.h)
#add_executable(BandSelect main.cpp SpectralClusterSelector.cpp SpectralClusterSelector.h)

target_link_libraries(BandSelect ${GDAL_LIBRARY} Boost::program_options Boost::log Eigen3::Eigen)