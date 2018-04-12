cmake_minimum_required(VERSION 3.5)
project(BandSelect)

set(CMAKE_CXX_STANDARD 11)

find_package(GDAL REQUIRED)
if (GDAL_FOUND)
    include_directories(${GDAL_INCLUDE_DIR})
endif ()

add_executable(BandSelect main.cpp)

target_link_libraries(BandSelect ${GDAL_LIBRARY})