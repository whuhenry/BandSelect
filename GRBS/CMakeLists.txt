include_directories(${PROJECT_SOURCE_DIR}/common)

add_executable(grbs GRBS.cpp)

target_link_libraries(grbs ${GDAL_LIBRARY} Boost::program_options Boost::log)