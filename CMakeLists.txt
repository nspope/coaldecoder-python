cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(extern/armadillo)
add_subdirectory(extern/pybind11)

project(coaldecoder)

pybind11_add_module(_coaldecoder src/coalescent_model.cpp)
target_include_directories(_coaldecoder PRIVATE extern/armadillo/include)
target_link_libraries(_coaldecoder PUBLIC armadillo)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
#target_compile_definitions(cmake_example PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
