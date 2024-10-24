cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 17)

project(cmake_example)

add_subdirectory(extern/pybind11)
pybind11_add_module(cmake_example src/main.cpp)
pybind11_add_module(coaldecoder src/coalescent_model.cpp)

target_include_directories(cmake_example PRIVATE extern/armadillo-code/include)
target_include_directories(coaldecoder PRIVATE extern/armadillo-code/include)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
#target_compile_definitions(cmake_example PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
