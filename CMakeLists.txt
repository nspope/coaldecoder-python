cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(extern/armadillo)
add_subdirectory(extern/pybind11)

project(coaldecoder)

#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)

#pybind11_add_module(arma_converter_test src/coalescent_model.cpp)
#target_include_directories(arma_converter_test PRIVATE extern/armadillo-code/include)

pybind11_add_module(coaldecoder src/coalescent_model.cpp)
target_include_directories(coaldecoder PRIVATE extern/armadillo/include)
target_link_libraries(coaldecoder PUBLIC armadillo)
#target_link_libraries(coaldecoder PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
#target_compile_definitions(cmake_example PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
