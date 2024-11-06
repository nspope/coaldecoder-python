#ifndef ARRAY_CONVERSION_HPP
#define ARRAY_CONVERSION_HPP
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <vector>

namespace py = pybind11;

template <typename T>
using pyarr = py::array_t<T, py::array::f_style | py::array::forcecast>;

template <typename T>
inline
arma::Cube<T> to_cube(pyarr<T> inp)
{
    py::buffer_info info = inp.request();
    if(info.ndim != 3) { throw pybind11::index_error("Incorrect shape for conversion to arma::Cube"); }
    return arma::Cube<T>(reinterpret_cast<T*>(info.ptr), info.shape[0], info.shape[1], info.shape[2]);
}

template <typename T>
inline
arma::Mat<T> to_mat(pyarr<T> inp)
{
    py::buffer_info info = inp.request();
    if(info.ndim != 2) { throw pybind11::index_error("Incorrect shape for conversion to arma::Mat"); }
    return arma::Mat<T>(reinterpret_cast<T*>(info.ptr), info.shape[0], info.shape[1]);
}

template <typename T>
inline
arma::Col<T> to_vec(pyarr<T> inp)
{
    py::buffer_info info = inp.request();
    if(info.ndim != 1) { throw pybind11::index_error("Incorrect shape for conversion to arma::Col"); }
    return arma::Col<T>(reinterpret_cast<T*>(info.ptr), info.shape[0]);
}


template <typename T>
inline
pyarr<T> from_cube(arma::Cube<T> inp)
{
    std::vector<py::ssize_t> shape = { (int) inp.n_rows, (int) inp.n_cols, (int) inp.n_slices };
    std::vector<py::ssize_t> stride = { (int) sizeof(T), (int) (sizeof(T) * inp.n_rows), (int) (sizeof(T) * inp.n_rows * inp.n_cols) };
    std::string format = py::format_descriptor<T>::format();
    py::ssize_t ndim = 3;
    py::ssize_t size = sizeof(T);
    py::buffer_info buffer(inp.memptr(), size, format, ndim, shape, stride);
    return pyarr<T>(buffer);
}


template <typename T>
inline
pyarr<T> from_mat(arma::Mat<T> inp)
{
    std::vector<py::ssize_t> shape = { (int) inp.n_rows, (int) inp.n_cols };
    std::vector<py::ssize_t> stride = { (int) sizeof(T), (int) (sizeof(T) * inp.n_rows) };
    std::string format = py::format_descriptor<T>::format();
    py::ssize_t ndim = 2;
    py::ssize_t size = sizeof(T);
    py::buffer_info buffer(inp.memptr(), size, format, ndim, shape, stride);
    return pyarr<T>(buffer);
}

template <typename T>
inline
pyarr<T> from_vec(arma::Col<T> inp)
{
    std::vector<py::ssize_t> shape = { (int) inp.n_rows };
    std::vector<py::ssize_t> stride = { (int) sizeof(T) };
    std::string format = py::format_descriptor<T>::format();
    py::ssize_t ndim = 1;
    py::ssize_t size = sizeof(T);
    py::buffer_info buffer(inp.memptr(), size, format, ndim, shape, stride);
    return pyarr<T>(buffer);
}

#endif
