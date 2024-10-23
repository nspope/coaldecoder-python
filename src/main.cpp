#include "convert.hpp"

// --- test
//
pyarr<double> add_cube(const pyarr<double> a, const pyarr<double> b) {
    arma::cube A = to_cube<double>(a);
    arma::cube B = to_cube<double>(b);
    arma::cube C = A + B;
    return from_cube<double>(C);
}

pyarr<double> add_mat(const pyarr<double> a, const pyarr<double> b) {
    arma::mat A = to_mat<double>(a);
    arma::mat B = to_mat<double>(b);
    arma::mat C = A + B;
    return from_mat<double>(C);
}

pyarr<double> add_col(const pyarr<double> a, const pyarr<double> b) {
    arma::vec A = to_vec<double>(a);
    arma::vec B = to_vec<double>(b);
    arma::vec C = A + B;
    return from_vec<double>(C);
}


PYBIND11_MODULE(cmake_example, m) {
    m.def("add_cube", &add_cube);
    m.def("add_mat", &add_mat);
    m.def("add_col", &add_col);
}
