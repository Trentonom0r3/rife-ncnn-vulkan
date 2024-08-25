#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "rife.h"

namespace py = pybind11;

// Method to convert a py::array to ncnn::Mat
ncnn::Mat pyarray_to_mat(py::array_t<float>& array);

// Method to convert ncnn::Mat to py::array
py::array_t<float> mat_to_pyarray(const ncnn::Mat& mat);

