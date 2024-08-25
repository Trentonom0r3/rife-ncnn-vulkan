#include "bindings.h"


// Method to convert a py::array to ncnn::Mat
ncnn::Mat pyarray_to_mat(py::array_t<float>& array) {
    auto buffer_info = array.request();
    int dims = buffer_info.ndim;

    if (dims == 2) {
        return ncnn::Mat((int)buffer_info.shape[1], (int)buffer_info.shape[0], (float*)buffer_info.ptr);
    }
    else if (dims == 3) {
        return ncnn::Mat((int)buffer_info.shape[2], (int)buffer_info.shape[1], (int)buffer_info.shape[0], (float*)buffer_info.ptr);
    }
    else {
        throw std::runtime_error("Unsupported array dimension");
    }
}

// Method to convert ncnn::Mat to py::array
py::array_t<float> mat_to_pyarray(const ncnn::Mat& mat) {
    std::vector<py::ssize_t> shape;
    if (mat.c == 1) {
        shape = { mat.h, mat.w };
    }
    else {
        shape = { mat.c, mat.h, mat.w };
    }

    // Assuming the Mat contains floats
    std::vector<py::ssize_t> strides;
    if (mat.c == 1) {
        strides = { static_cast<py::ssize_t>(sizeof(float) * mat.w), static_cast<py::ssize_t>(sizeof(float)) };
    } else {
        strides = { static_cast<py::ssize_t>(sizeof(float) * mat.h * mat.w), static_cast<py::ssize_t>(sizeof(float) * mat.w), static_cast<py::ssize_t>(sizeof(float)) };
    }
    return py::array(py::buffer_info(
        mat.data, sizeof(float), py::format_descriptor<float>::format(), mat.c == 1 ? 2 : 3, shape, strides));
}

PYBIND11_MODULE(_rife_ncnn_vulkan, m) {
    m.doc() = "Python bindings for RIFE using NCNN Vulkan";

    py::class_<RIFE>(m, "RIFE")
        .def(py::init<int, bool, bool, bool, int, bool, bool>(), 
             py::arg("gpuid"), 
             py::arg("tta_mode") = false, 
             py::arg("tta_temporal_mode") = false, 
             py::arg("uhd_mode") = false, 
             py::arg("num_threads") = 1, 
             py::arg("rife_v2") = false, 
             py::arg("rife_v4") = false)
        .def("load", [](RIFE &self, const std::string &modeldir) {
#if _WIN32
            return self.load(std::wstring(modeldir.begin(), modeldir.end()));
#else
            return self.load(modeldir);
#endif
        }, "Load the model from a directory")
        .def("process", [](RIFE& self, py::array_t<float> in0image, py::array_t<float> in1image, float timestep) {
            ncnn::Mat in0 = pyarray_to_mat(in0image);
            ncnn::Mat in1 = pyarray_to_mat(in1image);
            ncnn::Mat outimage;
            int result = self.process(in0, in1, timestep, outimage);
            return py::make_tuple(result, mat_to_pyarray(outimage));
        }, "Process frames using Vulkan", 
             py::arg("in0image"), 
             py::arg("in1image"), 
             py::arg("timestep"))
        .def("process_cpu", [](RIFE& self, py::array_t<float> in0image, py::array_t<float> in1image, float timestep) {
            ncnn::Mat in0 = pyarray_to_mat(in0image);
            ncnn::Mat in1 = pyarray_to_mat(in1image);
            ncnn::Mat outimage;
            int result = self.process_cpu(in0, in1, timestep, outimage);
            return py::make_tuple(result, mat_to_pyarray(outimage));
        }, "Process frames using CPU", 
             py::arg("in0image"), 
             py::arg("in1image"), 
             py::arg("timestep"));
}