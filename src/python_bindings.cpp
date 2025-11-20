#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "jpeg_encoder.h"

namespace py = pybind11;

PYBIND11_MODULE(jpeg_py, m) {
    m.doc() = "JPEG Parallel Encoder Python Bindings";

    py::class_<Image>(m, "Image")
        .def(py::init<>())
        .def_readwrite("data", &Image::data)
        .def_readwrite("width", &Image::width)
        .def_readwrite("height", &Image::height)
        .def_readwrite("channels", &Image::channels);

    py::class_<EncodedData>(m, "EncodedData")
        .def(py::init<>())
        .def_readwrite("data", &EncodedData::data)
        .def_readwrite("size", &EncodedData::size);

    py::class_<JPEGEncoder>(m, "JPEGEncoder")
        .def(py::init<int, int>(), 
             py::arg("quality") = 85, 
             py::arg("num_threads") = 4)
        .def("encode", &JPEGEncoder::encode)
        .def("decode", &JPEGEncoder::decode)
        .def("set_quality", &JPEGEncoder::set_quality)
        .def("set_threads", &JPEGEncoder::set_threads);
}
