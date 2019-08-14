/*
python wrapper for knncuda
Author: Dylan (Wenxuan) Wu
Date: 08/2019
*/





#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>

#include "knncuda.h"

namespace py = pybind11;

struct KnnData {
    py::array_t<float> knn_dist;
    py::array_t<int> knn_index;
};

KnnData knn(py::array_t<float> ref, py::array_t<float> query, int k) {
    KnnData return_object;

    auto buf_ref = ref.request();
    auto buf_query = query.request();
    //if (buf_ref.size != buf_query.size) throw std::runtime_error("sizes do not match!");

    float * ptr_ref = (float*)buf_ref.ptr;
    float * ptr_query = (float*)buf_query.ptr;

    int N_ref = ref.shape()[1];
    int dim_ref = ref.shape()[0];
    int N_query = query.shape()[1];
    int dim_query = query.shape()[0];

    if (dim_ref != dim_query) throw std::runtime_error("dim do not match!");

    float knn_dist[N_query*k];
    int knn_index[N_query*k];

    knn_cublas(ptr_ref, N_ref, ptr_query, N_query, dim_query, k, knn_dist, knn_index);

    return_object.knn_dist = py::array_t<float>({N_query, k}, knn_dist);
    return_object.knn_index = py::array_t<int>({N_query, k}, knn_index);
    return return_object;
}

PYBIND11_MODULE(knncuda, m) {
    m.doc() = "Using cuda to find k nearest neighbor";
    m.def("knn", &knn, "test code for knn");

    py::class_<KnnData> (m, "KnnData")
        .def_readwrite("knn_dist", &KnnData::knn_dist)
        .def_readwrite("knn_index", &KnnData::knn_index);
}



