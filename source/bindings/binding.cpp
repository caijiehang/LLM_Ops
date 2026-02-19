#include <nanobind/nanobind.h>
#include <stdexcept>
#include "device_query.h"
#include "host/launcher.h"
#include "nanobind/nb_defs.h"
#include "nanobind/ndarray.h"


namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(llm_ops, m){
    nb::class_<cuda_utils::DeviceMemoryInfo>(m,"memoryProps")
        .def_ro("l2CacheSize", &cuda_utils::DeviceMemoryInfo::l2CacheSize)
        .def_ro("regsPerBlock", &cuda_utils::DeviceMemoryInfo::regsPerBlock)
        .def_ro("regsPerMultiprocessor", &cuda_utils::DeviceMemoryInfo::regsPerMultiprocessor)
        .def_ro("memoryBusWidth", &cuda_utils::DeviceMemoryInfo::memoryBusWidth)
        .def_ro("memory_clock_rate", &cuda_utils::DeviceMemoryInfo::memory_clock_rate)
        .def_ro("sharedMemPerBlock", &cuda_utils::DeviceMemoryInfo::sharedMemPerBlock)
        .def_ro("sharedMemPerMultiprocessor", &cuda_utils::DeviceMemoryInfo::sharedMemPerMultiprocessor)
        .def_ro("totalGlobalMem", &cuda_utils::DeviceMemoryInfo::totalGlobalMem)
        .def_ro("memory_band_width", &cuda_utils::DeviceMemoryInfo::memory_band_width);

    nb::class_<cuda_utils::DeviceComputeInfo>(m,"computeProps")
        .def_ro("maxGridSize", &cuda_utils::DeviceComputeInfo::maxGridSize)
        .def_ro("maxThreadsDim", &cuda_utils::DeviceComputeInfo::maxThreadsDim)
        .def_ro("maxThreadsPerBlock", &cuda_utils::DeviceComputeInfo::maxThreadsPerBlock)
        .def_ro("maxBlocksPerMultiProcessor", &cuda_utils::DeviceComputeInfo::maxBlocksPerMultiProcessor)
        .def_ro("maxThreadsPerMultiProcessor", &cuda_utils::DeviceComputeInfo::maxThreadsPerMultiProcessor)
        .def_ro("multiProcessorCount", &cuda_utils::DeviceComputeInfo::multiProcessorCount)
        .def_ro("core_clock_rate", &cuda_utils::DeviceComputeInfo::core_clock_rate);

    nb::class_<cuda_utils::GpuDeviceProps>(m,"deviceProps")
        .def(nb::init<>())
        .def_ro("device_ID",&cuda_utils::GpuDeviceProps::device_ID)
        .def_ro("name",&cuda_utils::GpuDeviceProps::name)
        .def_ro("memoryProps",&cuda_utils::GpuDeviceProps::memory)
        .def_ro("computeProps",&cuda_utils::GpuDeviceProps::compute);

    m.def("get_device_propertity", &cuda_utils::get_device_propertity,"Get cuda device property",nb::arg("deviceId")=0);
    m.def("print_device_properties", &cuda_utils::print_device_properties,"Print cuda device property",nb::arg("deviceId")=0);

    m.def("sgemm_naive", [](
        int M, int N, int K,float alpha,
        nb::ndarray<const float, nb::ndim<2>, nb::device::cuda, nb::c_contig> A,    //const float
        nb::ndarray<const float, nb::ndim<2>, nb::device::cuda, nb::c_contig> B,
        float beta,
        nb::ndarray<float, nb::ndim<2>, nb::device::cuda, nb::c_contig> C
    ){
        if(A.shape(0) != M || A.shape(1) != K) throw std::runtime_error("Shape mistake with A");
        if(B.shape(0) != K || B.shape(1) != N) throw std::runtime_error("Shape mistake with B");
        if(C.shape(0) != M || C.shape(1) != N) throw std::runtime_error("Shape mistake with C");
        
        sgemm_naive(M,N,K,alpha,A.data(),B.data(),beta,C.data());
    },
    "M"_a, "N"_a, "K"_a, "alpha"_a, "A"_a.noconvert(), "B"_a.noconvert(), "beta"_a, "C"_a.noconvert(),
    "A naive SGEMM implementation calling CUDA code");

    m.def("sgemm_coalesce",[](
        int M,int N,int K,float alpha,
        nb::ndarray<const float, nb::ndim<2>, nb::device::cuda, nb::c_contig>A,
        nb::ndarray<const float, nb::ndim<2>, nb::device::cuda, nb::c_contig>B,
        float beta,
        nb::ndarray<float, nb::ndim<2>, nb::device::cuda, nb::c_contig>C
    ){
        if(A.shape(0) != M || A.shape(1) != K) throw std::runtime_error("Shape mistake with A");
        if(B.shape(0) != K || B.shape(1) != N) throw std::runtime_error("Shape mistake with B");
        if(C.shape(0) != M || C.shape(1) != N) throw std::runtime_error("Shape mistake with C");

        sgemm_coalesce(M,N,K,alpha,A.data(),B.data(),beta,C.data());
    },
    "M"_a, "N"_a, "K"_a, "alpha"_a, "A"_a.noconvert(), "B"_a.noconvert(), "beta"_a, "C"_a.noconvert(),
    "A SGEMM implementation coalesce global memory access calling CUDA code");

    m.def("sgemm_sm",[](
        int M, int N, int K, float alpha,
        nb::ndarray<const float, nb::ndim<2>, nb::device::cuda, nb::c_contig>A,
        nb::ndarray<const float, nb::ndim<2>, nb::device::cuda, nb::c_contig>B,
        float beta,
        nb::ndarray<float, nb::ndim<2>, nb::device::cuda, nb::c_contig>C
    ){
        if(A.shape(0) != M || A.shape(1) != K) throw std::runtime_error("Shape mistake with A");
        if(B.shape(0) != K || B.shape(1) != N) throw std::runtime_error("Shape mistake with B");
        if(C.shape(0) != M || C.shape(1) != N) throw std::runtime_error("Shape mistake with C");

        sgemm_sm(M, N, K, alpha, A.data(), B.data(), beta, C.data());
    },
    "M"_a, "N"_a, "K"_a, "alpha"_a, "A"_a.noconvert(), "B"_a.noconvert(), "beta"_a, "C"_a.noconvert(),
    "Shared-memory optimized SGEMM CUDA kernel");
};