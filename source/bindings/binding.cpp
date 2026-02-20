#include <cstdint>
#include <nanobind/nanobind.h>
#include <stdexcept>
#include "device_query.h"
#include "host/launcher.h"
#include "nanobind/nb_defs.h"
#include "nanobind/ndarray.h"


namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(llm_ops, m){

    //  创建子模块
    nb::module_ m_utils = m.def_submodule("m_utils","Utility functions and device property queries");
    nb::module_ m_gemm = m.def_submodule("gemm","GEMM (General Matrix Multiply) kernels");
    nb::module_ m_elementwise = m.def_submodule("elementwise","An element-wise operation operates on corresponding elements between tensors.");
    
    nb::class_<cuda_utils::DeviceMemoryInfo>(m_utils,"memoryProps")
        .def_ro("l2CacheSize", &cuda_utils::DeviceMemoryInfo::l2CacheSize)
        .def_ro("regsPerBlock", &cuda_utils::DeviceMemoryInfo::regsPerBlock)
        .def_ro("regsPerMultiprocessor", &cuda_utils::DeviceMemoryInfo::regsPerMultiprocessor)
        .def_ro("memoryBusWidth", &cuda_utils::DeviceMemoryInfo::memoryBusWidth)
        .def_ro("memory_clock_rate", &cuda_utils::DeviceMemoryInfo::memory_clock_rate)
        .def_ro("sharedMemPerBlock", &cuda_utils::DeviceMemoryInfo::sharedMemPerBlock)
        .def_ro("sharedMemPerMultiprocessor", &cuda_utils::DeviceMemoryInfo::sharedMemPerMultiprocessor)
        .def_ro("totalGlobalMem", &cuda_utils::DeviceMemoryInfo::totalGlobalMem)
        .def_ro("memory_band_width", &cuda_utils::DeviceMemoryInfo::memory_band_width);

    nb::class_<cuda_utils::DeviceComputeInfo>(m_utils,"computeProps")
        .def_ro("maxGridSize", &cuda_utils::DeviceComputeInfo::maxGridSize)
        .def_ro("maxThreadsDim", &cuda_utils::DeviceComputeInfo::maxThreadsDim)
        .def_ro("maxThreadsPerBlock", &cuda_utils::DeviceComputeInfo::maxThreadsPerBlock)
        .def_ro("maxBlocksPerMultiProcessor", &cuda_utils::DeviceComputeInfo::maxBlocksPerMultiProcessor)
        .def_ro("maxThreadsPerMultiProcessor", &cuda_utils::DeviceComputeInfo::maxThreadsPerMultiProcessor)
        .def_ro("multiProcessorCount", &cuda_utils::DeviceComputeInfo::multiProcessorCount)
        .def_ro("core_clock_rate", &cuda_utils::DeviceComputeInfo::core_clock_rate);

    nb::class_<cuda_utils::GpuDeviceProps>(m_utils,"deviceProps")
        .def(nb::init<>())
        .def_ro("device_ID",&cuda_utils::GpuDeviceProps::device_ID)
        .def_ro("name",&cuda_utils::GpuDeviceProps::name)
        .def_ro("memoryProps",&cuda_utils::GpuDeviceProps::memory)
        .def_ro("computeProps",&cuda_utils::GpuDeviceProps::compute);

    m_utils.def("get_device_propertity", &cuda_utils::get_device_propertity,"Get cuda device property",nb::arg("deviceId")=0);
    m_utils.def("print_device_properties", &cuda_utils::print_device_properties,"Print cuda device property",nb::arg("deviceId")=0);

    m_gemm.def("sgemm_naive", [](
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

    m_gemm.def("sgemm_coalesce",[](
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

    m_gemm.def("sgemm_sm",[](
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

    m_elementwise.def("elementwise_f32",[](
        nb::ndarray<float, nb::ndim<1>, nb::device::cuda, nb::c_contig>a,
        nb::ndarray<float, nb::ndim<1>, nb::device::cuda, nb::c_contig>b,
        nb::ndarray<float, nb::ndim<1>, nb::device::cuda, nb::c_contig>c,
        int N
    ){
        if(a.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor a");
        if(b.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor b");
        if(c.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor c");

        elementwise_add_f32(a.data(), b.data(), c.data(), N);
    },
    "a"_a,"b"_a,"c"_a,"N"_a);

    m_elementwise.def("elementwise_f32x4",[](
        nb::ndarray<float, nb::ndim<1>, nb::device::cuda, nb::c_contig>a,
        nb::ndarray<float, nb::ndim<1>, nb::device::cuda, nb::c_contig>b,
        nb::ndarray<float, nb::ndim<1>, nb::device::cuda, nb::c_contig>c,
        int N
    ){
        if(a.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor a");
        if(b.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor b");
        if(c.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor c");

        elementwise_add_f32x4(a.data(), b.data(), c.data(), N);
    },
    "a"_a,"b"_a,"c"_a,"N"_a);

    m_elementwise.def("elementwise_f16",[](
        nb::ndarray<nb::ndim<1>, nb::device::cuda, nb::c_contig>a,
        nb::ndarray<nb::ndim<1>, nb::device::cuda, nb::c_contig>b,
        nb::ndarray<nb::ndim<1>, nb::device::cuda, nb::c_contig>c,
        int N
    ){
        if(a.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor a");
        if(b.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor b");
        if(c.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor c");

        if(a.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || a.dtype().bits != 16){
            throw std::invalid_argument("input a must be a float16 tensor");
        }

        if(b.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || b.dtype().bits != 16){
            throw std::invalid_argument("input a must be a float16 tensor");
        }

        if(b.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || b.dtype().bits != 16){
            throw std::invalid_argument("input a must be a float16 tensor");
        }

        half* ptr_a = reinterpret_cast<half*>(a.data());
        half* ptr_b = reinterpret_cast<half*>(b.data());
        half* ptr_c = reinterpret_cast<half*>(c.data());

        elementwise_add_f16(ptr_a, ptr_b, ptr_c, N);
    },
    "a"_a,"b"_a,"c"_a,"N"_a);

    m_elementwise.def("elementwise_f16x2",[](
        nb::ndarray<nb::ndim<1>, nb::device::cuda, nb::c_contig>a,
        nb::ndarray<nb::ndim<1>, nb::device::cuda, nb::c_contig>b,
        nb::ndarray<nb::ndim<1>, nb::device::cuda, nb::c_contig>c,
        int N
    ){
        if(a.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor a");
        if(b.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor b");
        if(c.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor c");

        if(a.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || a.dtype().bits != 16){
            throw std::invalid_argument("input a must be a float16 tensor");
        }

        if(b.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || b.dtype().bits != 16){
            throw std::invalid_argument("input a must be a float16 tensor");
        }

        if(b.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || b.dtype().bits != 16){
            throw std::invalid_argument("input a must be a float16 tensor");
        }

        half* ptr_a = reinterpret_cast<half*>(a.data());
        half* ptr_b = reinterpret_cast<half*>(b.data());
        half* ptr_c = reinterpret_cast<half*>(c.data());

        elementwise_add_f16x2(ptr_a, ptr_b, ptr_c, N);
    },
    "a"_a,"b"_a,"c"_a,"N"_a);

    m_elementwise.def("elementwise_f16x8_pack",[](
        nb::ndarray<nb::ndim<1>, nb::device::cuda, nb::c_contig>a,
        nb::ndarray<nb::ndim<1>, nb::device::cuda, nb::c_contig>b,
        nb::ndarray<nb::ndim<1>, nb::device::cuda, nb::c_contig>c,
        int N
    ){
        if(a.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor a");
        if(b.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor b");
        if(c.shape(0)!=N) throw std::runtime_error("Shape mistake with tensor c");

        if(a.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || a.dtype().bits != 16){
            throw std::invalid_argument("input a must be a float16 tensor");
        }

        if(b.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || b.dtype().bits != 16){
            throw std::invalid_argument("input a must be a float16 tensor");
        }

        if(b.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || b.dtype().bits != 16){
            throw std::invalid_argument("input a must be a float16 tensor");
        }

        half* ptr_a = reinterpret_cast<half*>(a.data());
        half* ptr_b = reinterpret_cast<half*>(b.data());
        half* ptr_c = reinterpret_cast<half*>(c.data());

        elementwise_add_f16x8_pack(ptr_a, ptr_b, ptr_c, N);
    },
    "a"_a,"b"_a,"c"_a,"N"_a);
};