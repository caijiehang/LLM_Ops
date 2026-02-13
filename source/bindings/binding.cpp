#include <nanobind/nanobind.h>
#include "device_query.h"
#include "nanobind/nb_defs.h"

namespace nb = nanobind;

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

};