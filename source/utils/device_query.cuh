#pragma once

#include <cstddef>
#include<cuda_runtime.h>
#include<string>


namespace cuda_utils {
    
    struct DeviceMemoryInfo
    {
        int l2CacheSize;
        int regsPerBlock;
        int regsPerMultiprocessor;
        int memoryBusWidth;
        int memory_clock_rate;
        size_t sharedMemPerBlock;
        size_t sharedMemPerMultiprocessor;
        size_t totalGlobalMem;
        float memory_band_width;
    };

    struct DeviceComputeInfo
    {
        dim3 maxGridSize;
        dim3 maxThreadsDim;
        int maxThreadsPerBlock;
        int maxBlocksPerMultiProcessor;
        int maxThreadsPerMultiProcessor;
        int multiProcessorCount;
        int core_clock_rate;
    };

    struct GpuDeviceProps
    {
        int device_ID;
        std::string name;
        DeviceMemoryInfo memory;
        DeviceComputeInfo compute;
    };

    //  默认设备id为0
    GpuDeviceProps get_device_propertity(int deviceId=0);
    void print_device_properties(int device_ID=0);
}