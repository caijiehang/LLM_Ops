#include "device_query.h"
#include <cstdio>
#include <iostream>

namespace cuda_utils{

    GpuDeviceProps get_device_propertity(int deviceId)
    {
        cuda_utils::GpuDeviceProps props;

        cudaDeviceProp Raw_props;
        cudaGetDeviceProperties(&Raw_props, deviceId);

        props.name = Raw_props.name;

        props.memory.l2CacheSize = Raw_props.l2CacheSize;       
        props.memory.regsPerBlock = Raw_props.regsPerBlock;     
        props.memory.regsPerMultiprocessor = Raw_props.regsPerMultiprocessor;
        props.memory.sharedMemPerBlock = Raw_props.sharedMemPerBlock;
        props.memory.sharedMemPerMultiprocessor = Raw_props.sharedMemPerMultiprocessor;
        props.memory.totalGlobalMem = Raw_props.totalGlobalMem;
        props.memory.memoryBusWidth = Raw_props.memoryBusWidth;


        props.compute.maxThreadsDim.x = Raw_props.maxThreadsDim[0];
        props.compute.maxThreadsDim.y = Raw_props.maxThreadsDim[1];
        props.compute.maxThreadsDim.z = Raw_props.maxThreadsDim[2];

        props.compute.maxGridSize.x = Raw_props.maxGridSize[0];
        props.compute.maxGridSize.y = Raw_props.maxGridSize[1];
        props.compute.maxGridSize.z = Raw_props.maxGridSize[2];

        props.compute.maxThreadsPerBlock = Raw_props.maxThreadsPerBlock;
        props.compute.maxBlocksPerMultiProcessor = Raw_props.maxBlocksPerMultiProcessor;
        props.compute.maxThreadsPerMultiProcessor = Raw_props.maxThreadsPerMultiProcessor;
        props.compute.multiProcessorCount = Raw_props.multiProcessorCount;

        int memory_clock_rate = 0;
        cudaDeviceGetAttribute(&memory_clock_rate, cudaDevAttrMemoryClockRate, deviceId); 
        props.memory.memory_clock_rate = memory_clock_rate;

        int core_clock_rate = 0;
        cudaDeviceGetAttribute(&core_clock_rate, cudaDevAttrClockRate, deviceId);
        props.compute.core_clock_rate = core_clock_rate;

        props.memory.memory_band_width = 1000.0*props.memory.memory_clock_rate*2*(props.memory.memoryBusWidth/8.0) / 1e9;
        
        return props;
    }

    void print_device_properties(int device_ID)
    {
        GpuDeviceProps props = get_device_propertity(device_ID);

        std::cout << "the gpu deviceID is :                           " << props.device_ID << std::endl;
        std::cout << "the gpu name is :                               " << props.name << std::endl;
        
        std::cout << "=============MEMORY INFORMATION=================" << std::endl;        
        std::cout << "shared memory per block is :                    " << props.memory.sharedMemPerBlock/1024 << "KB" << std::endl;
        std::cout << "shared memory per multiprocessor is :           " << props.memory.sharedMemPerMultiprocessor/1024 << "KB" << std::endl;
        std::cout << "total global memory is :                        " << props.memory.totalGlobalMem/1024 << "KB" << std::endl;
        std::cout << "l2 cache memory is :                            " << props.memory.l2CacheSize/1024 << "KB" << std::endl;
        std::cout << "regs per block are :                            " << props.memory.regsPerBlock << std::endl;
        std::cout << "regs per multiprocessor are :                   " << props.memory.regsPerMultiprocessor << std::endl;
        std::cout << "memory Bus width are :                          " << (double)props.memory.memoryBusWidth/8 << "Bytes" << std::endl;
        std::cout << "memory clock rate are :                         " << props.memory.memory_clock_rate << "kHz" << std::endl;
        std::cout << "memory band width are :                         " << props.memory.memory_band_width << "GB/s" << std::endl;

        std::cout << "=============COMPUTE INFORMATION================" << std::endl;
        std::cout << "the number of the MultiProcessor is  :          " << props.compute.multiProcessorCount <<std::endl;
        std::cout << "max grid size is :                              " << "(" << props.compute.maxGridSize.x << "," << props.compute.maxGridSize.y << "," << props.compute.maxGridSize.z << ")" << std::endl;
        std::cout << "Max threads per block dimension: :              " << "(" << props.compute.maxThreadsDim.x << "," << props.compute.maxThreadsDim.y << "," << props.compute.maxThreadsDim.z << ")" << std::endl;
        std::cout << "max blocks per MultiProcessor is :              " << props.compute.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "max threads per block is :                      " << props.compute.maxThreadsPerBlock << std::endl;
        std::cout << "the core clock rate is :                        " << props.compute.core_clock_rate << std::endl;
        
    }
}



