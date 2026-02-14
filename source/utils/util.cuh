#pragma once 
#include <cuda_runtime.h>
#include <iostream>

__host__ __device__ int  inline div_up(int a,int b)
{
    return a+b-1/b;
}

void check_cuda_error(cudaError_t error,int line, const char* file)
{
    std::cout<<"the error occur at "<<file<<":"<<line<<std::endl;
    std::cout<<"the error code is :"<<error<<std::endl;
    std::cout<<"Error name :"<<cudaGetErrorName(error)<<std::endl;
    std::cout<<"Error string :"<<cudaGetErrorString(error)<<std::endl;
}