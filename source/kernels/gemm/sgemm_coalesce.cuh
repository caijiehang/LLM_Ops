#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <sys/types.h>

template<const uint BLOCKSIZE>
__global__ void sgemm_coalesce_kernel(int M,int N,int K,float alpha,const float* A,const float* B, float beta, float* C)
{
    const uint cCol = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;
    const uint cRow = blockIdx.y * BLOCKSIZE + threadIdx.x / BLOCKSIZE;

    if(cCol<N && cRow<M)
    {
        float tmp = 0.0f;
        for(int k = 0;k<K;++k)
        {
            tmp += A[cRow*K+k]*B[k*N+cCol];
        }
        C[cRow*N+cCol] = alpha * tmp + beta*C[cRow*N+cCol];
    }
}