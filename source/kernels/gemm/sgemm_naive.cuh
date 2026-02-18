#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sys/types.h>


__global__ void sgemm_naive_kernel(int M,int N, int K, float alpha,const float *A,const float* B, float beta, float *C)
{
    uint row = blockDim.y * blockIdx.y + threadIdx.y;
    uint col = blockDim.x * blockIdx.x + threadIdx.x;

    if(col<N && row<M)
    {
        float tmp = 0.0f;
        for(int i = 0; i<K;++i)
        {
            tmp += A[row*K+i]*B[i*N+col];
        }
        C[row*N+col] = alpha * tmp + beta * C[row*N+col];
    }
}
