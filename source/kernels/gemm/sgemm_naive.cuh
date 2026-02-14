#pragma once

#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <sys/types.h>


__global__ void __inline__ sgemm_naive_kernel(int M,int N, int K, float alpha, float *A, float* B, float beta, float *C)
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
