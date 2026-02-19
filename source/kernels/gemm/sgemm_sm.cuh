# pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sys/types.h>

template<int BLOCKSIZE>
__global__ void sgemm_sm_kernel(int M, int N,int K, float alpha,const float *A,const float *B, float beta, float *C)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    __shared__ float As[BLOCKSIZE*BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE*BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    A+=cRow*BLOCKSIZE*K;
    B+=cCol*BLOCKSIZE;
    C+=cRow*BLOCKSIZE*N+cCol*BLOCKSIZE;

    float tmp = 0.0f;
    for(int bkIdx = 0;bkIdx<K;bkIdx+=BLOCKSIZE)
    {
        // 将数据搬运到共享内存中
        As[threadRow*BLOCKSIZE+threadCol] = A[threadRow*K+threadCol];
        Bs[threadRow*BLOCKSIZE+threadCol] = B[threadRow*N+threadCol];

        __syncthreads();

        A+=BLOCKSIZE;
        B+=BLOCKSIZE*N;

        for(int dotidx = 0;dotidx<BLOCKSIZE;++dotidx)
        {
            tmp+=As[threadRow*BLOCKSIZE+dotidx]*Bs[dotidx*BLOCKSIZE+threadCol];
        }
        __syncthreads();
    }
    C[threadRow*N+threadCol] = alpha*tmp+beta*C[threadRow*N+threadCol];

}