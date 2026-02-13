#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <sys/types.h>

template<typename T>
__device__ void __inline__ sgemm_core_naive(int M, int N, int K, float alpha, T *A, T *B, float beta, T*C, int row, int col)
{
    if(col<N && row<M)
    {
        T tmp = 0.0f;
        for(int i = 0; i<K;++i)
        {
            tmp += A[row*K+i]*B[i*N+col];
        }
        C[row*N+col] = alpha * tmp + beta * C[row*N+col];
    }
}

template<typename T>
__global__ void sgemm_naive_kernel(int M,int N, int K, float alpha, T *A, T* B, float beta, T *C)
{
    uint row = blockDim.y * blockIdx.y + threadIdx.y;
    uint col = blockDim.x * blockIdx.x + threadIdx.x;

    sgemm_core_naive(M, N, K, alpha, A, B, beta, C, row, col);
}
