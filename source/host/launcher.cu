#include "gemm/sgemm_naive.cuh"
#include "util.cuh"
#include "launcher.h"

void sgemm_naive(int M,int N,int K,float alpha,const float *A,const float *B,float beta, float *C)
{
    dim3 grid(div_up(N, 32),div_up(M, 32));
    dim3 Block(32,32);
    
    sgemm_naive_kernel<<<grid,Block>>>(M,N,K,alpha,A,B,beta,C);
}
