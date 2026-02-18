#include "gemm/sgemm.cuh"
#include "util.cuh"
#include "launcher.h"

void sgemm_naive(int M,int N,int K,float alpha,const float *A,const float *B,float beta, float *C)
{
    dim3 Grid(div_up(N, 32),div_up(M, 32));
    dim3 Block(32,32);
    
    sgemm_naive_kernel<<<Grid,Block>>>(M,N,K,alpha,A,B,beta,C);
}

void sgemm_coalesce(int M,int N,int K,float alpha,const float *A,const float *B,float beta, float *C)
{
    const int BLOCKSIZE_LOCAL = 32;
    dim3 Grid(div_up(N, 32),div_up(M, 32));
    dim3 Block(BLOCKSIZE_LOCAL*BLOCKSIZE_LOCAL);
    
    sgemm_coalesce_kernel<BLOCKSIZE_LOCAL><<<Grid,Block>>>(M,N,K,alpha,A,B,beta,C);
}