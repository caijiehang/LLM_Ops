#include "gemm/sgemm.cuh"
#include "elementwise/elementwise_add.cuh"
#include "gemm/sgemm_sm.cuh"
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

void sgemm_sm(int M,int N,int K,float alpha,const float *A,const float *B,float beta, float *C)
{
    const int BLOCKSIZE_LOCAL = 32;
    dim3 Grid(div_up(N, 32),div_up(M, 32));
    dim3 Block(BLOCKSIZE_LOCAL*BLOCKSIZE_LOCAL);
    
    sgemm_sm_kernel<BLOCKSIZE_LOCAL><<<Grid,Block>>>(M, N, K, alpha, A, B, beta, C);
}

void elementwise_add_f32(float *a,float *b,float *c,int N)
{
    dim3 Grid(div_up(N, 256));
    dim3 Block(256);
    elementwise_add_f32_kernel<<<Grid,Block>>>(a, b, c, N);
}

void elementwise_add_f32x4(float *a,float *b,float *c,int N)
{
    dim3 Grid(div_up(N, 256));
    dim3 Block(256);
    elementwise_add_f32x4_kernel<<<Grid,Block>>>(a, b, c, N);
}

void elementwise_add_f16(half *a,half *b,half *c,int N)
{
    dim3 Grid(div_up(N, 256));
    dim3 Block(256);
    elementwise_add_f16_kernel<<<Grid,Block>>>(a, b, c, N);
}

void elementwise_add_f16x2(half *a,half *b,half *c,int N)
{
    dim3 Grid(div_up(N, 256));
    dim3 Block(256);
    elementwise_add_f16x2_kernel<<<Grid,Block>>>(a, b, c, N);
}

void elementwise_add_f16x8_pack(half *a,half *b,half *c,int N)
{
    dim3 Grid(div_up(N, 256));
    dim3 Block(256);
    elementwise_add_f16x8_pack_kernel<<<Grid,Block>>>(a, b, c, N);
}