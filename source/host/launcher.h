#pragma once
#include <cuda_fp16.h>

void sgemm_naive(int M,int N,int K,float alpha,const float *A,const float *B,float beta, float *C);
void sgemm_coalesce(int M,int N,int K,float alpha,const float *A,const float *B,float beta, float *C);
void sgemm_sm(int M,int N,int K,float alpha,const float *A,const float *B,float beta, float *C);
void elementwise_add_f32(float *a,float *b,float *c,int N);
void elementwise_add_f32x4(float *a,float *b,float *c,int N);
void elementwise_add_f16(half *a,half *b,half *c,int N);
void elementwise_add_f16x2(half *a,half *b,half *c,int N);
void elementwise_add_f16x8_pack(half *a,half *b,half *c,int N);