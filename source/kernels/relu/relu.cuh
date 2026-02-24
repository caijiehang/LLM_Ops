#include <cmath>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda_fp16.h>

__global__ void relu_fp32_kernel(float *x, float *y, int N)
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N)
    {
        y[idx] = fmaxf(x[idx], 0.0f);
    }
    
}

__global__ void relu_fp16_kernel(half *x, half *y, int N)
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N)
    {
        y[idx] = __hmax(x[idx],__float2half(0.0f));
    }
}

__global__ void relu_fp32x4_kernel(float *x, float *y, int N)
{
    int idx = 8*(blockDim.x*blockIdx.x+threadIdx.x);
    
    if(idx<N)
    {
        float4 reg_x = reinterpret_cast<float4*>(&x[idx])[0];
        float4 reg_y;
        reg_y.x = fmaxf(reg_x.x,0.0f);
        reg_y.y = fmaxf(reg_x.y,0.0f);
        reg_y.z = fmaxf(reg_x.z,0.0f);
        reg_y.w = fmaxf(reg_x.w,0.0f);
        reinterpret_cast<float4*>(&y[idx])[0] = reg_y;
    }
}

__global__ void relu_fp16x2_kernel(half *x, half *y, int N)
{
    int idx = 2*(blockDim.x*blockIdx.x+threadIdx.x);
    half fp16_0 = __float2half(0.0f);
    if(idx<N)
    {
        half2 reg_x = reinterpret_cast<half2*>(&x[idx])[0];
        half2 reg_y;
        reg_x.x = __hmax(reg_x.x,fp16_0);
        reg_x.y = __hmax(reg_x.y,fp16_0);
        reinterpret_cast<half2*>(&y[idx])[0] = reg_y;
    }
}

__global__ void relu_fp16x8_kernel(half *x, half *y, int N)
{
    int idx = 8*(blockDim.x*blockIdx.x+threadIdx.x);
    half pack_x[8], pack_y[8];
    half2 fp16x2_zero = {__float2half(0.0f),__float2half(0.0f)};

    reinterpret_cast<float4*>(&pack_x[0])[0] = reinterpret_cast<float4*>(&x[idx])[0];
    #pragma unroll
    for(int i = 0;i<8;i+=2)
    {
        reinterpret_cast<half2*>(&pack_y[i])[0] = __hmax2(reinterpret_cast<half2*>(&pack_x[i])[0],fp16x2_zero);
    }

    if((idx+7)<N)
    {
        reinterpret_cast<float4*>(&pack_y[0])[0] = reinterpret_cast<float4*>(&y[idx])[0];
    }
}