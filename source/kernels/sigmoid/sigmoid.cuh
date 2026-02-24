#include<cmath>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda_fp16.h>


#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f

#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

__global__ void sigmoid_naive_kernel(float *x,float *y, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<N)
    {
        float v = x[idx];
        // 对值进行截断
        v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
        y[idx] = 1.0f/(1.0f + expf(-v));
    }
}

__global__ void sigmoid_f32x4_kernel(float *x, float *y, int N)
{
    int idx = (blockDim.x*blockIdx.x+threadIdx.x)*4;

    float4 reg_x = reinterpret_cast<float4*>(&x[idx])[0];

    reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
    reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
    reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
    reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

    float4 reg_y;

    reg_y.x = 1.0f/(1.0f+expf(-reg_x.x));
    reg_y.y = 1.0f/(1.0f+expf(-reg_x.y));
    reg_y.z = 1.0f/(1.0f+expf(-reg_x.z));
    reg_y.w = 1.0f/(1.0f+expf(-reg_x.w));

    if(idx<N)
    {
        reinterpret_cast<float4*>(&y[idx])[0] = reg_y;
    }
}

__global__ void sigmoid_f16_kernel(half *x, half *y, int N)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    const half half_one = __float2half(1.0f);
    if(idx<N)
    {
        half val = x[idx];
        val = __hmin(__hmax(val,MIN_EXP_F16),MAX_EXP_F16);
        y[idx] = half_one / (half_one + hexp(-val));
    }
}

__global__ void sigmoid_f16x2_kernel(half *x, half *y, int N)
{
    int idx = 2*(blockDim.x*blockIdx.x + threadIdx.x);
    const half half_one = __float2half(1.0f);

    half2 val = reinterpret_cast<half2*>(&x[idx])[0];
    half2 y_val;

    val.x = __hmin(__hmax(val.x, MIN_EXP_F16), MAX_EXP_F16);
    val.y = __hmin(__hmax(val.y, MIN_EXP_F16), MAX_EXP_F16);

    y_val.x = half_one / (half_one+hexp(val.x));
    y_val.y = half_one / (half_one+hexp(val.y));

    if(idx<N)
    {
        reinterpret_cast<half2*>(&y[idx])[0] = y_val;
    }
}

__global__ void sigmoid_f16x8_kernel(half *x, half *y, int N)
{
    int idx = 8*(blockDim.x*blockIdx.x+threadIdx.x);
    half pack_x[8],pack_y[8];
    const half half_one = __float2half(1.0f);
    reinterpret_cast<float4*>(&pack_x[0])[0] = reinterpret_cast<float4*>(&x[idx])[0];

    #pragma unroll
    for(int i =0;i<8;++i)
    {
        half val = pack_x[i];
        val = __hmin(__hmax(val, MIN_EXP_F16), MAX_EXP_F16);
        pack_y[i] = half_one/(half_one+hexp(-val));
    }

    if(idx+7<N)
    {
        reinterpret_cast<float4*>(&y[idx])[0] = reinterpret_cast<float4*>(&pack_y[0])[0];
    }
}