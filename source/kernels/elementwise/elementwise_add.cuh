#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

__global__ void elementwise_add_f32_kernel(float *a, float *b,float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N)
{
    int idx = 4*(blockIdx.x*blockDim.x+threadIdx.x);

    if(idx<N)
    {
       float4 reg_a = reinterpret_cast<float4*>(&a[idx])[0];
       float4 reg_b = reinterpret_cast<float4*>(&b[idx])[0];
       float4 reg_c;
       reg_c.x = reg_a.x + reg_b.x;
       reg_c.y = reg_a.y + reg_b.y;
       reg_c.z = reg_a.z + reg_c.z;
       reg_c.w = reg_c.w + reg_c.w;

       reinterpret_cast<float4*>(&c[idx])[0] = reg_c;
    }
}

__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx<N)
    {
        c[idx] = __hadd(a[idx],b[idx]);
    }
}

__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N)
{
    int idx = 2*(blockIdx.x*blockDim.x+threadIdx.x);

    if(idx<N)
    {
        half2 reg_a = reinterpret_cast<half2*>(&a[idx])[0];
        half2 reg_b = reinterpret_cast<half2*>(&b[idx])[0];

        half2 reg_c;
        reg_c.x = __hadd(reg_a.x,reg_b.x);
        reg_c.y = __hadd(reg_a.y,reg_b.y);

        reinterpret_cast<half2*>(&c[idx])[0] = reg_c;
    }
}

__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N)
{
    int idx = 8*(blockIdx.x*blockDim.x+threadIdx.x);

    half pack_a[8],pack_b[8],pack_c[8];

    reinterpret_cast<float4*>(&pack_a)[0] = reinterpret_cast<float4*>(&a[idx])[0];
    reinterpret_cast<float4*>(&pack_b)[0] = reinterpret_cast<float4*>(&b[idx])[0];

    #pragma unroll
    for(int i =0;i<8;i+=2)
    {
        reinterpret_cast<half2*>(&pack_c[i])[0] = __hadd2(reinterpret_cast<half2*>(&pack_a)[i],reinterpret_cast<half2*>(&pack_b)[i]);
    }

    if(idx+7<N)
    {
        reinterpret_cast<float4*>(&c[idx])[0] = reinterpret_cast<float4*>(&pack_c)[0];
    }
    else {
        for(int i = 0;idx+i<N;++i)
        {
            c[idx+i] = __hadd(a[idx+i],b[idx+i]);
        }
    }

}



