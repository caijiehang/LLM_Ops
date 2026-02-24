#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_runtime_wrapper.h>
#include<cuda_fp16.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <ratio>

#define WarpSize 32

template <int KWarpSize>
__device__ __forceinline__ float warp_reduce_sum_f32(float val)
{
    #pragma unroll
    for(int mask = KWarpSize >>1;mask>=1;mask>>=1)
    {
        val += __shfl_xor_sync(0xffffffff,val,mask);
    }
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *x, float *y,int N)
{
    int tid = threadIdx.x;
    int index = blockDim.x*blockIdx.x+threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS+WarpSize-1)/WarpSize;
    __shared__ float reduce_smem[NUM_WARPS];

    float sum = index<N ? x[index] : 0.0f;

    int warp = tid/WarpSize;
    int lane = tid%WarpSize;

    sum = warp_reduce_sum_f32<warpSize>(sum);
    if(lane == 0)
    {
        reduce_smem[warp] = sum;
    }
    __syncthreads();
    sum = lane < NUM_WARPS ? reduce_smem[warp] : 0.0f;

    if(warp==0)
    {
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    }
    __syncthreads();
    if(tid==0)
    {
        atomicAdd(y,sum);
    }
}

template <const int NUM_THREADS=256/4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float *x, float *y, int N)
{
    int tid = threadIdx.x;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS+WarpSize-1)/warpSize;

    __shared__ float reduce_smem[NUM_WARPS];
    float4 reg_x = reinterpret_cast<float4*>(&x[index])[0];

    float sum = index<N ? (reg_x.x+reg_x.y+reg_x.z+reg_x.w) : 0.0f;

    int warp = tid/warpSize;
    int lane = tid%warpSize;

    sum = warp_reduce_sum_f32<warpSize>(sum);
    
    if(lane==0)
    {
        reduce_smem[warp] = sum;
    }
    __syncthreads();
    sum = lane<NUM_WARPS ? reduce_smem[lane] : 0.0f;
    if(warp == 0)
    {
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    }
    __syncthreads();
    if(tid==0)
    {
        atomicAdd(y,sum);
    }

}

template <int KWarpSize>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val)
{
    #pragma unroll
    for(int mask = KWarpSize>>1;mask>=1;mask>>=1)
    {
        val = __hadd(val,__shfl_xor_sync(0xffffffff,val,mask));
    }
    return val;
}

template <int KWarpSize>
__device__ __forceinline__ half warp_reduce_sum_f16_f32(half val)
{
    float val_f32 = __half2float(val);
    #pragma unroll
    for(int mask = KWarpSize>>1;mask>=1;mask>>=1)
    {
        val +=__shfl_xor_sync(0xffffffff,val_f32,mask);
    }
}

template <const int NUM_THREADS = 256>
__global__ void warp_reduce_sum_f16_f16_kernel(half *x, float *y, int N)
{
    int tid = threadIdx.x;
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS+WarpSize-1)/WarpSize;
    __shared__ float reduce_smem[NUM_WARPS];
    half sum_fp16 = index<N ? x[index] : __float2half(0.0f);

    int warp = tid/WarpSize;
    int lane = tid%WarpSize;

    sum_fp16 = warp_reduce_sum_f16_f16<WarpSize>(sum_fp16);
    if(lane==0)
    {
        reduce_smem[warp] = __half2float(sum_fp16);
    }
    __syncthreads();
    
    float sum = lane<NUM_WARPS ? reduce_smem[lane] : 0.0f;

    if(warp==0)
    {
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    }
    if(tid==0)
    {
        atomicAdd(y,sum);
    }

}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half *x,float *y,int N)
{
    int tid = threadIdx.x;
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS+WarpSize-1)/WarpSize;
    __shared__ float reduce_smem[NUM_WARPS];
    half sum_fp16 = index<N ? x[index] : __float2half(0.0f);

    int warp = tid/WarpSize;
    int lane = tid%WarpSize;

    float sum = warp_reduce_sum_f16_f32<WarpSize>(sum_fp16);

    if(lane==0)
    {
        reduce_smem[warp] = sum;
    }
    __syncthreads();

    sum = lane<NUM_WARPS ? reduce_smem[lane] : 0.0f;

    if(warp==0)
    {
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    }
    if(tid==0)
    {
        atomicAdd(y,sum);
    }
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half *x, float *y,int N)
{
    int tid = threadIdx.x;
    int index = (blockIdx.x*blockDim.x+threadIdx.x)*2;

    constexpr int NUM_WARPS = (NUM_THREADS+WarpSize-1)/WarpSize;
    __shared__ float reduce_smem[NUM_WARPS];

    half2 reg_x = reinterpret_cast<half2*>(&x[index])[0];
    half sum_f16 = index<N ? __hadd(reg_x.x,reg_x.y) : __float2half(0.0f);

    int lane = tid%WarpSize;
    int warp = tid/WarpSize;

    float sum = warp_reduce_sum_f16_f32<WarpSize>(sum_f16);

    if(lane==0)
    {
        reduce_smem[warp] = sum;
    }
    __syncthreads();

    sum = lane < NUM_WARPS ? reduce_smem[lane] : 0.0f;

    if(warp==0)
    {
        warp_reduce_sum_f32<NUM_WARPS>(sum);
    }
    if(tid==0)
    {
        atomicAdd(y,sum);
    }
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half *x,float *y,int N)
{
    int tid = threadIdx.x;
    int index = (blockIdx.x*blockDim.x+threadIdx.x)*2;

    constexpr int NUM_WARPS = (NUM_THREADS+WarpSize-1)/WarpSize;
    __shared__ float reduce_smem[NUM_WARPS];

    half2 reg_x = reinterpret_cast<half2*>(&x[index])[0];
    half sum_fp16 = __hadd(reg_x.x,reg_x.y);

    int lane = tid%WarpSize;
    int warp = tid/WarpSize;

    sum_fp16 = warp_reduce_sum_f16_f16<WarpSize>(sum_fp16);

    if(lane==0)
    {
        reduce_smem[warp] = __half2float(sum_fp16);
    }
    float sum = lane<NUM_WARPS ? reduce_smem[lane] : 0.0f;

    if(warp==0)
    {
        warp_reduce_sum_f32<NUM_WARPS>(sum);
    }
    if(tid==0)
    {
        atomicAdd(y,sum);
    }
    
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half *x, float *y, int N)
{
    int tid = threadIdx.x;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS+WarpSize-1)/WarpSize;
    __shared__ float reduce_smem[NUM_WARPS];

    half pack_x[8];
    reinterpret_cast<float4*>(&pack_x[0])[0] = reinterpret_cast<float4*>(&x[index])[0];
    
    half zero_fp16 = __float2half(0.0f);

    half sum_fp16 = zero_fp16;
    #pragma unroll
    for(int i = 0;i<8;++i)
    {
        sum_fp16+=((index+i)<N)?pack_x[i]:zero_fp16;
    }

    int lane = tid % WarpSize;
    int warp = tid / WarpSize;

    sum_fp16 = warp_reduce_sum_f16_f16<WarpSize>(sum_fp16);
    
    if(lane==0)
    {
        reduce_smem[warp] = __half2float(sum_fp16);
    }
    __syncthreads();

    float sum = lane<NUM_WARPS ? reduce_smem[lane] : 0.0f;

    if(warp==0)
    {
        warp_reduce_sum_f32<NUM_WARPS>(sum);
    }
    if(tid==0)
    {
        atomicAdd(y,sum);
    }
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(half *x, float *y, int N)
{
    int tid = threadIdx.x;
    int index = 8*(blockDim.x*blockIdx.x+threadIdx.x);
    constexpr int NUM_WARPS = (NUM_THREADS+WarpSize-1)/WarpSize;

    __shared__ float reduce_smem[NUM_WARPS];

    half pack_x[8];
    reinterpret_cast<float4*>(&pack_x[0])[0] = reinterpret_cast<float4*>(&x[index])[0];

    float sum_f32 = 0.0f;
    
    #pragma unroll
    for(int i =0;i<8;++i)
    {
        sum_f32 += ((index+i)<N) ? __half2float(pack_x[i]) : 0.0f;
    }

    sum_f32 = warp_reduce_sum_f32<WarpSize>(sum_f32);

    int lane = tid%WarpSize;
    int warp = tid/WarpSize;

    if(lane==0)
    {
        reduce_smem[warp] =  sum_f32;
    }
    __syncthreads();

    float sum = lane<NUM_WARPS ? reduce_smem[lane] : 0.0f;
    if(warp==0)
    {
        warp_reduce_sum_f32<NUM_WARPS>(sum);
    }
    if(tid==0)
    {
        atomicAdd(y,sum);
    }
}