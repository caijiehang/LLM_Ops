#include <cassert>
# include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sys/types.h>

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1D_tile(int M, int N, int K, float alpha, const float *A, const float *B,float beta, const float *C)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadcol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    A+=cRow*BM*K;
    B+=cCol*BN;
    C+=cRow*BM*N+cCol*BN;

    assert(BM*BK==blockDim.x);
    assert(BK*BN==blockDim.x);
    const int inner_RowA = threadIdx.x / BK;
    const int inner_ColA = threadIdx.x % BK;
    const int inner_RowB = threadIdx.x / BN;
    const int inner_ColB = threadIdx.x % BN;
    
    float result[TM];
    for(int bkidx = 0;bkidx<K;bkidx+=BK)
    {
        As[inner_RowA*BK+inner_ColA] = A[inner_RowA*K+inner_ColA];
        Bs[inner_RowB*BN+inner_ColB] = B[inner_RowB*N+inner_ColB];

        __syncthreads();

        A+=BK;
        B+=BK*N;

        for(int dotidx=0;dotidx<BK;++dotidx)
        {
            int Btmp = Bs[dotidx*BN+threadcol];
            for(int residx = 0;residx<TM;++residx)
            {
                result[residx] += As[(threadRow*TM+residx)*BK+dotidx]*Btmp;
            }
        }
        __syncthreads();
    }

    for(int residx = 0;residx<TM;++residx)
    {
        C[(threadRow*TM+residx)*N+threadcol] = alpha*result[residx] + beta * C[(threadRow*TM+residx)*N+threadcol];
    }
}