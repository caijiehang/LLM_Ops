import sys
import os
import torch
import torch.cuda.nvtx as nvtx
import numpy as np

sys.path.append("/home/ubuntu/llm_ops/scripts")

import llm_ops

def test_sgemm_naive():
    M, N, K = 4096,4096,4096
    alpha,beta = 0.8, 1

    A = torch.randn(M,K,dtype=torch.float32,device='cuda').contiguous()
    B = torch.randn(K,N,dtype=torch.float32,device='cuda').contiguous()
    C = torch.randn(M,N,dtype=torch.float32,device='cuda').contiguous()

    for _ in range(10):
        llm_ops.sgemm_naive(M,N,K,alpha,A,B,beta,C)
    
    print("Profiling...")

    nvtx.range_push("PyTorch_SGEMM")
    C_torch = alpha*(A@B)+beta*C
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    nvtx.range_push("Custom_Naive_SGEMM")
    llm_ops.sgemm_coalesce(M,N,K,alpha,A,B,beta,C)
    torch.cuda.synchronize()
    nvtx.range_pop()

    diff_max = (C-C_torch).abs().max()

    print(f"the max diff is {diff_max}")

if __name__=='__main__':
    test_sgemm_naive()
    # llm_ops.print_device_properties()

