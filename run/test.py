import sys
import os
import torch
import numpy as np

sys.path.append("/home/ubuntu/llm_ops/scripts")

import llm_ops

alpha = 1.0
beta = 0.0

M, N, K = 128, 128, 128
A = torch.randn(M, K, device='cuda', dtype=torch.float32).contiguous()
B = torch.randn(K, N, device='cuda', dtype=torch.float32).contiguous()
C = torch.zeros(M, N, device='cuda', dtype=torch.float32).contiguous()

A_torch = A.detach().clone()
B_torch = B.detach().clone()
C_torch = C.detach().clone()

C_torch = alpha * (A@B) + beta * C_torch

llm_ops.sgemm_naive(M, N, K, alpha, A, B, beta, C)

diff = (C-C_torch).abs().max()

print(f"the max diff is {diff}")
