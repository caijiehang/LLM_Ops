"""GEMM (General Matrix Multiply) kernels"""

from typing import Annotated

import numpy
from numpy.typing import NDArray


def sgemm_naive(M: int, N: int, K: int, alpha: float, A: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], B: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], beta: float, C: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda')]) -> None:
    """A naive SGEMM implementation calling CUDA code"""

def sgemm_coalesce(M: int, N: int, K: int, alpha: float, A: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], B: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], beta: float, C: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda')]) -> None:
    """A SGEMM implementation coalesce global memory access calling CUDA code"""

def sgemm_sm(M: int, N: int, K: int, alpha: float, A: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], B: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], beta: float, C: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda')]) -> None:
    """Shared-memory optimized SGEMM CUDA kernel"""
